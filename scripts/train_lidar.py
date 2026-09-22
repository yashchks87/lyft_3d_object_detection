"""LiDAR-only BEV detection training (bev_unet: segmentation; bev_centernet: CenterPoint head).

Streams samples from the MDS shards on the UC Volume, trains with DDP via
torchrun, evaluates the Lyft-style 3D mAP on the held-out val split every
epoch, and logs to Weights & Biases from the primary rank.

Single GPU:
    python scripts/train_lidar.py --cache /local_disk0/mds_cache \
        --out /local_disk0/runs/bev_unet_001 --wandb

Multi-GPU (one process per GPU):
    torchrun --standalone --nproc_per_node=4 scripts/train_lidar.py \
        --cache /local_disk0/mds_cache --out /local_disk0/runs/bev_unet_001 --wandb

Outputs under --out: config.json, metrics.jsonl, best.pt (highest val mAP),
last.pt (full resume state), optional epoch_*.pt, and _TRAINING_SUCCESS.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import signal
import sys
import tempfile
import time
from itertools import islice
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.special import expit

from scripts.box_ops import heatmap_to_boxes, masks_to_boxes
from scripts.lidar_data import (
    CLASS_NAMES,
    AugmentConfig,
    BEVConfig,
    LyftBEVDataset,
    collate_bev,
    load_class_stats,
    load_dataset_meta,
    validate_paths,
)
from scripts.lidar_metrics import evaluate_detections
from scripts.lidar_models import MODELS, CenterNetLoss, FocalDiceLoss, build_model

DEFAULT_MDS = Path('/Volumes/daai_ke_team/default/images/lyft_3d_object_detection/mds_shards')
DEFAULT_CLASS_STATS = Path(__file__).resolve().parents[1] / 'class_stats.json'


# --------------------------------------------------------------------------- #
# Distributed plumbing
# --------------------------------------------------------------------------- #
class Cluster:
    def __init__(self, rank=0, world_size=1, local_rank=0):
        self.rank, self.world_size, self.local_rank = rank, world_size, local_rank

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def primary(self):
        return self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def sum(self, values, device):
        if not self.distributed:
            return list(values)
        tensor = torch.tensor(list(values), dtype=torch.float64, device=device)
        dist.all_reduce(tensor)
        return tensor.tolist()

    def gather(self, rows):
        if not self.distributed:
            return list(rows)
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, list(rows))
        return [row for part in gathered for row in part]


def cluster_from_environment() -> Cluster:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size < 1 or not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise ValueError('Inconsistent WORLD_SIZE/RANK/LOCAL_RANK; launch one process or use torchrun.')
    if world_size > 1 and not {'MASTER_ADDR', 'MASTER_PORT'} <= set(os.environ):
        raise ValueError('Distributed training needs MASTER_ADDR/MASTER_PORT; launch with torchrun.')
    return Cluster(rank, world_size, local_rank)


def resolve_device(cluster: Cluster, device_arg: str) -> torch.device:
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if device_arg == 'auto' else device_arg)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; pass --device cpu or use a GPU node.')
        index = cluster.local_rank if cluster.distributed else (device.index or 0)
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {index} unavailable on this host.')
        device = torch.device('cuda', index)
        torch.cuda.set_device(device)
    elif device.type != 'cpu':
        raise ValueError('--device must select cpu or cuda.')
    return device


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mds', type=Path, default=DEFAULT_MDS,
                        help='Root of the MDS shards (contains train/, val/, dataset_meta.json).')
    parser.add_argument('--cache', type=Path, required=True,
                        help='Node-local shard cache; never on /Volumes. Shared by all ranks.')
    parser.add_argument('--out', type=Path, required=True,
                        help='New run directory under an existing parent.')
    parser.add_argument('--model', choices=MODELS, default='bev_unet')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Per-GPU batch size. Measured on A10G (23GB) at the default BEV grid: '
                             'bs=8 peaks at ~19.4GB with DDP headroom; bs=10+ risks OOM.')
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Peak LR at the reference 16-sample global batch; scaled linearly.')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-fraction', type=float, default=0.02,
                        help='Fraction of optimizer steps spent in linear warmup before cosine decay.')
    parser.add_argument('--clip-grad', type=float, default=5.0)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--bev-range', type=float, default=100.0, help='BEV half-extent, metres.')
    parser.add_argument('--bev-resolution', type=float, default=0.25, help='Metres per pixel.')
    parser.add_argument('--z-min', type=float, default=-2.5)
    parser.add_argument('--z-max', type=float, default=5.5)
    parser.add_argument('--z-bins', type=int, default=8)
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='bev_unet only: focal loss exponent.')
    parser.add_argument('--focal-alpha', type=float, default=0.75,
                        help='bev_unet only: focal loss positive-class weight.')
    parser.add_argument('--dice-weight', type=float, default=1.0,
                        help='bev_unet only: dice term weight.')
    parser.add_argument('--reg-weight', type=float, default=1.0,
                        help='bev_centernet only: L1 box-regression weight against the heatmap focal term.')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                        help='Validation decode threshold: mask binarisation for bev_unet; heatmap '
                             'peak confidence for bev_centernet (0.1 recommended there).')
    parser.add_argument('--class-stats', type=Path,
                        default=DEFAULT_CLASS_STATS if DEFAULT_CLASS_STATS.is_file() else None,
                        help='class_stats.json from scripts/compute_class_stats.py; '
                             'falls back to built-in constants when absent.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Loader workers per rank; BEV rasterization is CPU-bound, so keep '
                             'ranks x workers comfortably below the core count.')
    parser.add_argument('--cache-limit', default=None,
                        help="Optional streaming cache budget, e.g. '200gb'.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', type=Path, help='last.pt from a previous run to continue.')
    parser.add_argument('--save-epochs', action=argparse.BooleanOptionalAction, default=False,
                        help='Keep a weights checkpoint per epoch besides best.pt/last.pt.')
    parser.add_argument('--val-every', type=int, default=1, help='Validate every N epochs.')
    parser.add_argument('--val-max-batches', type=int, help='Debug-only validation batch cap per rank.')
    parser.add_argument('--max-train-batches', type=int, help='Debug-only training batch cap per rank.')
    parser.add_argument('--log-every', type=int, default=50,
                        help='Step-level wandb train loss logging interval, in optimizer steps.')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--wandb', action='store_true',
                        help='Log metrics, config, and system stats to Weights & Biases (primary rank only).')
    parser.add_argument('--wandb-project', default='Lyft 3D object detection')
    parser.add_argument('--wandb-entity', default='yashchks87')
    parser.add_argument('--wandb-run-name', default=None, help='Defaults to the output directory name.')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    parser.add_argument('--wandb-tags', nargs='*', default=[], metavar='TAG')
    parser.add_argument('--terminate-cluster', action='store_true',
                        help='Terminate (stop, not delete) this Databricks cluster after a fully '
                             'successful run, so no idle charges accrue. Crashes and Ctrl-C always '
                             'leave the cluster running for debugging and --resume. '
                             'Requires DATABRICKS_HOST/TOKEN/CLUSTER_ID.')
    return parser


def validate_args(args) -> Cluster:
    for name in ('epochs', 'batch_size', 'grad_accum', 'val_every', 'log_every',
                 'base_channels', 'depth', 'z_bins'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    for name in ('lr', 'clip_grad', 'bev_range', 'bev_resolution'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and positive.')
    for name in ('weight_decay', 'reg_weight'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and nonnegative.')
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError('--warmup-fraction must be in [0, 1).')
    if not 0 < args.score_threshold < 1:
        raise ValueError('--score-threshold must be in (0, 1).')
    if args.z_max <= args.z_min:
        raise ValueError('--z-max must exceed --z-min.')
    if args.num_workers < 0:
        raise ValueError('--num-workers must be nonnegative.')
    for name in ('val_max_batches', 'max_train_batches'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    size = 2 * args.bev_range / args.bev_resolution
    if abs(size - round(size)) > 1e-6 or round(size) % 2 ** args.depth:
        raise ValueError(f'BEV grid size {size:.2f} must be an integer divisible by '
                         f'2^depth = {2 ** args.depth} for the U-Net.')
    if args.resume is not None and not args.resume.is_file():
        raise ValueError(f'--resume checkpoint does not exist: {args.resume}')
    cluster = cluster_from_environment()
    if cluster.distributed and args.device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Distributed runs select the GPU from LOCAL_RANK; use --device auto.')
    return cluster


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def train_epoch(model, loader, device, *, criterion, optimizer, scheduler, scaler, amp,
                amp_dtype, grad_accum, clip_grad, max_batches, cluster, description,
                no_progress, run, log_every, global_step):
    model.train()
    batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    if batches < 1:
        raise ValueError('Cannot train with an empty loader.')
    total_loss = observed = updates = 0
    optimizer.zero_grad(set_to_none=True)
    with tqdm(islice(loader, batches), total=batches, desc=description, unit='batch',
              file=sys.stderr, mininterval=2.0, dynamic_ncols=True,
              disable=no_progress or not cluster.primary) as progress:
        for step, batch in enumerate(progress):
            images = batch['images'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            boundary = (step + 1) % grad_accum == 0 or step + 1 == batches
            accumulating = not boundary and isinstance(model, DistributedDataParallel)
            with model.no_sync() if accumulating else contextlib.nullcontext():
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    loss = criterion(model(images), targets) / grad_accum
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss; check learning rate and inputs.')
                scaler.scale(loss).backward()
            total_loss += loss.detach().item() * grad_accum
            observed += 1
            if boundary:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad,
                                         error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    updates += 1
                    global_step += 1
                    if run is not None and updates % log_every == 0:
                        run.log({'train/loss_step': total_loss / observed,
                                 'train/lr': optimizer.param_groups[0]['lr']}, step=global_step)
            progress.set_postfix(loss=f'{total_loss / observed:.4f}', refresh=False)
    total_loss, observed = cluster.sum([total_loss, observed], device)
    return {'loss': total_loss / observed, 'batches': int(observed),
            'optimizer_steps': updates, 'global_step': global_step}


@torch.no_grad()
def validate_epoch(model, loader, device, *, criterion, amp, amp_dtype, decode,
                   max_batches, cluster, description, no_progress):
    model.eval()
    batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    if batches < 1:
        raise ValueError('Cannot validate with an empty loader.')
    total_loss = observed = 0
    rows = []
    with tqdm(islice(loader, batches), total=batches, desc=description, unit='batch',
              file=sys.stderr, mininterval=2.0, dynamic_ncols=True,
              disable=no_progress or not cluster.primary) as progress:
        for batch in progress:
            images = batch['images'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = model(images)
                total_loss += criterion(logits, targets).item()
            observed += 1
            outputs = logits.float().cpu().numpy()
            for i, token in enumerate(batch['sample_tokens']):
                boxes, classes, scores = decode(outputs[i])
                rows.append((token,
                             {'boxes': boxes, 'classes': classes, 'scores': scores},
                             {'boxes': batch['gt_boxes'][i], 'classes': batch['gt_classes'][i]}))
    total_loss, observed = cluster.sum([total_loss, observed], device)
    unique = {}
    for token, prediction, truth in cluster.gather(rows):
        unique.setdefault(token, (prediction, truth))
    metrics = None
    if cluster.primary:
        predictions = [prediction for prediction, _ in unique.values()]
        truths = [truth for _, truth in unique.values()]
        metrics = evaluate_detections(predictions, truths, len(CLASS_NAMES))
        metrics['per_class'] = {CLASS_NAMES[cls]: ap for cls, ap in metrics['per_class'].items()}
    return {'loss': total_loss / observed, 'samples': len(unique), 'metrics': metrics}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class CheckpointPublisher:
    """Rank-0 writer that never blocks training on /Volumes (FUSE) I/O.

    torch.save lands on node-local staging (fast), and a single background
    thread copies each file to its destination in submission order. A stalled
    Volume mount therefore only delays checkpoint durability -- it can no
    longer keep rank 0 out of the next collective past the NCCL timeout,
    which is exactly how a previous run died. close() drains the backlog and
    reports any files that never reached their destination.
    """

    def __init__(self) -> None:
        self._staging = Path(tempfile.mkdtemp(prefix='checkpoint_staging_'))
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = {}
        self._sequence = 0
        self._buffers = {}

    def save(self, destination: Path, checkpoint: dict) -> None:
        self._sequence += 1
        staged = self._staging / f'{self._sequence:06d}_{destination.name}'
        with staged.open('wb') as file:
            torch.save(checkpoint, file)
        self._pending[destination] = self._executor.submit(self._publish, staged, destination)

    def append(self, destination: Path, line: str) -> None:
        """Append a line by rewriting the whole file through staging.

        UC Volumes (FUSE) raise OSError(29, 'Illegal seek') when opening an
        existing file in append mode, so the full content is accumulated in
        memory (seeded from the destination on first use, e.g. after resume),
        staged locally, and published with the same copy-and-replace path
        that checkpoints use.
        """
        buffer = self._buffers.get(destination)
        if buffer is None:
            buffer = self._buffers[destination] = []
            if destination.exists():
                buffer.append(destination.read_text(encoding='utf-8'))
        buffer.append(line)
        self._sequence += 1
        staged = self._staging / f'{self._sequence:06d}_{destination.name}'
        staged.write_text(''.join(buffer), encoding='utf-8')
        self._pending[destination] = self._executor.submit(self._publish, staged, destination)

    @staticmethod
    def _publish(staged: Path, destination: Path) -> None:
        temporary = destination.with_name(destination.name + '.tmp')
        try:
            shutil.copyfile(staged, temporary)
            os.replace(temporary, destination)
        finally:
            staged.unlink(missing_ok=True)

    def close(self, timeout_per_file: float = 1800) -> list[str]:
        """Wait for queued publications; returns descriptions of any failures."""
        self._executor.shutdown(wait=False)
        failures = []
        for destination, future in self._pending.items():
            try:
                future.result(timeout=timeout_per_file)
            except Exception as error:  # noqa: BLE001 -- collected for the caller
                failures.append(f'{destination}: {error!r}')
        shutil.rmtree(self._staging, ignore_errors=True)
        return failures


def _raise_keyboard_interrupt(signum, frame) -> None:  # noqa: ARG001 -- signal handler
    """Turn SIGTERM into the interrupt path so `finally` blocks still run."""
    raise KeyboardInterrupt(f'received signal {signum}')


def terminate_cluster() -> None:
    """Request termination (stop, NOT delete) of the current Databricks cluster."""
    import requests

    host = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
    token = os.environ.get('DATABRICKS_TOKEN', '')
    cluster_id = os.environ.get('DATABRICKS_CLUSTER_ID', '')
    if not (host and token and cluster_id):
        print('Cannot terminate cluster: DATABRICKS_HOST/TOKEN/CLUSTER_ID env vars are missing.',
              file=sys.stderr, flush=True)
        return
    print(f'Requesting termination of cluster {cluster_id} ...', flush=True)
    for attempt in range(1, 6):
        try:
            response = requests.post(f'{host}/api/2.1/clusters/delete',
                                     headers={'Authorization': f'Bearer {token}'},
                                     json={'cluster_id': cluster_id}, timeout=30)
            if response.ok:
                print('Cluster termination requested; this machine will stop shortly.', flush=True)
                return
            print(f'Terminate attempt {attempt}/5 failed: HTTP {response.status_code} '
                  f'{response.text[:500]}', file=sys.stderr, flush=True)
        except requests.RequestException as error:
            print(f'Terminate attempt {attempt}/5 failed: {error}', file=sys.stderr, flush=True)
        time.sleep(10 * attempt)
    print('Failed to terminate the cluster after 5 attempts; terminate it manually.',
          file=sys.stderr, flush=True)


def wandb_init(args, config, cluster):
    if not args.wandb or not cluster.primary:
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError('wandb is unavailable; pip install wandb or omit --wandb.') from error
    return wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                      name=args.wandb_run_name or args.out.name, mode=args.wandb_mode,
                      tags=args.wandb_tags or None, config=config)


def build_loader(dataset, args, device):
    worker_kwargs = ({'multiprocessing_context': 'spawn', 'persistent_workers': True,
                      'prefetch_factor': 2} if args.num_workers else {})
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=device.type == 'cuda', collate_fn=collate_bev,
                      drop_last=False, **worker_kwargs)


def train(args) -> list[dict]:
    cluster = validate_args(args)
    device = resolve_device(cluster, args.device)
    if cluster.distributed:
        # A wedged /Volumes mount once held rank 0 in write I/O past NCCL's
        # 10-minute default, aborting the run; give collectives generous slack.
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo',
                                device_id=device if device.type == 'cuda' else None,
                                timeout=timedelta(hours=2))
    try:
        return _train(args, cluster, device)
    finally:
        if cluster.distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(args, cluster: Cluster, device: torch.device) -> list[dict]:
    remote, cache = validate_paths(str(args.mds), str(args.cache))
    output = args.out.resolve()
    if output.exists() and args.resume is None:
        raise FileExistsError(f'Run output already exists; choose a new directory: {output}')
    if not output.parent.is_dir():
        raise ValueError(f'Run output parent must already exist: {output.parent}')
    for path in (remote, cache):
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('Run output, dataset, and cache must be separate, non-nested directories.')
    random.seed(args.seed + cluster.rank)
    np.random.seed((args.seed + cluster.rank) % 2 ** 32)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    dataset_meta = load_dataset_meta(remote)
    class_stats = load_class_stats(args.class_stats)
    bev = BEVConfig(xy_range=args.bev_range, resolution=args.bev_resolution,
                    z_min=args.z_min, z_max=args.z_max, z_bins=args.z_bins)
    head = 'center' if args.model == 'bev_centernet' else 'mask'
    streaming_kwargs = dict(cache_limit=args.cache_limit, shuffle_seed=args.seed)
    train_dataset = LyftBEVDataset(remote=str(remote / 'train'), local=str(cache / 'train'),
                                   bev=bev, training=True, augment=AugmentConfig(), head=head,
                                   batch_size=args.batch_size, **streaming_kwargs)
    val_dataset = LyftBEVDataset(remote=str(remote / 'val'), local=str(cache / 'val'),
                                 bev=bev, training=False, head=head,
                                 batch_size=args.batch_size, **streaming_kwargs)
    train_loader = build_loader(train_dataset, args, device)
    val_loader = build_loader(val_dataset, args, device)

    model = build_model(args.model, in_channels=bev.channels, num_classes=len(CLASS_NAMES),
                        base_channels=args.base_channels, depth=args.depth).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if cluster.distributed:
        model = DistributedDataParallel(model, device_ids=[device.index]
                                        if device.type == 'cuda' else None)
    raw_model = model.module if cluster.distributed else model
    criterion = (CenterNetLoss(reg_weight=args.reg_weight) if head == 'center'
                 else FocalDiceLoss(gamma=args.focal_gamma, alpha=args.focal_alpha,
                                    dice_weight=args.dice_weight))
    global_batch = args.batch_size * args.grad_accum * cluster.world_size
    learning_rate = args.lr * global_batch / 16
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=learning_rate,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(
        (len(train_loader) if args.max_train_batches is None
         else min(len(train_loader), args.max_train_batches)) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp = args.amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=amp and amp_dtype == torch.float16)

    start_epoch, best_map, global_step = 1, -math.inf, 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint.get('best_map', -math.inf)
        global_step = checkpoint.get('global_step', 0)

    if head == 'center':
        def decode(output: np.ndarray):
            return heatmap_to_boxes(expit(output[:len(CLASS_NAMES)]), output[len(CLASS_NAMES):],
                                    bev.grid, threshold=args.score_threshold)
    else:
        def decode(output: np.ndarray):
            return masks_to_boxes(expit(output), bev.grid, class_stats, CLASS_NAMES,
                                  threshold=args.score_threshold)
    config = {
        'model': args.model,
        'head': head,
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'dataset': {'path': str(remote), 'splits': dataset_meta['splits'],
                    'created_utc': dataset_meta.get('created_utc')},
        'bev': {field: getattr(bev, field) for field in bev.__dataclass_fields__},
        'bev_grid_size': bev.grid.size,
        'input_channels': bev.channels,
        'class_names': CLASS_NAMES,
        'class_stats': class_stats,
        'class_stats_source': str(args.class_stats) if args.class_stats else 'fallback',
        'parameters': parameters,
        'torch_version': str(torch.__version__),
        'device': str(device),
        'world_size': cluster.world_size,
        'effective_batch': global_batch,
        'learning_rate_scaled': learning_rate,
        'amp_dtype': str(amp_dtype) if amp else None,
        'steps_per_epoch': steps_per_epoch,
        'train_samples_per_rank': len(train_dataset),
        'val_samples_per_rank': len(val_dataset),
    }
    run = None
    cluster.barrier()
    if cluster.primary:
        output.mkdir(exist_ok=args.resume is not None)
        with (output / 'config.json').open('w', encoding='utf-8') as file:
            json.dump(config, file, indent=2, allow_nan=False)
            file.write('\n')
        print(json.dumps({'event': 'setup', **config}, allow_nan=False), flush=True)
        run = wandb_init(args, config, cluster)
    cluster.barrier()

    history = []
    publisher = CheckpointPublisher() if cluster.primary else None
    failures = []
    completed = False
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            start = time.perf_counter()
            training = train_epoch(
                model, train_loader, device, criterion=criterion, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, amp=amp, amp_dtype=amp_dtype,
                grad_accum=args.grad_accum, clip_grad=args.clip_grad,
                max_batches=args.max_train_batches, cluster=cluster,
                description=f'Epoch {epoch}/{args.epochs} train', no_progress=args.no_progress,
                run=run, log_every=args.log_every, global_step=global_step)
            global_step = training['global_step']
            result = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                      'train_loss': training['loss'], 'optimizer_steps': training['optimizer_steps']}
            validate_now = epoch % args.val_every == 0 or epoch == args.epochs
            improved = False
            if validate_now:
                validation = validate_epoch(
                    model, val_loader, device, criterion=criterion, amp=amp, amp_dtype=amp_dtype,
                    decode=decode, max_batches=args.val_max_batches, cluster=cluster,
                    description=f'Epoch {epoch}/{args.epochs} validation', no_progress=args.no_progress)
                result.update(validation_loss=validation['loss'], validation_samples=validation['samples'])
                if cluster.primary:
                    metrics = validation['metrics']
                    result['validation_map'] = metrics['map']
                    result['validation_per_class'] = metrics['per_class']
                    result['validation_per_threshold'] = {str(threshold): ap for threshold, ap
                                                          in metrics['per_threshold'].items()}
                    improved = metrics['map'] > best_map
                    best_map = max(best_map, metrics['map'])
            result['elapsed_seconds'] = time.perf_counter() - start
            history.append(result)
            if cluster.primary:
                if args.save_epochs:
                    publisher.save(output / f'epoch_{epoch:03d}.pt', {
                        'model_state': raw_model.state_dict(), 'epoch': epoch,
                        'metrics': result, 'config': config})
                if improved:
                    publisher.save(output / 'best.pt', {
                        'model_state': raw_model.state_dict(), 'epoch': epoch,
                        'metrics': result, 'config': config})
                publisher.save(output / 'last.pt', {
                    'model_state': raw_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'epoch': epoch, 'global_step': global_step, 'best_map': best_map,
                    'metrics': result, 'config': config})
                publisher.append(output / 'metrics.jsonl',
                                 json.dumps(result, allow_nan=False) + '\n')
                print(json.dumps({'event': 'epoch', **result}, allow_nan=False), flush=True)
                if run is not None:
                    logged = {'epoch': epoch, 'lr': result['lr'], 'train/loss': result['train_loss'],
                              'epoch/elapsed_seconds': result['elapsed_seconds']}
                    if validate_now:
                        logged.update({'val/loss': result['validation_loss'],
                                       'val/map': result['validation_map'],
                                       'val/best_map': best_map})
                        logged.update({f'val/ap_{name}': ap for name, ap
                                       in result['validation_per_class'].items()})
                    run.log(logged, step=global_step)
            cluster.barrier()
        completed = True
    finally:
        if publisher is not None:
            failures = publisher.close()
            for failure in failures:
                print(f'Checkpoint publication failed: {failure}', file=sys.stderr, flush=True)
        # An interrupted or failed run still has to be closed out on W&B with the
        # progress it did make; otherwise it is left dangling as "Crashed" with
        # no summary, which is how a signalled run shows up today.
        if not completed and run is not None:
            if math.isfinite(best_map):  # still -inf if nothing validated yet
                run.summary['best_map'] = best_map
            run.finish(exit_code=1)
    if failures:
        raise RuntimeError(f'{len(failures)} run file(s) never reached {output}; see stderr above.')
    if cluster.primary:
        with (output / '_TRAINING_SUCCESS').open('w', encoding='utf-8') as file:
            json.dump({'epochs': args.epochs, 'best_map': best_map,
                       'world_size': cluster.world_size}, file, allow_nan=False)
            file.write('\n')
        if run is not None:
            run.summary['best_map'] = best_map
            run.finish()
    return history


def main():
    parser = build_parser()
    args = parser.parse_args()
    primary = os.environ.get('RANK', '0') == '0'
    # Python kills the interpreter on SIGTERM without unwinding, so a signalled
    # run would drop staged checkpoints on the floor and leave W&B dangling.
    # torchrun relays SIGTERM to every rank, so all ranks unwind together and no
    # collective is left half-entered.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        train(args)
    except KeyboardInterrupt:
        parser.exit(130, 'Training interrupted. Checkpoints so far remain in the run directory. '
                         'Cluster left running; relaunch with --resume <out>/last.pt.\n')
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Training failed: {error}\n'
                       'Cluster left running; relaunch with --resume <out>/last.pt once fixed.\n')
    if args.terminate_cluster and primary:
        terminate_cluster()


if __name__ == '__main__':
    main()
