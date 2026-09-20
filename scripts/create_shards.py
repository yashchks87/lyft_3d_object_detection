#!/usr/bin/env python
"""Generate MosaicML Streaming (MDS) shards for the Lyft 3D Object Detection dataset.

Reads the raw nuScenes-style dataset from a UC Volume, converts every keyframe
sample into a self-contained training record (ego-frame points, ego-frame GT
boxes, ego pose, camera calibration refs), and writes MDS shards suitable for
`streaming.StreamingDataset`.

Output layout (under --out-root):
    train/  index.json + group_*/shard.*.mds     (scene-stratified)
    val/    index.json + group_*/shard.*.mds     (scene-stratified holdout)
    test/   index.json + group_*/shard.*.mds     (no GT)
    dataset_meta.json

Record schema (MDS columns):
    sample_token      str                 join key for submissions/debugging
    scene_token       str                 scene-level grouping
    points            ndarray:float32     [N, 5] ego-frame (x, y, z, intensity, ring)
    points_sensor_id  ndarray:uint8       [N] index into meta['lidars']
    gt_boxes          bytes               [M, 7] float32 ego-frame (cx, cy, cz, w, l, h, yaw);
                                          decode: np.frombuffer(b, np.float32).reshape(-1, 7)
    gt_classes        bytes               [M] int8 index into CLASS_NAMES;
                                          decode: np.frombuffer(b, np.int8)
    ego_pose          ndarray:float64     [4, 4] ego -> world transform
    meta              json                cameras/lidars calibration, timestamps, names

Design notes:
    - Work is split into fixed-size sample groups; a process pool converts groups
      in parallel (I/O bound: reads lidar bins from the Volume FUSE mount).
    - Each worker writes its group to fast local disk, then copies it to the
      Volume with size verification (index.json copied last as a completion
      marker), then deletes the local staging copy.
    - Restart-safe: groups already complete on the Volume are skipped.
    - Group indexes are merged into one index.json per split via
      `streaming.base.util.merge_index`.
    - With --terminate-cluster, the Databricks cluster is terminated (NOT
      deleted) after everything succeeds.

Usage (smoke test):
    python create_shards.py --limit 20 --splits train,val --out-root /tmp/mds_smoke

Full run with auto-terminate (see run_sharding.sh):
    nohup python -u create_shards.py --terminate-cluster > shard.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np

LOG = logging.getLogger("create_shards")

CLASS_NAMES = [
    "car", "pedestrian", "animal", "other_vehicle", "bus",
    "motorcycle", "truck", "emergency_vehicle", "bicycle",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

MDS_COLUMNS = {
    "sample_token": "str",
    "scene_token": "str",
    "points": "ndarray:float32",
    "points_sensor_id": "ndarray:uint8",
    # bytes (not ndarray) because streaming's ndarray codec rejects empty
    # arrays, and the test split has no ground truth.
    "gt_boxes": "bytes",
    "gt_classes": "bytes",
    "ego_pose": "ndarray:float64",
    "meta": "json",
}

# Set once in the parent process before the worker pool is forked; workers
# access it read-only through copy-on-write memory (no pickling of task data).
_WORKER_CTX: dict = {}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def quat_to_rot(q: list[float]) -> np.ndarray:
    """nuScenes-style quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_transform(rotation_quat: list[float], translation: list[float]) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = quat_to_rot(rotation_quat)
    tf[:3, 3] = translation
    return tf


# --------------------------------------------------------------------------- #
# Raw dataset loading / task building
# --------------------------------------------------------------------------- #
@dataclass
class SampleTask:
    """Everything a worker needs to emit one MDS record (no big tables)."""

    sample_token: str
    scene_token: str
    scene_name: str
    host: str
    timestamp: float
    ego_pose: np.ndarray                     # [4, 4] ego -> world
    ego_translation: list[float]
    ego_rotation: list[float]
    lidars: list[dict] = field(default_factory=list)   # channel, path, R (3x3), t (3)
    cameras: list[dict] = field(default_factory=list)  # calibration + file refs
    gt_boxes: np.ndarray = None              # [M, 7] float32, ego frame
    gt_classes: np.ndarray = None            # [M] int8
    gt_names: list[str] = field(default_factory=list)


def load_table(json_dir: Path, name: str):
    path = json_dir / f"{name}.json"
    LOG.info("Loading %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    with open(path) as f:
        return json.load(f)


def resolve_data_path(data_root: Path, raw_split: str, filename: str) -> str:
    """Map a table filename like 'lidar/x.bin' to '<root>/<split>_lidar/x.bin'."""
    return str(data_root / f"{raw_split}_{filename}")


def build_tasks(data_root: Path, raw_split: str) -> list[SampleTask]:
    """Parse the raw JSON tables into one lightweight task per keyframe sample."""
    json_dir = data_root / f"{raw_split}_data"
    sensors = {s["token"]: s for s in load_table(json_dir, "sensor")}
    calibs = {c["token"]: c for c in load_table(json_dir, "calibrated_sensor")}
    ego_poses = {e["token"]: e for e in load_table(json_dir, "ego_pose")}
    scenes = {s["token"]: s for s in load_table(json_dir, "scene")}
    samples = load_table(json_dir, "sample")

    sds_by_sample: dict[str, list] = {}
    for sd in load_table(json_dir, "sample_data"):
        if sd["is_key_frame"]:
            sds_by_sample.setdefault(sd["sample_token"], []).append(sd)

    anns_by_sample: dict[str, list] = {}
    has_gt = (json_dir / "sample_annotation.json").exists()
    if has_gt:
        categories = {c["token"]: c["name"] for c in load_table(json_dir, "category")}
        instance_cat = {
            i["token"]: categories[i["category_token"]]
            for i in load_table(json_dir, "instance")
        }
        for ann in load_table(json_dir, "sample_annotation"):
            anns_by_sample.setdefault(ann["sample_token"], []).append(ann)

    tasks: list[SampleTask] = []
    n_skipped = 0
    for sample in samples:
        token = sample["token"]
        scene = scenes[sample["scene_token"]]
        sds = sds_by_sample.get(token, [])
        lidar_sds = [sd for sd in sds if sensors[calibs[sd["calibrated_sensor_token"]]["sensor_token"]]["modality"] == "lidar"]
        cam_sds = [sd for sd in sds if sensors[calibs[sd["calibrated_sensor_token"]]["sensor_token"]]["modality"] == "camera"]
        if not lidar_sds:
            LOG.warning("Sample %s has no keyframe lidar data; skipping", token)
            n_skipped += 1
            continue

        def channel(sd):
            return sensors[calibs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]

        lidar_sds.sort(key=lambda sd: (channel(sd) != "LIDAR_TOP", channel(sd)))
        ref_ego = ego_poses[lidar_sds[0]["ego_pose_token"]]
        r_ego = quat_to_rot(ref_ego["rotation"])
        t_ego = np.asarray(ref_ego["translation"])

        lidars = []
        for sd in lidar_sds:
            cs = calibs[sd["calibrated_sensor_token"]]
            lidars.append(
                {
                    "channel": channel(sd),
                    "filename": sd["filename"],
                    "path": resolve_data_path(data_root, raw_split, sd["filename"]),
                    "sensor2ego_rotation": cs["rotation"],
                    "sensor2ego_translation": cs["translation"],
                    "timestamp": sd["timestamp"],
                }
            )
        cameras = []
        for sd in sorted(cam_sds, key=channel):
            cs = calibs[sd["calibrated_sensor_token"]]
            cameras.append(
                {
                    "channel": channel(sd),
                    "filename": sd["filename"],
                    "width": sd.get("width"),
                    "height": sd.get("height"),
                    "camera_intrinsic": cs.get("camera_intrinsic"),
                    "sensor2ego_rotation": cs["rotation"],
                    "sensor2ego_translation": cs["translation"],
                    "timestamp": sd["timestamp"],
                }
            )

        boxes, class_idxs, names = [], [], []
        for ann in anns_by_sample.get(token, []):
            name = instance_cat[ann["instance_token"]]
            center = quat_to_rot(ref_ego["rotation"]).T @ (np.asarray(ann["translation"]) - t_ego)
            r_rel = r_ego.T @ quat_to_rot(ann["rotation"])
            yaw = math.atan2(r_rel[1, 0], r_rel[0, 0])
            w, l, h = ann["size"]
            boxes.append([*center, w, l, h, yaw])
            class_idxs.append(CLASS_TO_IDX[name])
            names.append(name)

        tasks.append(
            SampleTask(
                sample_token=token,
                scene_token=sample["scene_token"],
                scene_name=scene["name"],
                host=scene["name"].split("-lidar")[0],
                timestamp=sample["timestamp"],
                ego_pose=make_transform(ref_ego["rotation"], ref_ego["translation"]),
                ego_translation=ref_ego["translation"],
                ego_rotation=ref_ego["rotation"],
                lidars=lidars,
                cameras=cameras,
                gt_boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
                gt_classes=np.asarray(class_idxs, dtype=np.int8),
                gt_names=names,
            )
        )

    # Deterministic order: by scene then time, so shards have temporal locality.
    tasks.sort(key=lambda t: (t.scene_name, t.timestamp))
    LOG.info("Built %d tasks for raw split '%s' (%d skipped)", len(tasks), raw_split, n_skipped)
    return tasks


def split_train_val(tasks: list[SampleTask], val_frac: float) -> tuple[list, list]:
    """Scene-level split, stratified by host so val covers every vehicle."""
    scenes_by_host: dict[str, list[str]] = {}
    for t in tasks:
        host_scenes = scenes_by_host.setdefault(t.host, [])
        if t.scene_token not in host_scenes:
            host_scenes.append(t.scene_token)

    val_scenes = set()
    for host in sorted(scenes_by_host):
        scene_list = sorted(scenes_by_host[host])
        n_val = max(1, round(len(scene_list) * val_frac))
        # Evenly spaced picks: deterministic and spread over time.
        step = len(scene_list) / n_val
        val_scenes.update(scene_list[min(int(i * step), len(scene_list) - 1)] for i in range(n_val))

    train = [t for t in tasks if t.scene_token not in val_scenes]
    val = [t for t in tasks if t.scene_token in val_scenes]
    LOG.info(
        "Scene split: %d train scenes / %d val scenes -> %d train / %d val samples",
        len({t.scene_token for t in train}), len(val_scenes), len(train), len(val),
    )
    return train, val


# --------------------------------------------------------------------------- #
# Shard writing (worker side)
# --------------------------------------------------------------------------- #
def load_sample_points(task: SampleTask) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load all lidars for a sample, transform into the ego frame, and merge."""
    clouds, sensor_ids, counts = [], [], []
    for i, lidar in enumerate(task.lidars):
        arr = np.fromfile(lidar["path"], dtype=np.float32)
        if arr.size == 0 or arr.size % 5 != 0:
            LOG.warning(
                "Corrupt lidar file (%d float32s, not divisible by 5): %s -- skipping file",
                arr.size, lidar["path"],
            )
            counts.append(0)
            continue
        pts = arr.reshape(-1, 5)
        rot = quat_to_rot(lidar["sensor2ego_rotation"])
        pts[:, :3] = pts[:, :3] @ rot.T + np.asarray(lidar["sensor2ego_translation"], dtype=np.float32)
        clouds.append(pts)
        sensor_ids.append(np.full(len(pts), i, dtype=np.uint8))
        counts.append(len(pts))
    if not clouds:
        raise RuntimeError(f"No readable lidar data for sample {task.sample_token}")
    return np.concatenate(clouds), np.concatenate(sensor_ids), counts


def task_to_record(task: SampleTask) -> dict:
    points, sensor_ids, counts = load_sample_points(task)
    meta = {
        "scene_name": task.scene_name,
        "host": task.host,
        "timestamp": task.timestamp,
        "ego_translation": task.ego_translation,
        "ego_rotation": task.ego_rotation,
        "lidars": [
            {k: v for k, v in dict(lidar, num_points=n).items() if k != "path"}
            for lidar, n in zip(task.lidars, counts)
        ],
        "cameras": task.cameras,
        "gt_names": task.gt_names,
    }
    return {
        "sample_token": task.sample_token,
        "scene_token": task.scene_token,
        "points": np.ascontiguousarray(points, dtype=np.float32),
        "points_sensor_id": sensor_ids,
        "gt_boxes": np.ascontiguousarray(task.gt_boxes).tobytes(),
        "gt_classes": np.ascontiguousarray(task.gt_classes).tobytes(),
        "ego_pose": task.ego_pose,
        "meta": meta,
    }


def group_is_complete(volume_group_dir: Path) -> bool:
    """A group is done iff its index.json exists and lists files of the right size."""
    index_path = volume_group_dir / "index.json"
    if not index_path.exists():
        return False
    try:
        index = json.loads(index_path.read_text())
        for shard in index["shards"]:
            f = volume_group_dir / shard["raw_data"]["basename"]
            if not f.exists() or f.stat().st_size != shard["raw_data"]["bytes"]:
                return False
        return True
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def copy_group_to_volume(local_dir: Path, volume_dir: Path, retries: int = 3) -> None:
    """Copy shard files then index.json last (completion marker), verifying sizes."""
    files = sorted(p for p in local_dir.iterdir() if p.name != "index.json")
    files.append(local_dir / "index.json")
    for attempt in range(1, retries + 1):
        try:
            volume_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                dst = volume_dir / src.name
                shutil.copyfile(src, dst)
                if dst.stat().st_size != src.stat().st_size:
                    raise IOError(f"Size mismatch after copy: {dst}")
            return
        except OSError as exc:
            LOG.warning("Copy attempt %d/%d for %s failed: %s", attempt, retries, volume_dir, exc)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def process_group(group_key: tuple[str, int]) -> dict:
    """Worker: convert one group of samples into MDS shards on the Volume."""
    from streaming import MDSWriter

    split, group_idx = group_key
    ctx = _WORKER_CTX
    tasks: list[SampleTask] = ctx["groups"][group_key]
    group_name = f"group_{group_idx:05d}"
    local_dir = Path(ctx["staging_root"]) / split / group_name
    volume_dir = Path(ctx["out_root"]) / split / group_name

    if group_is_complete(volume_dir):
        LOG.info("[%s/%s] already complete on volume; skipping", split, group_name)
        return {"split": split, "group": group_name, "samples": len(tasks), "skipped_existing": True}

    shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True)
    t0 = time.time()
    n_points = 0
    failed_samples: list[str] = []
    with MDSWriter(
        out=str(local_dir),
        columns=MDS_COLUMNS,
        compression=ctx["compression"],
        hashes=["xxh64"],
        size_limit=ctx["size_limit"],
        progress_bar=False,
    ) as writer:
        for task in tasks:
            try:
                record = task_to_record(task)
            except (RuntimeError, OSError) as exc:
                LOG.error("[%s/%s] sample %s failed: %s", split, group_name, task.sample_token, exc)
                failed_samples.append(task.sample_token)
                continue
            n_points += len(record["points"])
            writer.write(record)

    copy_group_to_volume(local_dir, volume_dir)
    shutil.rmtree(local_dir, ignore_errors=True)
    LOG.info(
        "[%s/%s] wrote %d samples (%.1fM points) in %.1fs",
        split, group_name, len(tasks) - len(failed_samples), n_points / 1e6, time.time() - t0,
    )
    return {
        "split": split,
        "group": group_name,
        "samples": len(tasks) - len(failed_samples),
        "failed_samples": failed_samples,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_split(split: str, tasks: list[SampleTask], args, executor_groups: dict) -> None:
    n_groups = max(1, math.ceil(len(tasks) / args.samples_per_group))
    for i in range(n_groups):
        executor_groups[(split, i)] = tasks[i * args.samples_per_group : (i + 1) * args.samples_per_group]


def merge_split_index(out_root: Path, split: str) -> int:
    """Merge per-group index.json files into a single split-level index.json."""
    from streaming.base.util import merge_index

    split_dir = out_root / split
    index_paths = sorted(str(p) for p in split_dir.glob("group_*/index.json"))
    if not index_paths:
        raise RuntimeError(f"No group indexes found under {split_dir}")
    merge_index(index_paths, out=str(split_dir), keep_local=True)
    merged = json.loads((split_dir / "index.json").read_text())
    n = sum(s["samples"] for s in merged["shards"])
    LOG.info("[%s] merged %d group indexes -> %d samples total", split, len(index_paths), n)
    return n


def verify_split(out_root: Path, split: str, n_checks: int = 5) -> None:
    """Prove the split is trainable: read random samples via StreamingDataset."""
    from streaming import StreamingDataset

    ds = StreamingDataset(local=str(out_root / split), batch_size=1, shuffle=False, predownload=1)
    assert len(ds) > 0, f"{split}: empty dataset"
    rng = np.random.default_rng(0)
    for idx in rng.choice(len(ds), size=min(n_checks, len(ds)), replace=False):
        rec = ds[int(idx)]
        pts, pose = rec["points"], rec["ego_pose"]
        boxes = np.frombuffer(rec["gt_boxes"], dtype=np.float32).reshape(-1, 7)
        classes = np.frombuffer(rec["gt_classes"], dtype=np.int8)
        assert pts.ndim == 2 and pts.shape[1] == 5 and pts.dtype == np.float32
        assert np.isfinite(pts[:, :3]).all(), f"{split}[{idx}]: non-finite points"
        assert boxes.shape == (len(classes), 7)
        assert pose.shape == (4, 4) and abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-3
        assert len(rec["points_sensor_id"]) == len(pts)
        assert len(rec["meta"]["gt_names"]) == len(classes)
    LOG.info("[%s] verification OK: %d samples, spot-checked %d records", split, len(ds), n_checks)
    del ds


def terminate_cluster() -> None:
    """Terminate (stop, NOT delete) the current Databricks cluster."""
    import requests

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID", "")
    if not (host and token and cluster_id):
        LOG.error("Missing DATABRICKS_HOST/TOKEN/CLUSTER_ID env vars; cannot terminate cluster")
        return
    LOG.info("Terminating cluster %s ...", cluster_id)
    for attempt in range(1, 6):
        try:
            resp = requests.post(
                f"{host}/api/2.1/clusters/delete",
                headers={"Authorization": f"Bearer {token}"},
                json={"cluster_id": cluster_id},
                timeout=30,
            )
            if resp.ok:
                LOG.info("Cluster termination requested successfully.")
                return
            LOG.warning("Terminate attempt %d failed: HTTP %d %s", attempt, resp.status_code, resp.text[:500])
        except requests.RequestException as exc:
            LOG.warning("Terminate attempt %d failed: %s", attempt, exc)
        time.sleep(10 * attempt)
    LOG.error("Failed to terminate cluster after 5 attempts; please terminate it manually.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path,
                   default=Path("/Volumes/daai_ke_team/default/images/lyft_3d_object_detection/extracted"))
    p.add_argument("--out-root", type=Path,
                   default=Path("/Volumes/daai_ke_team/default/images/lyft_3d_object_detection/mds_shards"))
    p.add_argument("--staging-root", type=Path, default=Path("/local_disk0/lyft_mds_staging"),
                   help="Fast local disk used to stage shards before copying to the Volume.")
    p.add_argument("--splits", default="train,val,test",
                   help="Comma-separated subset of {train,val,test}. train/val both come from train_data.")
    p.add_argument("--val-frac", type=float, default=0.15, help="Fraction of scenes (per host) held out for val.")
    p.add_argument("--num-workers", type=int, default=min(32, os.cpu_count() or 8))
    p.add_argument("--samples-per-group", type=int, default=512)
    p.add_argument("--size-limit", default="128mb", help="MDS shard size limit.")
    p.add_argument("--compression", default=None,
                   help="MDS compression (e.g. 'zstd:6'). Default: none, for zero-cost reads at training time.")
    p.add_argument("--limit", type=int, default=None, help="Cap samples per split (smoke testing).")
    p.add_argument("--terminate-cluster", action="store_true",
                   help="Terminate (stop, not delete) this Databricks cluster after success.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = set(splits) - {"train", "val", "test"}
    if unknown:
        LOG.error("Unknown splits: %s", unknown)
        return 2
    LOG.info("Args: %s", vars(args))
    t_start = time.time()

    # ---- Build per-sample tasks -------------------------------------------
    split_tasks: dict[str, list[SampleTask]] = {}
    if {"train", "val"} & set(splits):
        train_tasks, val_tasks = split_train_val(build_tasks(args.data_root, "train"), args.val_frac)
        if "train" in splits:
            split_tasks["train"] = train_tasks
        if "val" in splits:
            split_tasks["val"] = val_tasks
    if "test" in splits:
        split_tasks["test"] = build_tasks(args.data_root, "test")
    if args.limit:
        split_tasks = {k: v[: args.limit] for k, v in split_tasks.items()}
    gc.collect()

    # ---- Fan out groups to a process pool ---------------------------------
    groups: dict[tuple[str, int], list[SampleTask]] = {}
    for split, tasks in split_tasks.items():
        run_split(split, tasks, args, groups)
    total_samples = sum(len(v) for v in groups.values())
    LOG.info("Processing %d samples in %d groups with %d workers", total_samples, len(groups), args.num_workers)

    args.staging_root.mkdir(parents=True, exist_ok=True)
    _WORKER_CTX.update(
        groups=groups,
        staging_root=str(args.staging_root),
        out_root=str(args.out_root),
        compression=args.compression,
        size_limit=args.size_limit,
    )

    failed_groups: list[tuple[str, int]] = []
    failed_samples: list[str] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("fork")) as pool:
        futures = {pool.submit(process_group, key): key for key in sorted(groups)}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                result = fut.result()
                failed_samples.extend(result.get("failed_samples", []))
            except Exception:
                LOG.exception("Group %s failed", key)
                failed_groups.append(key)
            done += 1
            elapsed = time.time() - t_start
            LOG.info("Progress: %d/%d groups (%.1f min elapsed, ETA %.1f min)",
                     done, len(groups), elapsed / 60, elapsed / done * (len(groups) - done) / 60)

    shutil.rmtree(args.staging_root, ignore_errors=True)
    if failed_groups:
        LOG.error("%d group(s) FAILED: %s -- fix and re-run (completed groups are skipped automatically). "
                  "Cluster left running.", len(failed_groups), failed_groups)
        return 1

    # ---- Merge per-group indexes, verify, write dataset metadata ----------
    split_counts = {split: merge_split_index(args.out_root, split) for split in split_tasks}
    for split in split_tasks:
        verify_split(args.out_root, split)

    dataset_meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_data_root": str(args.data_root),
        "class_names": CLASS_NAMES,
        "splits": split_counts,
        "val_frac": args.val_frac,
        "failed_samples": failed_samples,
        "columns": MDS_COLUMNS,
        "conventions": {
            "points": "[N,5] float32 (x,y,z,intensity,ring), ego frame of the keyframe LIDAR_TOP ego pose",
            "gt_boxes": "bytes -> np.frombuffer(b, np.float32).reshape(-1, 7); (cx,cy,cz,w,l,h,yaw), "
                        "ego frame; yaw about +z, 0 = +x axis",
            "gt_classes": "bytes -> np.frombuffer(b, np.int8); index into class_names",
            "ego_pose": "[4,4] float64 ego->world; use to map predictions back to world for submission",
            "quaternions": "[w,x,y,z] (nuScenes convention)",
        },
    }
    meta_path = args.out_root / "dataset_meta.json"
    meta_path.write_text(json.dumps(dataset_meta, indent=2))
    LOG.info("Wrote %s", meta_path)
    if failed_samples:
        LOG.warning("%d individual sample(s) were skipped due to corrupt lidar data: %s",
                    len(failed_samples), failed_samples)

    LOG.info("ALL DONE in %.1f min. Split sizes: %s", (time.time() - t_start) / 60, split_counts)
    if args.terminate_cluster:
        terminate_cluster()
    return 0


if __name__ == "__main__":
    sys.exit(main())
