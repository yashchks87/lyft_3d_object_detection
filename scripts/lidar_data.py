"""Streaming data pipeline for LiDAR-only BEV training.

Reads the MDS shards produced by scripts/create_shards.py (see its module
docstring for the record schema), applies point-level geometric augmentation,
and rasterizes each sample into a multi-channel BEV image plus per-class
segmentation targets.

BEV input channels (geometry only -- Lyft intensity/ring are constant):
    [0 .. z_bins)      occupancy per height slab between z_min and z_max
    [z_bins]           log point density
    [z_bins + 1]       normalised maximum point height
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from streaming import StreamingDataset

from scripts.box_ops import BEVGrid, rasterize_boxes

CLASS_NAMES = [
    'car', 'pedestrian', 'animal', 'other_vehicle', 'bus',
    'motorcycle', 'truck', 'emergency_vehicle', 'bicycle',
]

# Fallback per-class box statistics (metres); superseded by class_stats.json
# from scripts/compute_class_stats.py when available. `cz` assumes the ego
# frame ground plane sits near z = -0.9 (measured on the shards).
FALLBACK_CLASS_STATS = {
    'car': {'w': 1.92, 'l': 4.75, 'h': 1.72, 'cz': -0.04},
    'pedestrian': {'w': 0.77, 'l': 0.81, 'h': 1.78, 'cz': -0.01},
    'animal': {'w': 0.36, 'l': 0.73, 'h': 0.51, 'cz': -0.65},
    'other_vehicle': {'w': 2.79, 'l': 8.20, 'h': 3.23, 'cz': 0.72},
    'bus': {'w': 2.96, 'l': 12.34, 'h': 3.44, 'cz': 0.82},
    'motorcycle': {'w': 0.96, 'l': 2.35, 'h': 1.59, 'cz': -0.11},
    'truck': {'w': 2.85, 'l': 10.24, 'h': 3.44, 'cz': 0.82},
    'emergency_vehicle': {'w': 2.45, 'l': 6.52, 'h': 2.39, 'cz': 0.30},
    'bicycle': {'w': 0.63, 'l': 1.76, 'h': 1.44, 'cz': -0.18},
}


@dataclass(frozen=True)
class BEVConfig:
    xy_range: float = 100.0
    resolution: float = 0.25
    z_min: float = -2.5
    z_max: float = 5.5
    z_bins: int = 8

    @property
    def channels(self) -> int:
        return self.z_bins + 2

    @property
    def grid(self) -> BEVGrid:
        return BEVGrid(self.xy_range, self.resolution)


@dataclass(frozen=True)
class AugmentConfig:
    rotation: float = math.pi / 8    # uniform yaw about +z, radians
    flip_probability: float = 0.5    # applied independently per axis
    scale: float = 0.05              # uniform in [1 - scale, 1 + scale]
    translation: float = 0.5         # metres, xy jitter


def validate_paths(remote: str, local: str) -> tuple[Path, Path]:
    remote_path, local_path = Path(remote), Path(local)
    if not remote_path.is_absolute() or not local_path.is_absolute():
        raise ValueError('Remote dataset and local cache must be absolute paths.')
    remote_path, local_path = remote_path.resolve(), local_path.resolve()
    if remote_path == local_path or remote_path in local_path.parents or local_path in remote_path.parents:
        raise ValueError('Remote dataset and local cache must be separate, non-nested directories.')
    if any(root == local_path or root in local_path.parents for root in (Path('/Volumes'), Path('/dbfs'))):
        raise ValueError('Cache must be on node-local disk, not /Volumes or /dbfs.')
    return remote_path, local_path


def load_dataset_meta(remote_root: Path) -> dict:
    meta_path = remote_root / 'dataset_meta.json'
    if not meta_path.is_file():
        raise ValueError(f'Missing dataset_meta.json under {remote_root}.')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('class_names') != CLASS_NAMES:
        raise ValueError('Shard class names do not match scripts/lidar_data.py CLASS_NAMES.')
    return meta


def load_class_stats(path: Path | None) -> dict:
    if path is None:
        return dict(FALLBACK_CLASS_STATS)
    stats = json.loads(Path(path).read_text(encoding='utf-8'))
    missing = [name for name in CLASS_NAMES
               if name not in stats or not {'w', 'l', 'h', 'cz'} <= set(stats[name])]
    if missing:
        raise ValueError(f'class stats file {path} is missing entries for: {missing}')
    return {name: stats[name] for name in CLASS_NAMES}


def augment_sample(points: np.ndarray, boxes: np.ndarray,
                   config: AugmentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Global rotation / flips / scale / translation on points and boxes together."""
    points, boxes = points.copy(), boxes.copy()
    angle = random.uniform(-config.rotation, config.rotation)
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    points[:, :2] = points[:, :2] @ rotation.T
    if len(boxes):
        boxes[:, :2] = boxes[:, :2] @ rotation.T
        boxes[:, 6] += angle
    if random.random() < config.flip_probability:  # flip across the x axis
        points[:, 1] *= -1
        if len(boxes):
            boxes[:, 1] *= -1
            boxes[:, 6] *= -1
    if random.random() < config.flip_probability:  # flip across the y axis
        points[:, 0] *= -1
        if len(boxes):
            boxes[:, 0] *= -1
            boxes[:, 6] = math.pi - boxes[:, 6]
    scale = random.uniform(1 - config.scale, 1 + config.scale)
    points[:, :3] *= scale
    shift = np.array([random.uniform(-config.translation, config.translation),
                      random.uniform(-config.translation, config.translation)], dtype=np.float32)
    points[:, :2] += shift
    if len(boxes):
        boxes[:, :6] *= scale
        boxes[:, :2] += shift
    return points, boxes


def encode_bev(points: np.ndarray, config: BEVConfig) -> np.ndarray:
    """Rasterize an ego-frame point cloud into the BEV input tensor [C, H, W]."""
    size = config.grid.size
    xy = points[:, :2]
    z = points[:, 2]
    keep = ((np.abs(xy) < config.xy_range).all(axis=1)
            & (z >= config.z_min) & (z < config.z_max))
    xy, z = xy[keep], z[keep]
    rows = np.clip(((xy[:, 0] + config.xy_range) / config.resolution).astype(np.int64), 0, size - 1)
    cols = np.clip(((xy[:, 1] + config.xy_range) / config.resolution).astype(np.int64), 0, size - 1)
    slabs = np.clip(((z - config.z_min) / (config.z_max - config.z_min)
                     * config.z_bins).astype(np.int64), 0, config.z_bins - 1)
    image = np.zeros((config.channels, size, size), dtype=np.float32)
    image[slabs, rows, cols] = 1.0
    density = np.zeros((size, size), dtype=np.float32)
    np.add.at(density, (rows, cols), 1.0)
    image[config.z_bins] = np.log1p(density) / 4.0
    height = np.zeros((size, size), dtype=np.float32)
    np.maximum.at(height, (rows, cols), (z - config.z_min) / (config.z_max - config.z_min))
    image[config.z_bins + 1] = height
    return image


class LyftBEVDataset(StreamingDataset):
    """Streams MDS records and yields BEV tensors, targets, and raw GT boxes.

    With torchrun, StreamingDataset partitions samples across ranks and loader
    workers by itself -- do NOT wrap it in a DistributedSampler. All ranks on a
    node must share the same `local` cache directory.
    """

    def __init__(self, remote: str, local: str, *, bev: BEVConfig, training: bool,
                 augment: AugmentConfig | None = None, batch_size: int = 1, **kwargs):
        remote_path, local_path = validate_paths(remote, local)
        if not (remote_path / 'index.json').is_file():
            raise ValueError(f'No index.json under {remote_path}; not an MDS split directory.')
        self.bev = bev
        self.training = training
        self.augment = augment if training else None
        kwargs.setdefault('shuffle', training)
        kwargs.setdefault('validate_hash', None)
        kwargs.setdefault('download_timeout', 300)
        super().__init__(remote=str(remote_path), local=str(local_path),
                         batch_size=batch_size, **kwargs)

    def __getitem__(self, index: int) -> dict:
        record = super().__getitem__(index)
        points = np.asarray(record['points'], dtype=np.float32)
        boxes = np.frombuffer(record['gt_boxes'], dtype=np.float32).reshape(-1, 7).copy()
        classes = np.frombuffer(record['gt_classes'], dtype=np.int8).astype(np.int64)
        if self.augment is not None:
            points, boxes = augment_sample(points, boxes, self.augment)
        image = encode_bev(points, self.bev)
        target = rasterize_boxes(boxes, classes, self.bev.grid, len(CLASS_NAMES))
        return {
            'image': torch.from_numpy(image),
            'target': torch.from_numpy(target),
            'sample_token': record['sample_token'],
            'gt_boxes': boxes,
            'gt_classes': classes,
        }


def collate_bev(samples: list[dict]) -> dict:
    return {
        'images': torch.stack([sample['image'] for sample in samples]),
        'targets': torch.stack([sample['target'] for sample in samples]),
        'sample_tokens': [sample['sample_token'] for sample in samples],
        'gt_boxes': [sample['gt_boxes'] for sample in samples],
        'gt_classes': [sample['gt_classes'] for sample in samples],
    }
