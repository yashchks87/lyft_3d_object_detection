"""Compute per-class box statistics from the training shards.

The BEV U-Net predicts footprints only; box height and vertical center come
from these per-class means at decode time. Writes class_stats.json (default:
repository root), which scripts/train_lidar.py picks up automatically.

Usage:
    python scripts/compute_class_stats.py --cache /local_disk0/mds_cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from streaming import StreamingDataset

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lidar_data import CLASS_NAMES, load_dataset_meta, validate_paths

DEFAULT_MDS = Path('/Volumes/daai_ke_team/default/images/lyft_3d_object_detection/mds_shards')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mds', type=Path, default=DEFAULT_MDS)
    parser.add_argument('--cache', type=Path, required=True,
                        help='Node-local shard cache; never on /Volumes.')
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'class_stats.json')
    parser.add_argument('--limit', type=int, default=5000,
                        help='Samples to scan, evenly spaced over the train split; 0 = all.')
    args = parser.parse_args()
    remote, cache = validate_paths(str(args.mds), str(args.cache))
    load_dataset_meta(remote)
    # Shares the training cache subdirectory, so shards downloaded here warm
    # the cache for train_lidar.py. Do not run while a training job is active.
    dataset = StreamingDataset(remote=str(remote / 'train'), local=str(cache / 'train'),
                               batch_size=1, shuffle=False, predownload=8)
    indices = (range(len(dataset)) if not args.limit or args.limit >= len(dataset)
               else np.linspace(0, len(dataset) - 1, args.limit).astype(int))
    accumulator = {name: [] for name in CLASS_NAMES}
    for count, index in enumerate(indices, start=1):
        record = dataset[int(index)]
        boxes = np.frombuffer(record['gt_boxes'], dtype=np.float32).reshape(-1, 7)
        classes = np.frombuffer(record['gt_classes'], dtype=np.int8)
        for box, cls in zip(boxes, classes):
            accumulator[CLASS_NAMES[cls]].append([box[3], box[4], box[5], box[2]])
        if count % 500 == 0:
            print(f'{count}/{len(indices)} samples scanned', flush=True)
    stats = {}
    for name, rows in accumulator.items():
        if not rows:
            print(f'WARNING: no {name!r} boxes seen; it will use fallback constants.')
            continue
        array = np.asarray(rows)
        mean, std = array.mean(0), array.std(0)
        stats[name] = {'w': round(float(mean[0]), 3), 'l': round(float(mean[1]), 3),
                       'h': round(float(mean[2]), 3), 'cz': round(float(mean[3]), 3),
                       'h_std': round(float(std[2]), 3), 'cz_std': round(float(std[3]), 3),
                       'count': len(rows)}
    args.out.write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.out} from {len(indices)} samples:')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
