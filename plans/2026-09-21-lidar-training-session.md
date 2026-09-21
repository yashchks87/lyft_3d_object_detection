# Session log: LiDAR-only training pipeline (Stage 1, BEV U-Net)

Date: 2026-09-21
Chat transcript (condensed but complete) between Yash and Devin covering the
design and build of the Stage 1 LiDAR-only training stack, plus the launch
debugging that followed.

---

## 1. Which fusion technique gives the best results: LiDAR-only, early, or late?

**Answer:** Early (feature-level) fusion wins in the modern literature, but the
2019 Kaggle competition itself was won by LiDAR-only models.

- **What won the 2019 competition:** LiDAR-only BEV models (U-Net segmentation,
  PointPillars/SECOND-style detectors) with ensembles, TTA, and multi-frame
  accumulation. Cameras were largely ignored: 7 images/frame cost too much
  compute for too little gain under a strict 3D IoU metric (0.55-0.95).
- **Modern benchmarks (nuScenes/Waymo):**
  1. Early/feature-level fusion (BEVFusion, TransFusion): best, +3-6 mAP over
     the same LiDAR backbone; gains concentrate in pedestrian/bicycle/motorcycle
     and distant sparse objects.
  2. Late fusion (CLOCs-style re-scoring): modest, +1-3 mAP; can only re-score,
     never recover missed objects.
  3. LiDAR-only: very strong baseline, ~90-95% of fused performance.
- **Plan implication:** Stage order LiDAR -> early -> late is right. Stage 1
  gets most of the score; Stage 2 (point painting + map) should add the largest
  fusion gain; Stage 3 likely smallest (a valuable observation in itself).
- Caveat: fusion gains depend on calibration/sync quality.

## 2. Which model for the LiDAR-only approach? (+ MDS sharding plan)

Model options laid out:

| Option | Trade-off |
|---|---|
| BEV U-Net (segmentation) | Simplest, Kaggle reference approach; lower ceiling (mask->box lossy) |
| PointPillars | Canonical detector; anchor tuning fiddly |
| PointPillars + CenterPoint head | Best accuracy/effort; recommended end state |
| MMDetection3D off-the-shelf | Fastest strong number, but owns the data pipeline (conflicts with MosaicML streaming) |

**Decision:** progress simple -> complex through all four; stages 1-3 share one
streaming dataloader.

**MDS shard schema decision:** serialize raw points + boxes + everything needed
later (fusion-proof). Final record schema (as built by `scripts/create_shards.py`):
`sample_token`, `scene_token`, `points [N,5] (x,y,z,intensity,ring)` ego-frame,
`points_sensor_id`, `gt_boxes [M,7] (cx,cy,cz,w,l,h,yaw)` ego-frame bytes,
`gt_classes` bytes, `ego_pose [4,4]`, `meta` (cameras/lidars calibration JSON).
Scene-stratified train/val split. Shards already generated at
`/Volumes/daai_ke_team/default/images/lyft_3d_object_detection/mds_shards`
(train 18,899 / val 3,780 / test 27,468).

## 3. Build the production training scripts (torchrun + wandb)

Environment discovered: Databricks cluster, 4x NVIDIA A10G (23GB), 48 vCPU,
182GB RAM, 3.5TB `/local_disk0`. Shard probe findings: ~65k points/sample,
ground plane at z ~ -0.9, boxes out to ~92m, **intensity=100 and ring=1
constant** (Lyft LiDAR quirk) -> BEV encoding is geometry-only.

Files created under `scripts/` (mirroring the brest_cancer_detection project
conventions):

| File | Purpose |
|---|---|
| `box_ops.py` | Rotated 3D box geometry: corners, Sutherland-Hodgman polygon clip, 3D IoU matrix, min-area rect (rotating calipers), BEV grid, box rasterization, mask->box decoding |
| `lidar_data.py` | `LyftBEVDataset(StreamingDataset)`: decode records, augmentation (rotation/flips/scale/translation on points+boxes), BEV encoding (8 z-slab occupancy + log density + max height = 10 channels, +/-100m @ 0.25m -> 800x800), per-class target masks, collate |
| `lidar_models.py` | `BEVUNet` (GroupNorm+SiLU, depth 4, base 32, head bias -4) + `FocalDiceLoss`; `build_model` registry for later stages |
| `lidar_metrics.py` | Lyft-style metric: per-class AP with 3D IoU matching swept over thresholds 0.55:0.05:0.95, averaged over classes present in GT |
| `train_lidar.py` | Trainer: torchrun DDP (StreamingDataset self-partitions, no DistributedSampler), AMP bf16, grad accum, cosine LR + warmup, LR auto-scaled by global batch/16, checkpoints (best.pt by val mAP / last.pt full resume state / optional epoch_*.pt), resume, wandb logging, metrics.jsonl, config.json |
| `compute_class_stats.py` | Scans train shards -> `class_stats.json` (per-class w/l/h/cz means) used to lift BEV footprints to 3D at decode time |
| `test_lidar_pipeline.py` | 23 unit tests (geometry, IoU, rasterize/decode roundtrip, encoding, augmentation invariants, model shapes/grads, metric edge cases) - all passing |
| `run_training.sh` | torchrun launcher (see final form below) |

Notes:
- User created their own venv (`python 3.12`, torch 2.14+cu130, streaming 0.13,
  wandb 0.30); `scipy==1.15.1` was added to requirements.txt (needed by box_ops).
- `class_stats.json` generated from 5,000 train samples (115k car boxes);
  measured car cz ~ +0.75 (fallback constants were wrong -> file matters).

## 4. What do "checkpoints" mean?

- `best.pt`: model weights only, overwritten on new best val mAP -> use for
  inference/submission.
- `last.pt`: weights + optimizer + scheduler + scaler + epoch/global step ->
  exact resume via `--resume`.
- `epoch_NNN.pt`: per-epoch weights, only with `--save-epochs`.
- Every checkpoint embeds the full run config (BEV grid, class stats, args) so
  inference can rebuild the exact setup from the file alone.

## 5. Store weights on the Volume (survive cluster loss)

No code change needed: `save_checkpoint()` stages through a temp file and
streams sequentially (FUSE-safe). Just point the run output at the Volume:
`RUNS_DIR=/Volumes/daai_ke_team/default/images/lyft_3d_object_detection/runs`.
Cache must stay on `/local_disk0` (enforced by `validate_paths`).

## 6. Auto-terminate the cluster after training (flag-based)

Added `--terminate-cluster` (default off), reusing the pattern from
`create_shards.py` (Databricks `clusters/delete` API = stop, not delete;
requires `DATABRICKS_HOST/TOKEN/CLUSTER_ID`; 5 retries with backoff).

Semantics:
- success -> terminate
- crash with >= 1 completed epoch -> terminate (everything persisted on Volume;
  resume later)
- setup failure / Ctrl-C -> cluster left running

## 7. Detached launch (release terminal immediately)

`run_training.sh` gained `DETACH=1`: nohup + pid file, returns immediately,
prints log path and stop command.

## 8. GPU capacity tuning (4x A10G)

Measured on the real model/grid (800x800x10, bf16, fwd+bwd):

| per-GPU batch | peak mem | step time | throughput |
|---|---|---|---|
| 4 | 17.0 GB | 264 ms | 15.2 samples/s |
| 8 | 19.4 GB | 506 ms | 15.8 samples/s |
| 10 | 23.2 GB (edge) | 626 ms | 16.0 samples/s |
| 12 | OOM | - | - |

Key insight: **GPUs are compute-saturated at bs=4** - throughput is flat, so
bigger batches don't train faster; they buy fewer/better-averaged steps.
Chosen defaults: `--batch-size 8` (82% memory, safe headroom), `--num-workers 8`
(dataloader measured at 27ms/sample -> ~290 samples/s per rank vs 16/s GPU
demand; CPU not a bottleneck). LR auto-scales with global batch. Expected
~5 min/epoch training side.

## 9. wandb logging

Confirmed logging (primary rank): full config at setup; `train/loss_step` +
`train/lr` every 50 steps; per epoch `train/loss`, `val/loss`, `val/map`,
`val/best_map`, per-class `val/ap_*`, elapsed; `best_map` summary; wandb system
metrics (GPU util/mem). Local mirrors: metrics.jsonl, config.json, log file.
Checkpoints deliberately NOT uploaded to wandb (live on Volume).

Project pointed at the user's existing wandb project: defaults set to
entity `yashchks87`, project `Lyft 3D object detection` (verified via API).

## 10. Launch #1 and the FUSE incident

Run submitted detached with log redirected to the Volume. Symptoms: empty-ish
log, GPUs 0%. Diagnosis: all 5 processes stuck in **D state (uninterruptible
kernel I/O)** - `fdget_pos` / `folio_wait_bit_common` - blocked on streaming
small stdout writes (tqdm etc.) through one shared fd into the /Volumes FUSE
mount. SIGKILL queued but cannot land while blocked in kernel I/O; processes
unkillable from userspace; they hold ~0.5-1GB GPU memory each; nvidia-smi
intermittently hangs.

Root-cause lesson: **never stream frequent small writes to a UC Volume FUSE
mount.** Bulk atomic writes (checkpoints, metrics.jsonl) were always fine.

## 11. Final logging design (user decision)

Live log ONLY on node-local disk (`/local_disk0/run_logs/<run>.log`); on run
end the log is copied ONCE to `RUNS_DIR`; only after the copy does
`--terminate-cluster` fire (intercepted by `run_training.sh` itself, not the
trainer, to guarantee copy-before-terminate ordering; crash-during-setup leaves
the cluster running). Weights/metrics keep writing directly to the Volume
throughout, unchanged. Trade-off: unexpected node death loses the local log -
wandb console/metrics + per-epoch metrics.jsonl on the Volume are the safety
net.

## 12. Can GPU memory be cleaned without restarting the cluster?

Not for D-state processes: SIGKILL only lands when the wedged kernel I/O
returns; `nvidia-smi --gpu-reset` refuses while contexts exist; the Volume FUSE
daemon lives outside the container so it cannot be restarted from here.
Practical answer: the zombies hold only ~1GB/GPU; bs=8 peaks at 19.4/23.7GB, so
relaunching without a restart is viable; `--terminate-cluster` reboots
everything for free at the end anyway.

## 13. Launch #2 failure: "Reused local directory"

Relaunch crashed in setup: streaming raised
`Reused local directory: ['/local_disk0/mds_cache/val'] ...` - the zombie ranks
still counted as live users of the cache dir in streaming's shared-memory
registry (not "stale", so `clean_stale_shared_memory()` correctly skipped
them). Fix: renamed the warm 33GB cache `/local_disk0/mds_cache` ->
`/local_disk0/mds_cache_r2` (instant same-fs rename, no re-download); relaunch
with `CACHE_DIR=/local_disk0/mds_cache_r2`. Also noted: an earlier shell
one-liner had mangled `source venv/bin/activate` and `rm -rf` into one command -
paste multi-line commands as separate lines.

Normal warm-up expectation: ~30-60s torchrun+streaming init, then a JSON
`"event": "setup"` line, then GPU memory jumps to ~19GB within a couple of
minutes.

## 14. Why download MDS shards to local disk at all - isn't the goal streaming?

Streaming = just-in-time shard download overlapped with training + local cache;
the cache IS the design (remote = source of truth, local = hot tier):
- Shuffled training does random ~1-2MB reads; object storage behind FUSE pays
  cloud latency per read - terrible for that pattern; local NVMe serves it at
  GB/s (measured 27ms/sample).
- 20 epochs x 33GB = 660GB of remote traffic without a cache vs 33GB once.
- This cluster's FUSE layer already proved fragile under small-write load.
- `--cache-limit` bounds disk usage via LRU eviction if disk were small.
- Losing `/local_disk0` on termination costs only a one-time 33GB re-warm.

---

## Current launch command (state at end of session)

```bash
cd /root/lyft_3d_object_detection
source venv/bin/activate

DETACH=1 \
CACHE_DIR=/local_disk0/mds_cache_r2 \
RUNS_DIR=/Volumes/daai_ke_team/default/images/lyft_3d_object_detection/runs \
    ./scripts/run_training.sh bev_unet_001 --epochs 20 --wandb --terminate-cluster
```

Monitor: `tail -f /local_disk0/run_logs/bev_unet_001.log` and the wandb run
`yashchks87/Lyft 3D object detection/bev_unet_001`.
Stop: `kill $(cat /local_disk0/run_logs/bev_unet_001.pid)`.
Resume after any interruption: add `--resume <RUNS_DIR>/bev_unet_001/last.pt`.
