"""Evaluation metric for LiDAR 3D detection.

Implements the Lyft/Kaggle-style score: average precision computed per class
with 3D IoU matching, swept over IoU thresholds 0.55 to 0.95 in steps of 0.05,
then averaged over thresholds and over the classes that appear in the ground
truth. AP itself is the area under the interpolated precision-recall curve.
"""

from __future__ import annotations

import numpy as np

from scripts.box_ops import iou_matrix_3d

IOU_THRESHOLDS = tuple(np.arange(0.55, 0.951, 0.05).round(2))


def average_precision(scores: np.ndarray, matched: np.ndarray, num_gt: int) -> float:
    """AP from per-prediction match flags, predictions already sorted by score."""
    if num_gt == 0:
        return float('nan')
    if not len(scores):
        return 0.0
    true_positives = np.cumsum(matched)
    recall = true_positives / num_gt
    precision = true_positives / np.arange(1, len(scores) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision))


def evaluate_detections(predictions: list[dict], ground_truths: list[dict], num_classes: int,
                        thresholds: tuple[float, ...] = IOU_THRESHOLDS) -> dict:
    """Compute the mAP sweep over paired per-sample predictions and ground truth.

    Each element of `predictions` has keys boxes [N, 7], classes [N], scores [N];
    each element of `ground_truths` has boxes [M, 7], classes [M]. Index i of
    both lists refers to the same sample.

    Returns {'map': float, 'per_class': {class index -> mean AP over thresholds},
    'per_threshold': {threshold -> mAP}} where classes absent from the ground
    truth are excluded everywhere.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError('predictions and ground_truths must pair up one to one.')
    # matches[cls] collects (score, matched-per-threshold) rows; gt_counts[cls] totals.
    match_rows: dict[int, list] = {cls: [] for cls in range(num_classes)}
    gt_counts = np.zeros(num_classes, dtype=np.int64)
    for prediction, truth in zip(predictions, ground_truths):
        pred_classes = np.asarray(prediction['classes'], dtype=np.int64)
        gt_classes = np.asarray(truth['classes'], dtype=np.int64)
        np.add.at(gt_counts, gt_classes, 1)
        for cls in np.unique(pred_classes):
            pred_mask = pred_classes == cls
            gt_mask = gt_classes == cls
            scores = np.asarray(prediction['scores'], dtype=np.float64)[pred_mask]
            order = np.argsort(-scores)
            pred_boxes = np.asarray(prediction['boxes'])[pred_mask][order]
            ious = iou_matrix_3d(pred_boxes, np.asarray(truth['boxes']).reshape(-1, 7)[gt_mask])
            matched = np.zeros((len(pred_boxes), len(thresholds)), dtype=bool)
            taken = np.zeros((len(thresholds), ious.shape[1]), dtype=bool)
            for i in range(len(pred_boxes)):
                for t, threshold in enumerate(thresholds):
                    candidates = np.where(~taken[t] & (ious[i] >= threshold))[0]
                    if len(candidates):
                        best = candidates[np.argmax(ious[i][candidates])]
                        taken[t, best] = True
                        matched[i, t] = True
            match_rows[cls].extend(zip(scores[order], matched))
    per_class, per_threshold = {}, {threshold: [] for threshold in thresholds}
    for cls in range(num_classes):
        if not gt_counts[cls]:
            continue
        rows = sorted(match_rows[cls], key=lambda row: -row[0])
        scores = np.asarray([row[0] for row in rows])
        matched = (np.asarray([row[1] for row in rows])
                   if rows else np.zeros((0, len(thresholds)), dtype=bool))
        class_aps = [average_precision(scores, matched[:, t], int(gt_counts[cls]))
                     for t in range(len(thresholds))]
        per_class[cls] = float(np.mean(class_aps))
        for threshold, ap in zip(thresholds, class_aps):
            per_threshold[threshold].append(ap)
    if not per_class:
        raise ValueError('Ground truth contains no boxes; cannot evaluate.')
    return {
        'map': float(np.mean(list(per_class.values()))),
        'per_class': per_class,
        'per_threshold': {threshold: float(np.mean(aps))
                          for threshold, aps in per_threshold.items() if aps},
    }
