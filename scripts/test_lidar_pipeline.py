"""Unit tests for the LiDAR BEV pipeline: geometry, encoding, decoding, metric.

Run with: python -m pytest scripts/test_lidar_pipeline.py -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.box_ops import (
    REGRESSION_CHANNELS,
    BEVGrid,
    box_corners_bev,
    clip_polygon,
    encode_center_targets,
    gaussian_radius,
    heatmap_to_boxes,
    iou_matrix_3d,
    masks_to_boxes,
    min_area_rect,
    polygon_area,
    rasterize_boxes,
)
from scripts.lidar_data import (
    CLASS_NAMES,
    FALLBACK_CLASS_STATS,
    AugmentConfig,
    BEVConfig,
    augment_sample,
    encode_bev,
)
from scripts.lidar_metrics import evaluate_detections
from scripts.lidar_models import BEVCenterNet, BEVUNet, CenterNetLoss, FocalDiceLoss


def make_box(cx=0.0, cy=0.0, cz=0.0, w=2.0, l=4.0, h=1.5, yaw=0.0):
    return np.array([[cx, cy, cz, w, l, h, yaw]], dtype=np.float32)


class TestGeometry:
    def test_corners_axis_aligned(self):
        corners = box_corners_bev(make_box(w=2.0, l=4.0))[0]
        assert np.allclose(np.abs(corners), [[2, 1]] * 4)
        assert np.isclose(polygon_area(corners), 8.0)

    def test_corners_rotation_invariant_area(self):
        for yaw in (0.3, math.pi / 2, -1.2):
            corners = box_corners_bev(make_box(yaw=yaw))[0]
            assert np.isclose(polygon_area(corners), 8.0)

    def test_clip_identical(self):
        square = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
        assert np.isclose(polygon_area(clip_polygon(square, square)), 4.0)

    def test_clip_disjoint(self):
        a = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
        assert polygon_area(clip_polygon(a, a + 10)) == 0.0

    def test_iou_identical_and_disjoint(self):
        box = make_box(yaw=0.7)
        assert np.isclose(iou_matrix_3d(box, box)[0, 0], 1.0)
        assert iou_matrix_3d(box, make_box(cx=50.0))[0, 0] == 0.0

    def test_iou_half_overlap(self):
        a, b = make_box(), make_box(cy=1.0)  # shift by half the width
        expected = 0.5 / 1.5  # intersection 4, union 12 (footprint terms)
        assert np.isclose(iou_matrix_3d(a, b)[0, 0], expected)

    def test_iou_z_disjoint(self):
        assert iou_matrix_3d(make_box(cz=0.0), make_box(cz=5.0))[0, 0] == 0.0

    def test_iou_yaw_periodicity(self):
        assert np.isclose(iou_matrix_3d(make_box(), make_box(yaw=math.pi))[0, 0], 1.0)

    def test_min_area_rect_recovers_box(self):
        rng = np.random.default_rng(0)
        yaw = 0.5
        local = rng.uniform([-2, -1], [2, 1], size=(500, 2))
        local[:4] = [[-2, -1], [2, -1], [2, 1], [-2, 1]]  # pin the extremes
        cos, sin = math.cos(yaw), math.sin(yaw)
        points = local @ np.array([[cos, sin], [-sin, cos]]) + [3.0, -1.0]
        cx, cy, extent_a, extent_b, angle = min_area_rect(points)
        assert np.isclose(cx, 3.0, atol=0.05) and np.isclose(cy, -1.0, atol=0.05)
        assert np.isclose(sorted([extent_a, extent_b]), [2.0, 4.0], atol=0.05).all()
        # Orientation is recovered up to a 90-degree ambiguity.
        assert np.isclose((angle - yaw) % (math.pi / 2), 0.0, atol=0.02) or \
            np.isclose((angle - yaw) % (math.pi / 2), math.pi / 2, atol=0.02)

    def test_min_area_rect_degenerate(self):
        cx, cy, ea, eb, angle = min_area_rect(np.array([[1.0, 2.0], [1.0, 2.0]]))
        assert np.isclose(cx, 1.0) and np.isclose(cy, 2.0)


class TestRasterizeDecode:
    grid = BEVGrid(20.0, 0.25)

    def test_rasterize_footprint_area(self):
        masks = rasterize_boxes(make_box(w=2.0, l=4.0), [0], self.grid, 2)
        area = masks[0].sum() * self.grid.resolution ** 2
        assert masks[1].sum() == 0
        assert abs(area - 8.0) < 1.5  # discretisation tolerance

    def test_rasterize_out_of_range(self):
        masks = rasterize_boxes(make_box(cx=500.0), [0], self.grid, 1)
        assert masks.sum() == 0

    def test_roundtrip_box_through_masks(self):
        original = make_box(cx=4.0, cy=-3.0, w=2.0, l=4.6, yaw=0.4)
        masks = rasterize_boxes(original, [0], self.grid, len(CLASS_NAMES))
        probabilities = masks * 0.9
        boxes, classes, scores = masks_to_boxes(
            probabilities, self.grid, FALLBACK_CLASS_STATS, CLASS_NAMES, threshold=0.3)
        assert len(boxes) == 1 and classes[0] == 0 and scores[0] > 0.8
        decoded = boxes[0].copy()
        decoded[2], decoded[5] = original[0, 2], original[0, 5]  # z comes from stats
        iou = iou_matrix_3d(original, decoded[None])[0, 0]
        assert iou > 0.7, f'roundtrip IoU too low: {iou:.3f}'

    def test_decode_empty(self):
        probabilities = np.zeros((len(CLASS_NAMES), self.grid.size, self.grid.size), np.float32)
        boxes, classes, scores = masks_to_boxes(
            probabilities, self.grid, FALLBACK_CLASS_STATS, CLASS_NAMES)
        assert len(boxes) == 0 and len(classes) == 0 and len(scores) == 0


class TestCenterTargets:
    grid = BEVGrid(20.0, 0.25)

    def test_gaussian_radius_positive_and_monotonic(self):
        small = gaussian_radius(4.0, 3.0)
        large = gaussian_radius(40.0, 12.0)
        assert 0 < small < large

    def test_encode_peak_and_regression(self):
        box = make_box(cx=4.1, cy=-3.2, cz=0.6, w=2.0, l=4.6, h=1.7, yaw=0.4)
        heatmap, regression, mask = encode_center_targets(box, [2], self.grid, len(CLASS_NAMES))
        assert heatmap.shape == (len(CLASS_NAMES), self.grid.size, self.grid.size)
        assert regression.shape == (REGRESSION_CHANNELS, self.grid.size, self.grid.size)
        assert mask.sum() == 1.0 and heatmap.max() == 1.0
        assert heatmap[[cls for cls in range(len(CLASS_NAMES)) if cls != 2]].max() == 0.0
        row, col = np.unravel_index(heatmap[2].argmax(), heatmap[2].shape)
        assert mask[0, row, col] == 1.0
        values = regression[:, row, col]
        assert np.allclose(np.abs(values[:2]), [0.5, 0.5], atol=0.5)  # sub-pixel offsets
        assert np.isclose(values[2], 0.6)
        assert np.allclose(np.exp(values[3:6]), [2.0, 4.6, 1.7], rtol=1e-5)
        assert np.isclose(math.atan2(values[6], values[7]), 0.4, atol=1e-5)

    def test_encode_out_of_range_skipped(self):
        heatmap, regression, mask = encode_center_targets(
            make_box(cx=500.0), [0], self.grid, len(CLASS_NAMES))
        assert heatmap.sum() == 0 and regression.sum() == 0 and mask.sum() == 0

    def test_roundtrip_center_decode(self):
        original = make_box(cx=4.1, cy=-3.2, cz=0.6, w=2.0, l=4.6, h=1.7, yaw=0.4)
        heatmap, regression, _ = encode_center_targets(original, [0], self.grid, len(CLASS_NAMES))
        boxes, classes, scores = heatmap_to_boxes(heatmap, regression, self.grid, threshold=0.99)
        assert len(boxes) == 1 and classes[0] == 0 and scores[0] == 1.0
        iou = iou_matrix_3d(original, boxes)[0, 0]
        assert iou > 0.99, f'roundtrip IoU too low: {iou:.3f}'

    def test_adjacent_boxes_stay_separate(self):
        # The bev_unet failure mode: parked cars 2.5m apart merged into one blob.
        cars = np.concatenate([make_box(cy=-1.25), make_box(cy=1.25)])
        heatmap, regression, mask = encode_center_targets(cars, [0, 0], self.grid, len(CLASS_NAMES))
        assert mask.sum() == 2.0
        boxes, classes, _ = heatmap_to_boxes(heatmap, regression, self.grid, threshold=0.99)
        assert len(boxes) == 2 and (classes == 0).all()
        ious = iou_matrix_3d(cars, boxes)
        assert ious.max(axis=1).min() > 0.99  # each GT recovered by its own peak

    def test_decode_empty(self):
        empty = np.zeros((len(CLASS_NAMES), self.grid.size, self.grid.size), np.float32)
        boxes, classes, scores = heatmap_to_boxes(
            empty, np.zeros((REGRESSION_CHANNELS, self.grid.size, self.grid.size), np.float32),
            self.grid)
        assert len(boxes) == 0 and len(classes) == 0 and len(scores) == 0


class TestEncoding:
    def test_encode_channels_and_occupancy(self):
        config = BEVConfig(xy_range=10.0, resolution=0.5, z_min=-2.0, z_max=2.0, z_bins=4)
        points = np.array([[0.0, 0.0, -1.9, 100.0, 1.0],
                           [5.0, -5.0, 1.5, 100.0, 1.0],
                           [50.0, 0.0, 0.0, 100.0, 1.0]], dtype=np.float32)  # last out of range
        image = encode_bev(points, config)
        assert image.shape == (6, 40, 40)
        assert image[0, 20, 20] == 1.0        # low slab at origin
        assert image[3, 30, 10] == 1.0        # high slab at (5, -5)
        assert np.isclose(image[4].sum(), 2 * np.log1p(1.0) / 4.0)  # two occupied cells

    def test_augment_keeps_points_in_boxes(self):
        rng = np.random.default_rng(1)
        boxes = make_box(cx=5.0, cy=2.0, yaw=0.3)
        local = rng.uniform([-1.9, -0.9], [1.9, 0.9], size=(100, 2))
        cos, sin = math.cos(0.3), math.sin(0.3)
        xy = local @ np.array([[cos, sin], [-sin, cos]]) + [5.0, 2.0]
        points = np.concatenate([xy, np.zeros((100, 3))], axis=1).astype(np.float32)
        for _ in range(10):
            new_points, new_boxes = augment_sample(points, boxes, AugmentConfig())
            relative = new_points[:, :2] - new_boxes[0, :2]
            yaw = new_boxes[0, 6]
            cos, sin = math.cos(yaw), math.sin(yaw)
            along = relative[:, 0] * cos + relative[:, 1] * sin
            across = -relative[:, 0] * sin + relative[:, 1] * cos
            assert (np.abs(along) <= new_boxes[0, 4] / 2 + 1e-4).all()
            assert (np.abs(across) <= new_boxes[0, 3] / 2 + 1e-4).all()


class TestModel:
    def test_forward_shapes_and_loss(self):
        model = BEVUNet(in_channels=6, num_classes=9, base_channels=8, depth=3)
        images = torch.randn(2, 6, 64, 64)
        logits = model(images)
        assert logits.shape == (2, 9, 64, 64)
        targets = torch.zeros(2, 9, 64, 64)
        targets[0, 0, 10:20, 10:20] = 1.0
        loss = FocalDiceLoss()(logits, targets)
        assert torch.isfinite(loss) and loss.item() > 0
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters())

    def test_empty_target_loss_finite(self):
        loss = FocalDiceLoss()(torch.randn(1, 9, 32, 32), torch.zeros(1, 9, 32, 32))
        assert torch.isfinite(loss)

    def _center_targets(self, grid):
        heatmap, regression, mask = encode_center_targets(
            np.concatenate([make_box(cx=1.0, cy=-2.0, yaw=0.3), make_box(cx=-4.0, cy=3.0)]),
            [0, 4], grid, 9)
        return torch.from_numpy(np.concatenate([heatmap, regression, mask]))[None]

    def test_centernet_forward_and_loss(self):
        grid = BEVGrid(8.0, 0.25)  # 64 x 64
        model = BEVCenterNet(in_channels=6, num_classes=9, base_channels=8, depth=3)
        logits = model(torch.randn(1, 6, grid.size, grid.size))
        assert logits.shape == (1, 9 + REGRESSION_CHANNELS, grid.size, grid.size)
        loss = CenterNetLoss()(logits, self._center_targets(grid))
        assert torch.isfinite(loss) and loss.item() > 0
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters()
                   if parameter.requires_grad)

    def test_centernet_loss_prefers_correct_predictions(self):
        grid = BEVGrid(8.0, 0.25)
        targets = self._center_targets(grid)
        heatmap = targets[:, :9]
        good_logits = torch.cat([torch.logit(heatmap.clamp(1e-4, 1 - 1e-4)),
                                 targets[:, 9:-1]], dim=1)
        bad_logits = torch.cat([torch.logit((1 - heatmap).clamp(1e-4, 1 - 1e-4)),
                                targets[:, 9:-1] + 3.0], dim=1)
        criterion = CenterNetLoss()
        good, bad = criterion(good_logits, targets), criterion(bad_logits, targets)
        # Gaussian tails are soft negatives, so even a perfect prediction pays a
        # small penalty-reduced cost; the ordering and scale gap are what matter.
        assert good < 0.1 and bad > 10 * good

    def test_centernet_loss_empty_targets_finite(self):
        empty = torch.zeros(1, 9 + REGRESSION_CHANNELS + 1, 32, 32)
        loss = CenterNetLoss()(torch.randn(1, 9 + REGRESSION_CHANNELS, 32, 32), empty)
        assert torch.isfinite(loss)


class TestMetric:
    def test_perfect_predictions(self):
        truth = {'boxes': make_box(), 'classes': np.array([0])}
        prediction = {'boxes': make_box(), 'classes': np.array([0]),
                      'scores': np.array([0.9])}
        result = evaluate_detections([prediction], [truth], num_classes=9)
        assert np.isclose(result['map'], 1.0)
        assert set(result['per_class']) == {0}

    def test_no_predictions(self):
        truth = {'boxes': make_box(), 'classes': np.array([0])}
        empty = {'boxes': np.zeros((0, 7)), 'classes': np.zeros(0, int), 'scores': np.zeros(0)}
        result = evaluate_detections([empty], [truth], num_classes=9)
        assert result['map'] == 0.0

    def test_false_positive_other_class_ignored(self):
        truth = {'boxes': make_box(), 'classes': np.array([0])}
        prediction = {'boxes': np.concatenate([make_box(), make_box(cx=30.0)]),
                      'classes': np.array([0, 2]), 'scores': np.array([0.9, 0.8])}
        result = evaluate_detections([prediction], [truth], num_classes=9)
        assert np.isclose(result['map'], 1.0)  # class 2 has no GT -> excluded

    def test_duplicate_detection_is_fp(self):
        truth = {'boxes': make_box(), 'classes': np.array([0])}
        prediction = {'boxes': np.concatenate([make_box(), make_box()]),
                      'classes': np.array([0, 0]), 'scores': np.array([0.9, 0.8])}
        result = evaluate_detections([prediction], [truth], num_classes=9)
        assert 0.99 < result['map'] <= 1.0  # AP unaffected: TP ranked above the duplicate

    def test_loose_box_fails_high_thresholds(self):
        truth = {'boxes': make_box(), 'classes': np.array([0])}
        prediction = {'boxes': make_box(cy=0.5), 'classes': np.array([0]),
                      'scores': np.array([0.9])}
        result = evaluate_detections([prediction], [truth], num_classes=9)
        assert 0.0 < result['map'] < 1.0
