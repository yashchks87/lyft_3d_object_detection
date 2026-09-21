"""Rotated 3D box geometry for BEV LiDAR detection.

Box convention (matches the MDS shards): (cx, cy, cz, w, l, h, yaw) in the ego
frame, where `l` is the extent along the heading (yaw about +z, 0 = +x axis)
and `w` is the extent across it. All functions are pure numpy so they can run
inside dataloader workers and evaluation without torch.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


def box_corners_bev(boxes: np.ndarray) -> np.ndarray:
    """[N, 7] boxes -> [N, 4, 2] BEV corner polygons (counter-clockwise)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    half = np.stack([boxes[:, 4] / 2, boxes[:, 3] / 2], axis=1)  # (l/2, w/2)
    template = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)
    local = template[None, :, :] * half[:, None, :]
    cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    rot = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], axis=1)
    return np.einsum('nij,nkj->nki', rot, local) + boxes[:, None, :2]


def polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area of a polygon given as [K, 2] vertices."""
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of `subject` by convex counter-clockwise `clipper`."""
    output = list(subject)
    for i in range(len(clipper)):
        if not output:
            return np.empty((0, 2))
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = (b[0] - a[0], b[1] - a[1])
        polygon, output = output, []
        signs = [edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) for p in polygon]
        for j, p in enumerate(polygon):
            q, sp, sq = polygon[(j + 1) % len(polygon)], signs[j], signs[(j + 1) % len(polygon)]
            if sp >= 0:
                output.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                output.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return np.asarray(output).reshape(-1, 2)


def iou_matrix_3d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise 3D IoU (rotated BEV overlap x vertical overlap), [A, B]."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 7)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 7)
    result = np.zeros((len(boxes_a), len(boxes_b)))
    if not len(boxes_a) or not len(boxes_b):
        return result
    corners_a, corners_b = box_corners_bev(boxes_a), box_corners_bev(boxes_b)
    volumes_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    volumes_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]
    # Prefilter: centers further apart than the sum of the BEV circumradii cannot overlap.
    radii_a = np.hypot(boxes_a[:, 3], boxes_a[:, 4]) / 2
    radii_b = np.hypot(boxes_b[:, 3], boxes_b[:, 4]) / 2
    distances = np.linalg.norm(boxes_a[:, None, :2] - boxes_b[None, :, :2], axis=2)
    for i, j in zip(*np.nonzero(distances <= radii_a[:, None] + radii_b[None, :])):
        z_overlap = (min(boxes_a[i, 2] + boxes_a[i, 5] / 2, boxes_b[j, 2] + boxes_b[j, 5] / 2)
                     - max(boxes_a[i, 2] - boxes_a[i, 5] / 2, boxes_b[j, 2] - boxes_b[j, 5] / 2))
        if z_overlap <= 0:
            continue
        bev_overlap = polygon_area(clip_polygon(corners_a[i], corners_b[j]))
        intersection = bev_overlap * z_overlap
        union = volumes_a[i] + volumes_b[j] - intersection
        result[i, j] = intersection / union if union > 0 else 0.0
    return result


def min_area_rect(points: np.ndarray) -> tuple[float, float, float, float, float]:
    """Minimum-area enclosing rectangle of 2D points via rotating calipers.

    Returns (cx, cy, extent_along, extent_across, angle) where `angle` is the
    orientation of the `extent_along` axis. Degenerate inputs (fewer than three
    points or collinear) fall back to an axis-aligned box.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    try:
        hull = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        low, high = points.min(0), points.max(0)
        center = (low + high) / 2
        return center[0], center[1], high[0] - low[0], high[1] - low[1], 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    best = None
    for angle in angles:
        cos, sin = np.cos(angle), np.sin(angle)
        rotated = hull @ np.array([[cos, sin], [-sin, cos]]).T
        low, high = rotated.min(0), rotated.max(0)
        extent = high - low
        area = extent[0] * extent[1]
        if best is None or area < best[0]:
            center = (low + high) / 2 @ np.array([[cos, -sin], [sin, cos]]).T
            best = (area, center, extent, angle)
    _, center, extent, angle = best
    return center[0], center[1], extent[0], extent[1], angle


class BEVGrid:
    """Mapping between ego-frame metres and BEV pixels.

    Pixel (row, col) = (x bin, y bin); pixel centers at offset + (index + 0.5) * resolution.
    """

    def __init__(self, xy_range: float, resolution: float):
        if xy_range <= 0 or resolution <= 0:
            raise ValueError('BEV range and resolution must be positive.')
        self.xy_range = float(xy_range)
        self.resolution = float(resolution)
        self.size = int(round(2 * xy_range / resolution))

    def metres_to_pixels(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy) + self.xy_range) / self.resolution - 0.5

    def pixels_to_metres(self, indices: np.ndarray) -> np.ndarray:
        return (np.asarray(indices, dtype=np.float64) + 0.5) * self.resolution - self.xy_range


def rasterize_boxes(boxes: np.ndarray, classes: np.ndarray, grid: BEVGrid,
                    num_classes: int) -> np.ndarray:
    """Paint rotated box footprints into per-class binary masks [C, H, W]."""
    masks = np.zeros((num_classes, grid.size, grid.size), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    for box, cls in zip(boxes, np.asarray(classes, dtype=np.int64)):
        radius = np.hypot(box[3], box[4]) / 2
        low = np.floor(grid.metres_to_pixels(box[:2] - radius)).astype(int)
        high = np.ceil(grid.metres_to_pixels(box[:2] + radius)).astype(int) + 1
        low, high = np.clip(low, 0, grid.size), np.clip(high, 0, grid.size)
        if (high <= low).any():
            continue
        rows, cols = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]),
                                 indexing='ij')
        centers = grid.pixels_to_metres(np.stack([rows, cols], axis=-1)) - box[:2]
        cos, sin = np.cos(box[6]), np.sin(box[6])
        along = centers[..., 0] * cos + centers[..., 1] * sin
        across = -centers[..., 0] * sin + centers[..., 1] * cos
        inside = (np.abs(along) <= box[4] / 2) & (np.abs(across) <= box[3] / 2)
        masks[cls, low[0]:high[0], low[1]:high[1]][inside] = 1.0
    return masks


def masks_to_boxes(probabilities: np.ndarray, grid: BEVGrid, class_stats: dict,
                   class_names: list[str], threshold: float = 0.3, min_pixels: int = 2,
                   max_detections: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode per-class BEV probability maps into 3D boxes.

    Connected components above `threshold` become rotated rectangles
    (minimum-area fit over pixel centers); height and vertical center come from
    per-class statistics. Returns (boxes [N, 7], classes [N], scores [N])
    sorted by descending score and capped at `max_detections`.
    """
    boxes, classes, scores = [], [], []
    for cls, class_name in enumerate(class_names):
        stats = class_stats[class_name]
        labelled, count = ndimage.label(probabilities[cls] >= threshold)
        for label_id, component in enumerate(ndimage.find_objects(labelled, count), start=1):
            if component is None:
                continue
            local_rows, local_cols = np.nonzero(labelled[component] == label_id)
            if len(local_rows) < min_pixels:
                continue
            rows = local_rows + component[0].start
            cols = local_cols + component[1].start
            centers = grid.pixels_to_metres(np.stack([rows, cols], axis=1))
            cx, cy, extent_along, extent_across, angle = min_area_rect(centers)
            length = max(extent_along + grid.resolution, grid.resolution)
            width = max(extent_across + grid.resolution, grid.resolution)
            boxes.append([cx, cy, stats['cz'], width, length, stats['h'], angle])
            classes.append(cls)
            scores.append(float(probabilities[cls][rows, cols].mean()))
    if not boxes:
        return np.zeros((0, 7), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32)
    order = np.argsort(scores)[::-1][:max_detections]
    return (np.asarray(boxes, np.float32)[order], np.asarray(classes, np.int64)[order],
            np.asarray(scores, np.float32)[order])
