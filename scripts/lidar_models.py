"""Models and losses for LiDAR-only BEV detection.

Model progression over the shared BEV encoding:
  - bev_unet (stage 1): per-class footprint masks; boxes recovered by
    scripts/box_ops.masks_to_boxes (connected components -> min-area rect).
  - bev_centernet (stage 2): CenterPoint-style head on the same U-Net trunk;
    per-class center heatmaps plus dense box regression, decoded by
    scripts/box_ops.heatmap_to_boxes (peak picking, no class statistics).
The registry keeps room for the later stages (PointPillars).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from scripts.box_ops import REGRESSION_CHANNELS


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    def conv(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, cout), cout), nn.SiLU(inplace=True)]
    return nn.Sequential(*conv(in_channels, out_channels), *conv(out_channels, out_channels))


class BEVUNet(nn.Module):
    """U-Net over the BEV pseudo-image; emits [B, num_classes, H, W] logits."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32,
                 depth: int = 4):
        super().__init__()
        if depth < 1 or base_channels < 8:
            raise ValueError('depth must be >= 1 and base_channels >= 8.')
        widths = [base_channels * 2 ** i for i in range(depth + 1)]
        self.stem = _block(in_channels, widths[0])
        self.encoders = nn.ModuleList(_block(widths[i], widths[i + 1]) for i in range(depth))
        self.upsamplers = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(depth)))
        self.decoders = nn.ModuleList(
            _block(2 * widths[i], widths[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], num_classes, 1)
        nn.init.constant_(self.head.bias, -4.0)  # start near-empty: sigmoid(-4) ~ 0.018

    def features(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for encoder in self.encoders:
            skips.append(encoder(functional.max_pool2d(skips[-1], 2)))
        x = skips.pop()
        for upsampler, decoder in zip(self.upsamplers, self.decoders):
            x = decoder(torch.cat([upsampler(x), skips.pop()], dim=1))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class BEVCenterNet(BEVUNet):
    """CenterPoint-style head on the BEV U-Net trunk.

    Emits [B, num_classes + REGRESSION_CHANNELS, H, W]: per-class center
    heatmap logits followed by the box regression channels (row/col offset,
    cz, log w/l/h, sin/cos yaw). One heatmap peak per object means adjacent
    instances cannot merge, and boxes carry their own size/height/yaw instead
    of being lifted from class statistics.
    """

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32,
                 depth: int = 4):
        super().__init__(in_channels, num_classes, base_channels=base_channels, depth=depth)
        self.num_classes = num_classes
        self.head = nn.Identity()  # the U-Net segmentation head is unused
        width = base_channels

        def make_head(out_channels: int, bias: float) -> nn.Sequential:
            final = nn.Conv2d(width, out_channels, 1)
            nn.init.constant_(final.bias, bias)
            return nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
                                 nn.GroupNorm(min(8, width), width), nn.SiLU(inplace=True), final)

        self.heatmap_head = make_head(num_classes, -4.0)  # start near-empty
        self.regression_head = make_head(REGRESSION_CHANNELS, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return torch.cat([self.heatmap_head(features), self.regression_head(features)], dim=1)


class FocalDiceLoss(nn.Module):
    """Sigmoid focal loss (normalised by positive count) plus soft Dice.

    Both terms cope with the extreme foreground sparsity of BEV maps: focal
    keeps the millions of easy background pixels from swamping the gradient,
    Dice directly optimises overlap per class.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, dice_weight: float = 1.0):
        super().__init__()
        if not 0 < alpha < 1 or gamma < 0 or dice_weight < 0:
            raise ValueError('alpha must be in (0, 1); gamma and dice_weight nonnegative.')
        self.gamma, self.alpha, self.dice_weight = gamma, alpha, dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = (alpha_t * (1 - p_t) ** self.gamma * bce).sum()
        positives = targets.sum().clamp(min=1.0)
        loss = focal / positives
        if self.dice_weight:
            intersection = (probabilities * targets).sum(dim=(0, 2, 3))
            cardinality = probabilities.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
            dice = (2 * intersection + 1.0) / (cardinality + 1.0)
            loss = loss + self.dice_weight * (1 - dice).mean()
        return loss


class CenterNetLoss(nn.Module):
    """Penalty-reduced focal loss on center heatmaps plus masked L1 regression.

    Expects logits [B, C + REGRESSION_CHANNELS, H, W] from BEVCenterNet and
    targets [B, C + REGRESSION_CHANNELS + 1, H, W] from
    scripts/box_ops.encode_center_targets (heatmap, regression, center mask).
    The focal term follows CornerNet/CenterPoint: positives are the exact
    center pixels (heatmap == 1), and negatives near a center are down-weighted
    by (1 - heatmap)^beta. Both terms are normalised by the positive count.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, reg_weight: float = 1.0):
        super().__init__()
        if alpha < 0 or beta < 0 or reg_weight < 0:
            raise ValueError('alpha, beta, and reg_weight must be nonnegative.')
        self.alpha, self.beta, self.reg_weight = alpha, beta, reg_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1] - REGRESSION_CHANNELS
        if targets.shape[1] != logits.shape[1] + 1:
            raise ValueError('CenterNetLoss targets must add one center-mask channel to the logits.')
        heat_logits = logits[:, :num_classes].float()
        heatmap = targets[:, :num_classes]
        positives = heatmap.eq(1.0)
        probabilities = torch.sigmoid(heat_logits)
        positive_loss = -functional.logsigmoid(heat_logits) * (1 - probabilities) ** self.alpha
        negative_loss = (-functional.logsigmoid(-heat_logits) * probabilities ** self.alpha
                         * (1 - heatmap) ** self.beta)
        count = positives.sum().clamp(min=1.0)
        loss = (torch.where(positives, positive_loss, negative_loss).sum()) / count
        mask = targets[:, -1:]
        regression = logits[:, num_classes:].float()
        regression_targets = targets[:, num_classes:-1]
        l1 = (functional.l1_loss(regression, regression_targets, reduction='none') * mask).sum()
        return loss + self.reg_weight * l1 / (count * REGRESSION_CHANNELS)


MODELS = ('bev_unet', 'bev_centernet')


def build_model(name: str, *, in_channels: int, num_classes: int,
                base_channels: int = 32, depth: int = 4) -> nn.Module:
    if name == 'bev_unet':
        return BEVUNet(in_channels, num_classes, base_channels=base_channels, depth=depth)
    if name == 'bev_centernet':
        return BEVCenterNet(in_channels, num_classes, base_channels=base_channels, depth=depth)
    raise ValueError(f'Unknown model {name!r}; choose from {MODELS}.')
