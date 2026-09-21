"""Models and losses for LiDAR-only BEV detection.

Stage 1 of the model progression: a BEV U-Net that predicts per-class object
footprint masks; boxes are recovered by scripts/box_ops.masks_to_boxes.
The registry keeps room for the later stages (PointPillars, CenterPoint head).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for encoder in self.encoders:
            skips.append(encoder(functional.max_pool2d(skips[-1], 2)))
        x = skips.pop()
        for upsampler, decoder in zip(self.upsamplers, self.decoders):
            x = decoder(torch.cat([upsampler(x), skips.pop()], dim=1))
        return self.head(x)


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


MODELS = ('bev_unet',)


def build_model(name: str, *, in_channels: int, num_classes: int,
                base_channels: int = 32, depth: int = 4) -> nn.Module:
    if name == 'bev_unet':
        return BEVUNet(in_channels, num_classes, base_channels=base_channels, depth=depth)
    raise ValueError(f'Unknown model {name!r}; choose from {MODELS}.')
