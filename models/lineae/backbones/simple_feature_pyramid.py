"""A small learned P3/P4/P5 adapter for stride-16 ViT features."""

from __future__ import annotations

from torch import Tensor, nn


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _norm_act(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.GroupNorm(_group_count(channels), channels), nn.GELU())


class SimpleFeaturePyramid(nn.Module):
    """Build stride 8, 16, and 32 features from one stride-16 map."""

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.out_channels = out_channels
        self.p3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            _norm_act(out_channels),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _norm_act(out_channels),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _norm_act(out_channels),
        )

    def forward(self, feature: Tensor) -> list[Tensor]:
        return [self.p3(feature), self.p4(feature), self.p5(feature)]
