"""HGNetV2 LINEAE adapters, including Gazelle/DEIMv2 small derivatives.

The Atto/Femto/Pico stage layouts follow the local Gazelle reference, which in
turn credits DEIMv2 and D-FINE.  Unlike the reference loader, LINEAE never
downloads weights implicitly and records every shape-matched or skipped tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from ..hgnetv2 import HGNetv2
from .base import LINEAEBackbone, unwrap_state_dict


@dataclass(frozen=True)
class HGNetCheckpointReport:
    path: Path
    architecture: str
    source_tensor_count: int
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    strict: bool

    @property
    def tensor_count(self) -> int:
        return len(self.loaded_keys)


class HGNetV2Backbone(LINEAEBackbone):
    preprocess_profile = "linea"

    _ARCHITECTURES = {
        "hgnetv2_atto": "Atto",
        "hgnetv2_femto": "Femto",
        "hgnetv2_pico": "Pico",
        "hgnetv2_n": "B0",
        "hgnetv2_b0": "B0",
    }

    def __init__(
        self,
        *,
        name: str,
        weights_path: str | Path | None,
        use_lab: bool = True,
        freeze_norm: bool = False,
        trainable_depth: int = 0,
    ) -> None:
        super().__init__()
        try:
            architecture = self._ARCHITECTURES[name.lower()]
        except KeyError as error:
            raise ValueError(f"unsupported LINEAE HGNetV2 architecture: {name}") from error
        derivative = architecture != "B0"
        return_indices = [1, 2] if derivative else [1, 2, 3]
        self.core = HGNetv2(
            architecture,
            use_lab=use_lab,
            return_idx=return_indices,
            freeze_at=-1,
            freeze_norm=freeze_norm,
            pretrained=False,
        )
        core_channels = tuple(self.core._out_channels[index] for index in return_indices)
        self.p5 = None
        if derivative:
            last_channels = core_channels[-1]
            # A dense C -> C 3x3 projection makes the synthetic P5 larger than
            # HGNetV2-B0's complete native stage 4, reversing the intended
            # A/F/P/N capacity order.  Pooling followed by a learned pointwise
            # channel mixer preserves the P5 feature contract and is faster on
            # small feature maps than a depthwise projection on common GPUs.
            self.p5 = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(last_channels, last_channels, 1, bias=False),
                nn.BatchNorm2d(last_channels),
                nn.ReLU(inplace=True),
            )
            self.out_channels = (core_channels[0], core_channels[1], last_channels)
        else:
            self.out_channels = core_channels
        if weights_path is not None:
            self.checkpoint_report = self._load_checkpoint(Path(weights_path), architecture)
        self.set_trainable_depth(trainable_depth)

    def _load_checkpoint(self, path: Path, architecture: str) -> HGNetCheckpointReport:
        if not path.is_file():
            raise FileNotFoundError(f"HGNetV2 checkpoint not found: {path}")
        source = unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        target = self.core.state_dict()
        loaded = {}
        shape_mismatches = []
        for key, value in source.items():
            if key in target and target[key].shape == value.shape:
                loaded[key] = value
            elif key in target:
                shape_mismatches.append(
                    f"{key}: checkpoint={tuple(value.shape)}, model={tuple(target[key].shape)}"
                )
        # PyTorch BatchNorm creates ``num_batches_tracked`` when old checkpoints
        # omit it, so those buffers are neither source tensors nor architecture
        # mismatches.
        missing = sorted(
            key for key in set(target) - set(loaded)
            if not key.endswith(".num_batches_tracked")
        )
        unexpected = sorted(set(source) - set(target))
        strict = architecture == "B0"
        if strict:
            self.core.load_state_dict(source, strict=True)
            missing = []
            unexpected = []
            shape_mismatches = []
            loaded = dict(source)
        else:
            incompatible = self.core.load_state_dict(loaded, strict=False)
            actual_missing = {
                key for key in incompatible.missing_keys
                if not key.endswith(".num_batches_tracked")
            }
            if actual_missing != set(missing) or incompatible.unexpected_keys:
                raise RuntimeError("internal error while applying partial HGNetV2 checkpoint")
        return HGNetCheckpointReport(
            path=path,
            architecture=architecture,
            source_tensor_count=len(source),
            loaded_keys=tuple(sorted(loaded)),
            missing_keys=tuple(missing),
            unexpected_keys=tuple(unexpected),
            shape_mismatches=tuple(sorted(shape_mismatches)),
            strict=strict,
        )

    @property
    def num_blocks(self) -> int:
        return len(self.core.stages)

    def set_trainable_depth(self, depth: int) -> Sequence[nn.Parameter]:
        self.core.requires_grad_(False)
        if depth < 0:
            modules = []
        elif depth == 0 or depth >= self.num_blocks:
            modules: Sequence[nn.Module] = [self.core]
        else:
            modules = list(self.core.stages[-depth:])
        for module in modules:
            module.requires_grad_(True)
        return [parameter for parameter in self.core.parameters() if parameter.requires_grad]

    def forward(self, images: Tensor) -> list[Tensor]:
        if images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError(f"LINEAE HGNetV2 inputs must be divisible by 32, got {images.shape[-2:]}")
        features = list(self.core(images))
        if self.p5 is not None:
            features.append(self.p5(features[-1]))
        self.validate_features(images, features)
        self.last_feature_shapes = [tuple(feature.shape) for feature in features]
        return features


__all__ = ["HGNetCheckpointReport", "HGNetV2Backbone"]
