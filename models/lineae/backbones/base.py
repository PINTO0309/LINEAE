"""Common backbone contract used by LINEAE.

New code in this file is part of LINEAE.  Backbones must return P3/P4/P5 feature
maps in increasing stride order so the detector never has to branch on a model
name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from torch import Tensor, nn


@dataclass(frozen=True)
class CheckpointLoadReport:
    path: Path
    tensor_count: int
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()

    @property
    def strict(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys


def unwrap_state_dict(checkpoint: object) -> Mapping[str, Tensor]:
    """Return a tensor state dict from common checkpoint wrapper formats."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")

    state: object = checkpoint
    inference_model = checkpoint.get("inference_model")
    if inference_model == "ema_model":
        ema = checkpoint.get("ema_model")
        if not isinstance(ema, Mapping) or not isinstance(ema.get("model"), Mapping):
            raise TypeError("checkpoint selects EMA inference weights but EMA state is invalid")
        state = ema["model"]
    elif inference_model not in {None, "model"}:
        raise ValueError(f"unknown checkpoint inference model: {inference_model!r}")
    else:
        for key in ("model", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break

    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise TypeError("checkpoint state_dict must map string keys to tensors")

    tensors = {key: value for key, value in state.items() if isinstance(value, Tensor)}
    if not tensors:
        raise ValueError("checkpoint contains no tensors")
    if len(tensors) != len(state):
        non_tensor = sorted(key for key, value in state.items() if not isinstance(value, Tensor))
        raise TypeError(f"checkpoint state_dict contains non-tensor entries: {non_tensor[:5]}")
    return tensors


class LINEAEBackbone(nn.Module, ABC):
    out_channels: tuple[int, int, int]
    out_strides: tuple[int, int, int] = (8, 16, 32)
    preprocess_profile: str
    checkpoint_report: CheckpointLoadReport | None = None

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        """Number of ordered units exposed to progressive unfreezing."""

    @abstractmethod
    def set_trainable_depth(self, depth: int) -> Sequence[nn.Parameter]:
        """Enable final units: ``-1`` none, ``0`` all, positive final ``depth``."""

    def validate_features(self, images: Tensor, features: Sequence[Tensor]) -> None:
        if len(features) != 3:
            raise RuntimeError(f"backbone must return three features, got {len(features)}")
        image_h, image_w = images.shape[-2:]
        for index, (feature, channels, stride) in enumerate(
            zip(features, self.out_channels, self.out_strides, strict=True)
        ):
            expected = (image_h // stride, image_w // stride)
            if feature.ndim != 4:
                raise RuntimeError(f"P{index + 3} must be BCHW, got {tuple(feature.shape)}")
            if feature.shape[1] != channels or feature.shape[-2:] != expected:
                raise RuntimeError(
                    f"P{index + 3} contract violation: got {tuple(feature.shape)}, "
                    f"expected channels={channels}, spatial={expected}"
                )
