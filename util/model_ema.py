"""Exponential moving-average weights with resumable state."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from torch import nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class ModelEMA:
    """Keep an evaluation-only EMA copy without changing optimizer topology."""

    def __init__(self, model: nn.Module, *, decay: float, device: torch.device) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        self.module = copy.deepcopy(_unwrap(model)).to(device).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = _unwrap(model).state_dict()
        averaged = self.module.state_dict()
        if source.keys() != averaged.keys():
            raise RuntimeError("EMA/source model state dictionaries have different keys")
        # The first enabled update synchronizes an EMA that may have been dormant
        # during ema_epoch warm-up. Subsequent updates use the configured decay.
        decay = self.decay if self.num_updates else 0.0
        for name, value in averaged.items():
            current = source[name].detach().to(device=value.device)
            if value.is_floating_point() or value.is_complex():
                value.mul_(decay).add_(current.to(dtype=value.dtype), alpha=1.0 - decay)
            else:
                value.copy_(current)
        self.num_updates += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "model": self.module.state_dict(),
            "decay": self.decay,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if "model" not in state:
            raise ValueError("EMA checkpoint is missing model state")
        saved_decay = float(state.get("decay", self.decay))
        if saved_decay != self.decay:
            raise ValueError(
                f"EMA decay mismatch: checkpoint={saved_decay}, current={self.decay}"
            )
        model_state = state["model"]
        if not isinstance(model_state, Mapping):
            raise TypeError("EMA model state must be a mapping")
        self.module.load_state_dict(model_state, strict=True)
        self.num_updates = int(state.get("num_updates", 0))
        self.module.eval().requires_grad_(False)


__all__ = ["ModelEMA"]
