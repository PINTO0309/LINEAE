"""Optimizer and progressive-unfreezing helpers."""

from __future__ import annotations

import torch


def trainable_depth_for_epoch(
    *,
    epoch: int,
    total_blocks: int,
    initial_depth: int,
    initial_freeze_epochs: int,
    unfreeze_interval: int,
    progressive: bool,
) -> int:
    if total_blocks <= 0:
        return 0
    if progressive and initial_freeze_epochs > 0 and epoch < initial_freeze_epochs:
        return -1
    if initial_depth < 0:
        return -1
    if initial_depth == 0 or initial_depth >= total_blocks:
        return total_blocks
    if not progressive or unfreeze_interval <= 0:
        return initial_depth
    additions = (epoch - initial_freeze_epochs) // unfreeze_interval
    return min(total_blocks, initial_depth + additions)


def build_lr_scheduler(args, optimizer: torch.optim.Optimizer):
    name = getattr(args, "lr_scheduler", "multistep").lower()
    step_unit = getattr(args, "scheduler_step_unit", "epoch").lower()
    if step_unit not in {"epoch", "optimizer"}:
        raise ValueError(f"unsupported scheduler_step_unit: {step_unit}")
    scale = getattr(args, "optimizer_steps_per_epoch", 1) if step_unit == "optimizer" else 1
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs * scale),
            eta_min=getattr(args, "min_lr", 1e-7),
        )
    if name == "multistep":
        milestones = args.lr_drop_list
        if isinstance(milestones, int):
            milestones = [milestones]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[milestone * scale for milestone in milestones],
            gamma=0.1,
        )
    raise ValueError(f"unsupported lr_scheduler: {name}")
