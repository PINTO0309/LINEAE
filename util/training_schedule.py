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
    total_units = int(args.epochs * scale)
    warmup_units = 0
    if getattr(args, "use_warmup", False):
        configured_warmup = int(getattr(args, "warmup_iters", 0))
        if configured_warmup <= 0:
            raise ValueError("warmup_iters must be positive when use_warmup=True")
        # LinearWarmup advances once per successful optimizer update. Epoch-step
        # schedulers retain their legacy epoch semantics, while optimizer-step
        # schedulers consume only the part of the fixed run horizon after warmup.
        if step_unit == "optimizer":
            if configured_warmup >= total_units:
                raise ValueError(
                    "warmup_iters must be smaller than the total optimizer-step "
                    f"horizon ({configured_warmup} >= {total_units})"
                )
            warmup_units = configured_warmup
    # At the update that completes warmup, engine.py advances the downstream
    # scheduler once after the optimizer has used the final warmup LR. The +1
    # keeps its final state aligned with the end of the overall run.
    post_warmup_units = total_units - warmup_units + (1 if warmup_units else 0)
    args.lr_scheduler_total_units_resolved = total_units
    args.lr_scheduler_warmup_units_resolved = warmup_units
    args.lr_scheduler_post_warmup_units_resolved = post_warmup_units
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, post_warmup_units),
            eta_min=getattr(args, "min_lr", 1e-7),
        )
    if name == "multistep":
        milestones = args.lr_drop_list
        if isinstance(milestones, int):
            milestones = [milestones]
        resolved_milestones = [milestone * scale for milestone in milestones]
        if warmup_units:
            if any(milestone < warmup_units for milestone in resolved_milestones):
                raise ValueError(
                    "optimizer-step LR milestones must not occur before warmup ends"
                )
            resolved_milestones = [
                milestone - warmup_units + 1
                for milestone in resolved_milestones
            ]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=resolved_milestones,
            gamma=0.1,
        )
    raise ValueError(f"unsupported lr_scheduler: {name}")
