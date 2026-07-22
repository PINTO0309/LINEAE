"""Reproducible training-state persistence for LINEAE."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist

from .image_preprocess import validate_checkpoint_image_preprocess
from models.lineae.backbones.base import unwrap_state_dict


CHECKPOINT_FORMAT_VERSION = 2

REQUIRED_RESUME_FIELDS = frozenset({
    "format_version",
    "model",
    "optimizer",
    "scheduler",
    "warmup_scheduler",
    "scaler",
    "epoch",
    "epoch_complete",
    "global_step",
    "sampler_epoch",
    "best_metric_name",
    "best_metric",
    "best_epoch",
    "inference_model",
    "config",
    "git",
    "rng_state",
})

RESUME_CRITICAL_FIELDS = (
    "modelname",
    "variant",
    "backbone",
    "backbone_weights",
    "backbone_checkpoint_sha256",
    "backbone_pyramid_channels",
    "synthetic_p5_schema",
    "backbone_trainable_layers",
    "use_lab",
    "freeze_norm",
    "use_checkpoint",
    "progressive_unfreeze",
    "initial_freeze_epochs",
    "unfreeze_interval",
    "dino_intermediate_layers",
    "num_classes",
    "hybrid_encoder",
    "hidden_dim",
    "in_channels_encoder",
    "feat_channels_decoder",
    "feat_strides",
    "num_feature_levels",
    "nheads",
    "dim_feedforward",
    "dropout",
    "transformer_activation",
    "pre_norm",
    "query_dim",
    "dec_n_points",
    "reg_max",
    "reg_scale",
    "eval_idx",
    "expansion",
    "depth_mult",
    "pe_temperatureH",
    "pe_temperatureW",
    "dec_layers",
    "num_queries",
    "num_select",
    "use_dn",
    "dn_number",
    "dn_line_noise_scale",
    "dn_line_noise_schema",
    "dn_label_noise_ratio",
    "criterionname",
    "criterion_type",
    "matcher_type",
    "set_cost_class",
    "set_cost_lines",
    "focal_alpha",
    "losses",
    "endpoint_invariant_lines",
    "endpoint_loss_schema",
    "weight_dict",
    "coco_path",
    "data_aug_scales",
    "data_aug_max_size",
    "data_aug_scales2_resize",
    "data_aug_scales2_crop",
    "image_preprocess_schema",
    "use_lmap",
    "multi_scale_train",
    "train_multiscale_scales",
    "use_photometric_distort",
    "photometric_distort_probability",
    "image_mean",
    "image_std",
    "eval_spatial_size",
    "seed",
    "world_size",
    "num_workers",
    "batch_size_train",
    "batch_size_val",
    "pin_memory",
    "prefetch_factor",
    "multiprocessing_sharing_strategy",
    "gradient_accumulation_steps",
    "training_profile",
    "amp",
    "lr",
    "betas",
    "weight_decay",
    "optimizer_fused",
    "model_parameters",
    "epochs",
    "lr_scheduler",
    "min_lr",
    "lr_drop_list",
    "scheduler_step_unit",
    "use_warmup",
    "warmup_iters",
    "clip_max_norm",
    "distill_weight",
    "distill_teacher_config",
    "distill_teacher_checkpoint",
    "distill_teacher_checkpoint_sha256",
    "distill_allow_unqualified_teacher",
    "distill_teacher_resize",
    "distill_matching_mode",
    "distill_confidence_threshold",
    "distill_top_k",
    "distill_match_cost_class",
    "distill_match_cost_line",
    "distill_teacher_gt_max_distance",
    "distill_confidence_power",
    "distill_class_weight",
    "distill_line_weight",
    "distill_feature_weight",
    "distill_feature_loss",
    "distill_teacher_feature_channels",
    "distill_warmup_steps",
    "distill_temperature_start",
    "distill_temperature_end",
    "distill_temperature_steps",
    "distill_temperature_steps_resolved",
    "selection_metric",
    "sap_evaluation_protocol",
    "selection_mode",
    "use_ema",
    "ema_decay",
    "ema_epoch",
    "eval_ema",
)


# Model/preprocessing semantics that must remain identical when a completed
# training checkpoint is used as fresh model initialization. Optimization,
# data augmentation, loss, unfreezing, distillation, and runtime settings are
# deliberately absent: those belong to the new run rather than the source run.
INITIALIZATION_CRITICAL_FIELDS = (
    "modelname",
    "variant",
    "backbone",
    "backbone_pyramid_channels",
    "synthetic_p5_schema",
    "use_lab",
    "batch_norm_type",
    "dino_intermediate_layers",
    "num_classes",
    "hybrid_encoder",
    "hidden_dim",
    "in_channels_encoder",
    "feat_channels_decoder",
    "feat_strides",
    "num_feature_levels",
    "nheads",
    "dim_feedforward",
    "dropout",
    "transformer_activation",
    "pre_norm",
    "query_dim",
    "dec_n_points",
    "reg_max",
    "reg_scale",
    "eval_idx",
    "expansion",
    "depth_mult",
    "pe_temperatureH",
    "pe_temperatureW",
    "dec_layers",
    "num_queries",
    "embed_init_tgt",
    "image_preprocess_schema",
    "image_mean",
    "image_std",
    "eval_spatial_size",
)


def _canonical_config_value(value):
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def collect_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def collect_distributed_rng_states() -> list[dict[str, Any]]:
    """Collect each rank's RNG state without making rank-zero state authoritative."""
    local_state = collect_rng_state()
    if not dist.is_available() or not dist.is_initialized():
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_state)
    if any(state is None for state in gathered):
        raise RuntimeError("failed to gather RNG state from every distributed rank")
    return [state for state in gathered if state is not None]


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def restore_checkpoint_rng_state(checkpoint: Mapping[str, Any]) -> None:
    """Restore the calling rank's RNG state, with legacy rank-zero fallback."""
    ranked_states = checkpoint.get("rng_state_by_rank")
    if ranked_states is None:
        restore_rng_state(checkpoint.get("rng_state"))
        return
    if not isinstance(ranked_states, (list, tuple)) or not all(
        isinstance(state, Mapping) for state in ranked_states
    ):
        raise TypeError("checkpoint rng_state_by_rank must contain one mapping per rank")
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if len(ranked_states) != world_size:
        raise ValueError(
            "checkpoint RNG world-size mismatch: "
            f"checkpoint={len(ranked_states)}, current={world_size}"
        )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    restore_rng_state(ranked_states[rank])


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(
                args,
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    revision = _run("git", "rev-parse", "HEAD")
    dirty_output = _run("git", "status", "--porcelain")
    return {"revision": revision, "dirty": bool(dirty_output and dirty_output != "unknown")}


def build_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    warmup_scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    args: Any,
    repo_root: Path,
    best_metric_name: str | None = None,
    best_metric: float | None = None,
    best_epoch: int | None = None,
    ema_model: Any = None,
    inference_model: str = "model",
    rng_state_by_rank: list[Mapping[str, Any]] | None = None,
    epoch_complete: bool = True,
) -> dict[str, Any]:
    if inference_model not in {"model", "ema_model"}:
        raise ValueError(f"unsupported inference model selector: {inference_model!r}")
    if inference_model == "ema_model" and ema_model is None:
        raise ValueError("inference_model='ema_model' requires EMA state")
    if rng_state_by_rank is not None and not rng_state_by_rank:
        raise ValueError("rng_state_by_rank must not be empty")
    rng_state = collect_rng_state() if rng_state_by_rank is None else rng_state_by_rank[0]
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "warmup_scheduler": warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": epoch,
        "epoch_complete": bool(epoch_complete),
        "global_step": global_step,
        "sampler_epoch": epoch,
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "inference_model": inference_model,
        "config": dict(vars(args)),
        "git": git_metadata(repo_root),
        # Keep the legacy rank-zero field so old read-only consumers remain
        # usable. Training resume selects the rank-specific state below.
        "rng_state": rng_state,
    }
    if rng_state_by_rank is not None:
        checkpoint["rng_state_by_rank"] = list(rng_state_by_rank)
    if ema_model is not None:
        checkpoint["ema_model"] = ema_model.state_dict()
    return checkpoint


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_resume_checkpoint(checkpoint: Mapping[str, Any], args: Any) -> None:
    if checkpoint.get("epoch_complete") is False:
        raise ValueError(
            "cannot resume a partial-epoch checkpoint; restart the bounded diagnostic "
            "or resume the latest completed-epoch checkpoint"
        )
    missing = sorted(REQUIRED_RESUME_FIELDS - checkpoint.keys())
    if missing:
        raise ValueError(f"resume checkpoint is missing required fields: {missing}")
    if checkpoint["epoch_complete"] is not True:
        raise ValueError("resume checkpoint epoch_complete must be boolean true")
    version = checkpoint.get("format_version", 0)
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version {version}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    validate_checkpoint_image_preprocess(checkpoint)
    saved_config = checkpoint["config"]
    if not isinstance(saved_config, Mapping):
        raise TypeError("resume checkpoint config must be a mapping")
    if not getattr(args, "eval", False):
        required_semantic_markers = {
            "dn_line_noise_schema": "corrected denoising-line noise",
            "endpoint_loss_schema": "non-degenerate endpoint-loss tie handling",
            "synthetic_p5_schema": "efficient synthetic-P5 architecture",
            "image_preprocess_schema": "OpenCV image preprocessing",
        }
        for field, description in required_semantic_markers.items():
            if getattr(args, field, None) and field not in saved_config:
                raise ValueError(
                    f"resume checkpoint predates the {description} schema; "
                    "start a new run from the backbone initialization weights"
                )
    if checkpoint["sampler_epoch"] != checkpoint["epoch"]:
        raise ValueError(
            "resume checkpoint sampler_epoch must match its completed epoch"
        )
    inference_model = checkpoint["inference_model"]
    if inference_model not in {"model", "ema_model"}:
        raise ValueError(
            f"unsupported resume inference model selector: {inference_model!r}"
        )
    if inference_model == "ema_model" and checkpoint.get("ema_model") is None:
        raise ValueError(
            "resume checkpoint selects EMA inference weights but has no EMA state"
        )
    for field in RESUME_CRITICAL_FIELDS:
        if field in saved_config and hasattr(args, field):
            saved = _canonical_config_value(saved_config[field])
            current = _canonical_config_value(getattr(args, field))
            if saved != current:
                raise ValueError(
                    f"resume config mismatch for {field}: checkpoint={saved!r}, "
                    f"current={current!r}"
                )


def validate_initialization_checkpoint(
    checkpoint: Mapping[str, Any], args: Any
) -> Mapping[str, torch.Tensor]:
    """Validate a completed full-model checkpoint for a fresh training run."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("initialization checkpoint must be a mapping")
    required = {"format_version", "model", "epoch", "epoch_complete", "config", "inference_model"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(
            f"initialization checkpoint is missing required fields: {missing}"
        )
    if checkpoint["epoch_complete"] is not True:
        raise ValueError("initialization checkpoint must represent a completed epoch")
    version = checkpoint.get("format_version", 0)
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported initialization checkpoint format version "
            f"{version}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    validate_checkpoint_image_preprocess(checkpoint)
    saved_config = checkpoint["config"]
    if not isinstance(saved_config, Mapping):
        raise TypeError("initialization checkpoint config must be a mapping")

    for field in INITIALIZATION_CRITICAL_FIELDS:
        if not hasattr(args, field):
            continue
        if field not in saved_config:
            raise ValueError(
                f"initialization checkpoint is missing model semantic field {field!r}"
            )
        saved = _canonical_config_value(saved_config[field])
        current = _canonical_config_value(getattr(args, field))
        if saved != current:
            raise ValueError(
                f"initialization config mismatch for {field}: "
                f"checkpoint={saved!r}, current={current!r}"
            )
    return unwrap_state_dict(checkpoint)


def initialize_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    args: Any,
) -> dict[str, Any]:
    """Strict-load only selected model weights after a mutation-free preflight."""
    source_state = validate_initialization_checkpoint(checkpoint, args)
    target_state = model.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    feature_projection_prefix = "distill_feature_projections."
    target_feature_projection_keys = {
        key for key in target_keys if key.startswith(feature_projection_prefix)
    }
    feature_kd_enabled = float(getattr(args, "distill_feature_weight", 0.0)) > 0
    allowed_missing = sorted(
        key
        for key in missing
        if feature_kd_enabled and key in target_feature_projection_keys
    )
    allowed_missing_set = set(allowed_missing)
    rejected_missing = sorted(set(missing) - allowed_missing_set)
    allowed_unexpected = sorted(
        key
        for key in unexpected
        if not target_feature_projection_keys
        and key.startswith(feature_projection_prefix)
    )
    allowed_unexpected_set = set(allowed_unexpected)
    rejected_unexpected = sorted(set(unexpected) - allowed_unexpected_set)
    incompatible = sorted(
        key
        for key in source_keys & target_keys
        if (
            source_state[key].shape != target_state[key].shape
            or source_state[key].dtype != target_state[key].dtype
        )
    )
    if rejected_missing or rejected_unexpected or incompatible:
        details = []
        if rejected_missing:
            details.append(f"missing={rejected_missing[:5]}")
        if rejected_unexpected:
            details.append(f"unexpected={rejected_unexpected[:5]}")
        if incompatible:
            key = incompatible[0]
            details.append(
                "shape_or_dtype="
                f"{key}: checkpoint={tuple(source_state[key].shape)}/{source_state[key].dtype}, "
                f"model={tuple(target_state[key].shape)}/{target_state[key].dtype}"
            )
        raise ValueError(
            "initialization checkpoint model state mismatch: " + "; ".join(details)
        )

    loadable_state = {
        key: value
        for key, value in source_state.items()
        if key not in allowed_unexpected_set
    }
    incompatible_keys = model.load_state_dict(loadable_state, strict=False)
    if (
        set(incompatible_keys.missing_keys) != allowed_missing_set
        or incompatible_keys.unexpected_keys
    ):
        raise RuntimeError(
            "initialization load did not match its preflight-approved feature keys"
        )
    return {
        "source_epoch": int(checkpoint["epoch"]),
        "inference_model": checkpoint["inference_model"],
        "tensor_count": len(source_state),
        "loaded_tensor_count": len(loadable_state),
        "new_state_keys": allowed_missing,
        "ignored_source_keys": allowed_unexpected,
        "strict_shared_state": True,
    }


def restore_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    warmup_scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    ema_model: Any = None,
) -> None:
    component_states = {
        "scheduler": scheduler is not None,
        "warmup_scheduler": warmup_scheduler is not None,
        "scaler": scaler is not None and scaler.is_enabled(),
        "ema_model": ema_model is not None,
    }
    for field, runtime_enabled in component_states.items():
        checkpoint_enabled = checkpoint.get(field) is not None
        if checkpoint_enabled != runtime_enabled:
            raise ValueError(
                f"resume {field} state mismatch: checkpoint_enabled="
                f"{checkpoint_enabled}, runtime_enabled={runtime_enabled}"
            )

    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_state(optimizer, device)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if warmup_scheduler is not None:
        warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler"])
    if scaler is not None and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])
    if ema_model is not None:
        ema_model.load_state_dict(checkpoint["ema_model"])
    restore_checkpoint_rng_state(checkpoint)
