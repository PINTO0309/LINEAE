"""Profile bounded real LINEAE training steps without producing a trained model."""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import BatchImageCollateFunction, build_dataset
from engine import train_one_epoch
from main import build_frozen_teacher, create, data_loader_options
from models.lineae.distillation import resolve_distillation_temperature_steps
from util.artifact_validation import validate_training_profile_report
from util.experiment import config_fingerprint, sha256_file
from util.get_param_dicts import build_adamw_optimizer
from util.model_ema import ModelEMA
from util.slconfig import DictAction, SLConfig
from util.training_schedule import trainable_depth_for_epoch


PROFILE_METRICS = (
    "transfer_ms",
    "student_supervised_ms",
    "teacher_forward_ms",
    "kd_loss_ms",
    "online_kd_ms",
    "backward_ms",
    "optimizer_ms",
    "step_ms",
    "throughput_images_per_second",
)


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": statistics.fmean(values),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def _checkpoint_record(path_value) -> dict | None:
    if not path_value:
        return None
    path = Path(path_value)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def profile(args) -> dict:
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if args.global_step < 0:
        raise ValueError("global step must be non-negative")
    config = SLConfig.fromfile(args.config)
    if args.options:
        config.merge_from_dict(args.options)
    config.coco_path = str(args.coco_path)
    config.device = args.device
    config.amp = args.amp
    config.seed = args.seed
    config.num_workers = args.num_workers
    config.distributed = False
    config.rank = 0
    config.world_size = 1
    config.verify_optimizer_step = False
    config.output_dir = ""
    config.gradient_accumulation_steps = 1
    config.distill_teacher_cache_dir = ""
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        config.batch_size_train = args.batch_size
    if isinstance(config.eval_spatial_size, int):
        config.eval_spatial_size = [config.eval_spatial_size, config.eval_spatial_size]

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, _ = create(config, "modelname")
    criterion = create(config, "criterionname")
    model.to(device)
    criterion.to(device)
    teacher_model, distillation_criterion = build_frozen_teacher(config, device)

    total_blocks = getattr(model.backbone, "num_blocks", 0)
    trainable_depth = trainable_depth_for_epoch(
        epoch=args.epoch,
        total_blocks=total_blocks,
        initial_depth=getattr(config, "backbone_trainable_layers", 0),
        initial_freeze_epochs=getattr(config, "initial_freeze_epochs", 0),
        unfreeze_interval=getattr(config, "unfreeze_interval", 0),
        progressive=getattr(config, "progressive_unfreeze", False),
    )
    if hasattr(model.backbone, "set_trainable_depth"):
        model.backbone.set_trainable_depth(trainable_depth)
    optimizer = build_adamw_optimizer(config, model, device)
    scaler = torch.amp.GradScaler(
        str(device),
        enabled=config.amp,
        init_scale=getattr(config, "amp_init_scale", 65536.0),
    )
    ema = None
    if getattr(config, "use_ema", False):
        ema = ModelEMA(model, decay=config.ema_decay, device=device)

    # Isolate data-order and augmentation randomness from optional teacher
    # construction so matched no-KD/KD profiles see the same batches.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sampler_generator = torch.Generator().manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed + 1)
    dataset = build_dataset("train", config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train,
        sampler=RandomSampler(dataset, generator=sampler_generator),
        drop_last=True,
        collate_fn=BatchImageCollateFunction(
            base_size=config.eval_spatial_size[0],
            base_size_repeat=3 if getattr(config, "multi_scale_train", True) else None,
            minimum_tokens=(
                config.num_queries
                if getattr(config, "multi_scale_train", True)
                else None
            ),
            feature_strides=config.feat_strides,
        ),
        generator=loader_generator,
        **data_loader_options(config, device),
    )
    config.train_multiscale_scales = list(
        loader.collate_fn.scales or [config.eval_spatial_size[0]]
    )
    requested_steps = args.warmup + args.iterations
    if len(loader) < requested_steps:
        raise ValueError(
            f"training loader has {len(loader)} batches but {requested_steps} are required"
        )
    config.optimizer_steps_per_epoch = len(loader)
    config.distill_temperature_steps_resolved = None
    if distillation_criterion is not None:
        config.distill_temperature_steps_resolved = resolve_distillation_temperature_steps(
            config.distill_temperature_steps,
            optimizer_steps_per_epoch=config.optimizer_steps_per_epoch,
            epochs=config.epochs,
        )
        distillation_criterion.set_temperature_steps(
            config.distill_temperature_steps_resolved
        )
    samples = []
    _, completed_steps, _ = train_one_epoch(
        model,
        criterion,
        loader,
        optimizer,
        device,
        args.epoch,
        config.clip_max_norm,
        args=config,
        max_steps=requested_steps,
        scaler=scaler,
        start_global_step=args.global_step,
        teacher_model=teacher_model,
        distillation_criterion=distillation_criterion,
        ema_m=ema,
        step_profile_callback=samples.append,
    )
    if completed_steps != requested_steps or len(samples) != requested_steps:
        raise RuntimeError(
            f"profile completed {completed_steps} optimizer steps and {len(samples)} samples; "
            f"expected {requested_steps}"
        )
    measured = samples[args.warmup:]
    if not all(sample["optimizer_stepped"] for sample in measured):
        raise RuntimeError("a measured optimizer step was skipped")
    phase_summary = {
        metric: _summary([float(sample[metric]) for sample in measured])
        for metric in PROFILE_METRICS
    }
    peak_values = [sample["peak_memory_mib"] for sample in measured]
    peak_memory = (
        _summary([float(value) for value in peak_values])
        if all(value is not None for value in peak_values)
        else None
    )
    kd_fraction = [
        float(sample["online_kd_ms"]) / float(sample["step_ms"])
        for sample in measured
    ]
    annotation = args.coco_path / "annotations" / "lines_train2017.json"
    report = {
        "format": "lineae_training_profile_v1",
        "config": str(Path(args.config).resolve()),
        "resolved_config_sha256": config_fingerprint(config),
        "variant": getattr(config, "variant", None),
        "dataset": {
            "root": str(args.coco_path.resolve()),
            "annotation_sha256": sha256_file(annotation),
            "samples": len(dataset),
        },
        "initialization": _checkpoint_record(getattr(config, "backbone_weights", None)),
        "teacher": (
            _checkpoint_record(getattr(config, "distill_teacher_checkpoint", None))
            if teacher_model is not None
            else None
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "amp": bool(config.amp),
        "seed": seed,
        "epoch": args.epoch,
        "start_global_step": args.global_step,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "batch_size": config.batch_size_train,
        "gradient_accumulation_steps": 1,
        "optimizer_fused": optimizer.lineae_fused,
        "trainable_depth": trainable_depth,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "distill_temperature_steps_resolved": (
            config.distill_temperature_steps_resolved
        ),
        "phase_ms": phase_summary,
        "peak_memory_mib": peak_memory,
        "online_kd_fraction": _summary(kd_fraction),
        "samples": measured,
    }
    validate_training_profile_report(report, require_cuda=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--coco-path", type=Path, default=Path("data/wireframe_processed"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--options", nargs="+", action=DictAction)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(profile(args), indent=2))


if __name__ == "__main__":
    main()
