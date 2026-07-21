"""Machine-readable provenance records for LINEAE runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from .training_state import git_metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(config) -> str:
    """Hash the fully merged config rather than only its top-level source file."""
    if hasattr(config, "_cfg_dict"):
        config = config._cfg_dict.to_dict()
    encoded = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _jsonable(value: Any):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _file_record(path_value: str | Path | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_experiment_records(
    *,
    args,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    repo_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = _jsonable(vars(args))
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dataset_root = Path(args.coco_path)
    annotation_records = []
    annotations = dataset_root / "annotations"
    if annotations.is_dir():
        for path in sorted(annotations.glob("*.json")):
            annotation_records.append(_file_record(path))

    report = getattr(model.backbone, "checkpoint_report", None)
    if report is not None:
        (output_dir / "backbone_load_report.json").write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    group_records = []
    for index, group in enumerate(optimizer.param_groups):
        group_records.append({
            "index": index,
            "lr": group["lr"],
            "weight_decay": group.get("weight_decay", 0.0),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
        })
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    accumulation = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
    cuda_device = (
        torch.device(args.device)
        if torch.cuda.is_available() and str(args.device).startswith("cuda")
        else None
    )
    cuda_properties = torch.cuda.get_device_properties(cuda_device) if cuda_device else None
    manifest = {
        "command": [sys.executable, *sys.argv],
        "config_file": str(Path(args.config_file).resolve()),
        "git": git_metadata(repo_root),
        "dataset": {
            "root": str(dataset_root.resolve()),
            "annotations": annotation_records,
        },
        "checkpoints": {
            "initialization": _file_record(getattr(args, "backbone_weights", None)),
            "resume": _file_record(getattr(args, "resume", None)),
            "teacher": _file_record(
                getattr(args, "distill_teacher_checkpoint", None)
                if getattr(args, "distill_weight", 0.0) > 0 else None
            ),
        },
        "training": {
            "seed": args.seed,
            "world_size": getattr(args, "world_size", 1),
            "batch_size_per_rank": getattr(args, "batch_size_train", None),
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": (
                getattr(
                    args,
                    "effective_batch_size_resolved",
                    getattr(args, "batch_size_train", 0)
                    * getattr(args, "world_size", 1)
                    * accumulation,
                )
            ),
            "reference_effective_batch_size": getattr(
                args, "recipe_reference_effective_batch_size", None
            ),
            "optimizer_steps_per_epoch": getattr(
                args, "optimizer_steps_per_epoch", None
            ),
            "total_optimizer_steps": getattr(
                args, "total_optimizer_steps_resolved", None
            ),
            "amp": args.amp,
            "optimizer": type(optimizer).__name__,
            "optimizer_fused": bool(getattr(optimizer, "lineae_fused", False)),
            "profile": getattr(args, "training_profile", None),
            "train_multiscale_scales": getattr(args, "train_multiscale_scales", None),
            "num_workers": getattr(args, "num_workers", 0),
            "pin_memory": bool(
                getattr(args, "pin_memory", True)
                and str(getattr(args, "device", "cpu")).startswith("cuda")
            ),
            "prefetch_factor": (
                getattr(args, "prefetch_factor", 2)
                if getattr(args, "num_workers", 0) > 0 else None
            ),
            "multiprocessing_sharing_strategy": (
                getattr(args, "multiprocessing_sharing_strategy", "file_system")
                if getattr(args, "num_workers", 0) > 0 else None
            ),
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "optimizer_groups": group_records,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(cuda_device) if cuda_device else None,
            "gpu_total_memory_mib": (
                cuda_properties.total_memory / (1024 ** 2) if cuda_properties else None
            ),
            "gpu_compute_capability": (
                f"{cuda_properties.major}.{cuda_properties.minor}"
                if cuda_properties else None
            ),
            "cudnn": torch.backends.cudnn.version(),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["config_fingerprint", "sha256_file", "write_experiment_records"]
