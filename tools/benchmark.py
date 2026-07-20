"""Reproducible batch-1 Torch latency and memory benchmark for LINEAE."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from main import create
from models.lineae.backbones.base import unwrap_state_dict
from models.lineae.linea_utils import select_top_line_predictions
from util.experiment import sha256_file
from util.profiler import stats as complexity_stats
from util.slconfig import SLConfig


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark(args) -> dict:
    config = SLConfig.fromfile(args.config)
    spatial_size = args.spatial_size
    if spatial_size is None:
        configured = config.eval_spatial_size
        spatial_size = configured if isinstance(configured, int) else configured[0]
    else:
        config.enforce_variant_input = False
    config.eval_spatial_size = (spatial_size, spatial_size)
    if args.checkpoint:
        config.pretrained = False
    model, _ = create(config, "modelname")
    checkpoint_hash = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
        checkpoint_hash = sha256_file(checkpoint_path)
    complexity = None if args.skip_flops else complexity_stats(model, config)
    model.deploy()
    device = torch.device(args.device)
    model.to(device).eval()
    images = torch.randn(1, 3, spatial_size, spatial_size, device=device)

    def inference():
        outputs = model(images)
        return select_top_line_predictions(
            outputs["pred_logits"], outputs["pred_lines"], config.num_select
        )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.amp.autocast(device.type, enabled=args.amp):
                inference()
        synchronize()
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            with torch.amp.autocast(device.type, enabled=args.amp):
                inference()
            synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    device_properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    result = {
        "format": "lineae_torch_benchmark_v1",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_mib": (
            device_properties.total_memory / (1024 ** 2) if device_properties else None
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "spatial_size": spatial_size,
        "batch_size": 1,
        "amp": args.amp,
        "deploy_mode": True,
        "num_select": int(config.num_select),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters": total_parameters,
        "complexity": complexity,
        "peak_memory_mib": peak_memory,
        "latency_ms": {
            "mean": statistics.fmean(timings),
            "p50": percentile(timings, 50),
            "p95": percentile(timings, 95),
            "min": min(timings),
            "max": max(timings),
        },
        "samples_ms": timings if args.include_samples else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spatial-size", type=int)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args)
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
