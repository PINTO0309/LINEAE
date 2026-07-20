"""Compare matched bounded no-KD and online-KD LINEAE training profiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from util.artifact_validation import (
    validate_training_profile_comparison,
    validate_training_profile_report,
)
from util.experiment import sha256_file


MATCHED_FIELDS = (
    "variant",
    "device",
    "gpu",
    "python",
    "torch",
    "cuda_runtime",
    "cudnn",
    "amp",
    "seed",
    "epoch",
    "start_global_step",
    "warmup",
    "iterations",
    "batch_size",
    "gradient_accumulation_steps",
    "optimizer_fused",
    "trainable_depth",
    "trainable_parameters",
)


def _read_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read training profile: {path}") from error
    validate_training_profile_report(report, require_cuda=False)
    return report


def _same(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"profiles have mismatched {label}: {actual!r} != {expected!r}")


def _mean(report: dict, metric: str) -> float:
    return float(report["phase_ms"][metric]["mean"])


def _ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"cannot calculate {label} from a non-positive baseline")
    return numerator / denominator


def compare(baseline_path: Path, distilled_path: Path) -> dict:
    baseline = _read_report(baseline_path)
    distilled = _read_report(distilled_path)
    if baseline["teacher"] is not None:
        raise ValueError("baseline profile must not contain a teacher")
    if distilled["teacher"] is None:
        raise ValueError("distilled profile must contain an online teacher")

    for field in MATCHED_FIELDS:
        _same(distilled.get(field), baseline.get(field), field)
    for field in ("annotation_sha256", "samples"):
        _same(distilled["dataset"].get(field), baseline["dataset"].get(field), f"dataset.{field}")
    _same(
        distilled["initialization"]["sha256"],
        baseline["initialization"]["sha256"],
        "initialization.sha256",
    )
    baseline_input_sizes = [sample["input_size"] for sample in baseline["samples"]]
    distilled_input_sizes = [sample["input_size"] for sample in distilled["samples"]]
    _same(distilled_input_sizes, baseline_input_sizes, "measured input-size sequence")

    phase_delta_ms = {
        metric: _mean(distilled, metric) - _mean(baseline, metric)
        for metric in (
            "transfer_ms",
            "student_supervised_ms",
            "teacher_forward_ms",
            "kd_loss_ms",
            "online_kd_ms",
            "backward_ms",
            "optimizer_ms",
            "step_ms",
        )
    }
    baseline_step = _mean(baseline, "step_ms")
    distilled_step = _mean(distilled, "step_ms")
    baseline_throughput = _mean(baseline, "throughput_images_per_second")
    distilled_throughput = _mean(distilled, "throughput_images_per_second")
    baseline_peak = baseline["peak_memory_mib"]
    distilled_peak = distilled["peak_memory_mib"]
    if (baseline_peak is None) != (distilled_peak is None):
        raise ValueError("profiles disagree on peak-memory availability")
    peak_delta = (
        float(distilled_peak["mean"]) - float(baseline_peak["mean"])
        if baseline_peak is not None
        else None
    )
    direct_kd = _mean(distilled, "online_kd_ms")
    observed_overhead = distilled_step - baseline_step
    report = {
        "format": "lineae_training_profile_comparison_v1",
        "baseline": {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
            "config": baseline["config"],
            "resolved_config_sha256": baseline["resolved_config_sha256"],
        },
        "distilled": {
            "path": str(distilled_path.resolve()),
            "sha256": sha256_file(distilled_path),
            "config": distilled["config"],
            "resolved_config_sha256": distilled["resolved_config_sha256"],
            "teacher_sha256": distilled["teacher"]["sha256"],
        },
        "matched_context": {
            field: baseline.get(field)
            for field in MATCHED_FIELDS
        },
        "dataset_annotation_sha256": baseline["dataset"]["annotation_sha256"],
        "initialization_sha256": baseline["initialization"]["sha256"],
        "input_sizes": baseline_input_sizes,
        "phase_delta_ms": phase_delta_ms,
        "direct_online_kd_ms": direct_kd,
        "direct_online_kd_fraction": float(distilled["online_kd_fraction"]["mean"]),
        "observed_step_overhead_ms": observed_overhead,
        "observed_step_overhead_percent": 100.0 * observed_overhead / baseline_step,
        "step_time_ratio": _ratio(distilled_step, baseline_step, "step-time ratio"),
        "throughput_ratio": _ratio(
            distilled_throughput,
            baseline_throughput,
            "throughput ratio",
        ),
        "peak_memory_delta_mib": peak_delta,
    }
    for key, value in report.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"comparison produced non-finite {key}")
    validate_training_profile_comparison(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--distilled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.distilled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
