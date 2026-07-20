"""Aggregate repeated LINEAE runs into uncertainty-aware Pareto evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from util.artifact_validation import (
    validate_evaluation_report,
    validate_torch_benchmark_report,
)
from util.experiment import sha256_file


@dataclass(frozen=True)
class RepeatedRun:
    label: str
    seed: int
    run_dir: Path
    evaluation: Path
    benchmark: Path


def run_argument(value: str) -> RepeatedRun:
    try:
        identity, paths = value.split("=", 1)
        label, seed_text = identity.rsplit(":", 1)
        run_dir, evaluation, benchmark = paths.split(",", 2)
        seed = int(seed_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "run must be LABEL:SEED=RUN_DIR,EVALUATION_JSON,BENCHMARK_JSON"
        ) from error
    if not label or seed < 0 or not run_dir or not evaluation or not benchmark:
        raise argparse.ArgumentTypeError(
            "run must contain a label, non-negative seed, and three paths"
        )
    return RepeatedRun(
        label=label,
        seed=seed,
        run_dir=Path(run_dir),
        evaluation=Path(evaluation),
        benchmark=Path(benchmark),
    )


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("repeated metric values must be finite and non-empty")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    degrees_of_freedom = len(values) - 1
    critical = _T_CRITICAL_95.get(degrees_of_freedom, 1.96)
    margin = critical * standard_deviation / math.sqrt(len(values))
    return {
        "values": values,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95": [mean - margin, mean + margin],
    }


def _read_json(path: Path, label: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error


def _load_run(record: RepeatedRun) -> dict:
    completion_path = record.run_dir / "run_complete.json"
    manifest_path = record.run_dir / "run_manifest.json"
    checkpoint_path = record.run_dir / "checkpoint_best.pth"
    completion = _read_json(completion_path, "run completion")
    manifest = _read_json(manifest_path, "run manifest")
    if completion.get("format") != "lineae_run_completion_v1":
        raise ValueError(f"{record.label}:{record.seed} has an unsupported run completion")
    if completion.get("status") != "full":
        raise ValueError(f"{record.label}:{record.seed} is not a full training run")
    if not checkpoint_path.is_file():
        raise ValueError(f"{record.label}:{record.seed} has no best checkpoint")
    checkpoint_hash = sha256_file(checkpoint_path)
    if completion.get("best_checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"{record.label}:{record.seed} completion/checkpoint hash mismatch")
    training = manifest.get("training")
    if not isinstance(training, dict) or training.get("seed") != record.seed:
        raise ValueError(f"{record.label}:{record.seed} run-manifest seed mismatch")
    config = manifest.get("config_file")
    if not isinstance(config, str) or not config:
        raise ValueError(f"{record.label}:{record.seed} run manifest lacks config_file")

    evaluation = _read_json(record.evaluation, "evaluation report")
    benchmark = _read_json(record.benchmark, "benchmark report")
    validate_evaluation_report(
        evaluation,
        expected_checkpoint=checkpoint_hash,
        expected_config=config,
    )
    validate_torch_benchmark_report(
        benchmark,
        expected_checkpoint=checkpoint_hash,
        expected_config=config,
        require_cuda=True,
        require_samples=True,
    )
    teacher = manifest.get("checkpoints", {}).get("teacher")
    if not isinstance(teacher, dict) or not teacher.get("exists"):
        raise ValueError(f"{record.label}:{record.seed} has no recorded teacher")
    teacher_hash = teacher.get("sha256")
    if not isinstance(teacher_hash, str) or len(teacher_hash) != 64:
        raise ValueError(f"{record.label}:{record.seed} has no teacher SHA-256")
    return {
        "record": record,
        "completion": completion,
        "manifest": manifest,
        "evaluation": evaluation,
        "benchmark": benchmark,
        "checkpoint_hash": checkpoint_hash,
        "teacher_hash": teacher_hash,
        "config": config,
    }


def _same(value, expected, label: str) -> None:
    if value != expected:
        raise ValueError(f"repeated runs have mismatched {label}: {value!r} != {expected!r}")


def _dominates_mean(candidate: dict, point: dict) -> bool:
    no_worse = (
        candidate["sap10"]["mean"] >= point["sap10"]["mean"]
        and candidate["latency_p50_ms"]["mean"] <= point["latency_p50_ms"]["mean"]
        and candidate["peak_memory_mib"]["mean"] <= point["peak_memory_mib"]["mean"]
    )
    strictly_better = (
        candidate["sap10"]["mean"] > point["sap10"]["mean"]
        or candidate["latency_p50_ms"]["mean"] < point["latency_p50_ms"]["mean"]
        or candidate["peak_memory_mib"]["mean"] < point["peak_memory_mib"]["mean"]
    )
    return no_worse and strictly_better


def _dominates_robust(candidate: dict, point: dict) -> bool:
    no_worse = (
        candidate["sap10"]["ci95"][0] >= point["sap10"]["ci95"][1]
        and candidate["latency_p50_ms"]["ci95"][1]
        <= point["latency_p50_ms"]["ci95"][0]
        and candidate["peak_memory_mib"]["ci95"][1]
        <= point["peak_memory_mib"]["ci95"][0]
    )
    strictly_better = (
        candidate["sap10"]["ci95"][0] > point["sap10"]["ci95"][1]
        or candidate["latency_p50_ms"]["ci95"][1]
        < point["latency_p50_ms"]["ci95"][0]
        or candidate["peak_memory_mib"]["ci95"][1]
        < point["peak_memory_mib"]["ci95"][0]
    )
    return no_worse and strictly_better


def analyze(
    records: list[RepeatedRun],
    *,
    min_seeds: int = 3,
    baseline_label: str | None = None,
    minimum_sap10_gain: float = 0.0,
) -> dict:
    if min_seeds < 3:
        raise ValueError("repeated Pareto analysis requires at least three seeds")
    if not math.isfinite(minimum_sap10_gain) or minimum_sap10_gain < 0:
        raise ValueError("minimum sAP10 gain must be finite and non-negative")
    if not records:
        raise ValueError("at least one repeated run is required")
    loaded = [_load_run(record) for record in records]
    grouped = defaultdict(list)
    for run in loaded:
        grouped[run["record"].label].append(run)
    if baseline_label is not None and baseline_label not in grouped:
        raise ValueError(f"baseline label is absent: {baseline_label}")

    seed_sets = {}
    for label, runs in grouped.items():
        seeds = [run["record"].seed for run in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{label} contains a duplicate seed")
        if len(seeds) < min_seeds:
            raise ValueError(f"{label} has {len(seeds)} seeds; at least {min_seeds} are required")
        seed_sets[label] = set(seeds)
    expected_seeds = next(iter(seed_sets.values()))
    for label, seeds in seed_sets.items():
        _same(seeds, expected_seeds, f"seed set for {label}")

    first = loaded[0]
    first_evaluation = first["evaluation"]
    first_benchmark = first["benchmark"]
    expected_teacher = first["teacher_hash"]
    expected_datasets = {
        dataset: {
            field: first_evaluation["datasets"][dataset][field]
            for field in ("annotation_sha256", "samples")
        }
        for dataset in ("wireframe", "york")
    }
    protocol_fields = (
        "device",
        "gpu",
        "torch",
        "cuda_runtime",
        "cudnn",
        "batch_size",
        "amp",
        "warmup",
        "iterations",
        "deploy_mode",
    )
    expected_protocol = {field: first_benchmark.get(field) for field in protocol_fields}
    for run in loaded:
        _same(run["teacher_hash"], expected_teacher, "teacher checkpoint SHA-256")
        for dataset, expected in expected_datasets.items():
            current = run["evaluation"]["datasets"][dataset]
            for field, value in expected.items():
                _same(current[field], value, f"{dataset}.{field}")
        current_protocol = {
            field: run["benchmark"].get(field)
            for field in protocol_fields
        }
        _same(current_protocol, expected_protocol, "benchmark protocol")

    points = []
    run_values = {}
    for label, runs in grouped.items():
        runs.sort(key=lambda run: run["record"].seed)
        first_label_run = runs[0]
        for run in runs[1:]:
            _same(run["config"], first_label_run["config"], f"config for {label}")
            _same(
                run["benchmark"]["spatial_size"],
                first_label_run["benchmark"]["spatial_size"],
                f"spatial size for {label}",
            )
            _same(
                run["benchmark"]["parameters"],
                first_label_run["benchmark"]["parameters"],
                f"parameter count for {label}",
            )
        sap10 = [float(run["evaluation"]["datasets"]["wireframe"]["sap10"]) for run in runs]
        latency = [float(run["benchmark"]["latency_ms"]["p50"]) for run in runs]
        memory = [float(run["benchmark"]["peak_memory_mib"]) for run in runs]
        run_values[label] = {
            run["record"].seed: (sap, latency_value, memory_value)
            for run, sap, latency_value, memory_value in zip(
                runs, sap10, latency, memory, strict=True
            )
        }
        points.append({
            "label": label,
            "config": first_label_run["config"],
            "spatial_size": first_label_run["benchmark"]["spatial_size"],
            "parameters": first_label_run["benchmark"]["parameters"],
            "sap10": _summary(sap10),
            "latency_p50_ms": _summary(latency),
            "peak_memory_mib": _summary(memory),
            "runs": [
                {
                    "seed": run["record"].seed,
                    "checkpoint_sha256": run["checkpoint_hash"],
                    "run_dir": str(run["record"].run_dir),
                    "evaluation": str(run["record"].evaluation),
                    "evaluation_sha256": sha256_file(run["record"].evaluation),
                    "benchmark": str(run["record"].benchmark),
                    "benchmark_sha256": sha256_file(run["record"].benchmark),
                }
                for run in runs
            ],
        })

    for point in points:
        mean_dominators = [
            candidate["label"]
            for candidate in points
            if candidate is not point and _dominates_mean(candidate, point)
        ]
        robust_dominators = [
            candidate["label"]
            for candidate in points
            if candidate is not point and _dominates_robust(candidate, point)
        ]
        point["pareto_mean"] = not mean_dominators
        point["dominated_by_mean"] = mean_dominators
        point["pareto_robust"] = not robust_dominators
        point["dominated_by_robust"] = robust_dominators

    if baseline_label is not None:
        baseline_values = run_values[baseline_label]
        for point in points:
            if point["label"] == baseline_label:
                point["paired_vs_baseline"] = None
                continue
            candidate_values = run_values[point["label"]]
            sap_delta = [
                candidate_values[seed][0] - baseline_values[seed][0]
                for seed in sorted(expected_seeds)
            ]
            latency_delta = [
                candidate_values[seed][1] - baseline_values[seed][1]
                for seed in sorted(expected_seeds)
            ]
            memory_delta = [
                candidate_values[seed][2] - baseline_values[seed][2]
                for seed in sorted(expected_seeds)
            ]
            sap_summary = _summary(sap_delta)
            latency_summary = _summary(latency_delta)
            memory_summary = _summary(memory_delta)
            point["paired_vs_baseline"] = {
                "sap10_delta": sap_summary,
                "latency_p50_ms_delta": latency_summary,
                "peak_memory_mib_delta": memory_summary,
                "accuracy_gain_confident": (
                    sap_summary["ci95"][0] > minimum_sap10_gain
                ),
                "latency_gain_confident": latency_summary["ci95"][1] < 0.0,
                "memory_gain_confident": memory_summary["ci95"][1] < 0.0,
            }

    return {
        "format": "lineae_repeated_pareto_v1",
        "seed_count": len(expected_seeds),
        "seeds": sorted(expected_seeds),
        "minimum_sap10_gain": minimum_sap10_gain,
        "baseline_label": baseline_label,
        "teacher_checkpoint_sha256": expected_teacher,
        "datasets": expected_datasets,
        "benchmark_protocol": expected_protocol,
        "points": points,
        "pareto_mean": [point["label"] for point in points if point["pareto_mean"]],
        "pareto_robust": [point["label"] for point in points if point["pareto_robust"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=run_argument, action="append", required=True)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--baseline-label")
    parser.add_argument("--minimum-sap10-gain", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.run,
        min_seeds=args.min_seeds,
        baseline_label=args.baseline_label,
        minimum_sap10_gain=args.minimum_sap10_gain,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
