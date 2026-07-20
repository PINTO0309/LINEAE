"""Identify non-dominated LINEAE variants from evaluation/benchmark archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from util.artifact_validation import (
    validate_evaluation_report,
    validate_torch_benchmark_report,
)
from util.experiment import sha256_file


def _model_argument(value: str):
    try:
        variant, evaluation, benchmark = value.split("=", 1)[0], *value.split("=", 1)[1].split(",", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("model must be VARIANT=EVALUATION_JSON,BENCHMARK_JSON") from error
    return variant, Path(evaluation), Path(benchmark)


def analyze(records):
    points = []
    for variant, evaluation_path, benchmark_path in records:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        evaluation_hash = evaluation.get("checkpoint_sha256")
        benchmark_hash = benchmark.get("checkpoint_sha256")
        if not evaluation_hash or evaluation_hash != benchmark_hash:
            raise ValueError(
                f"{variant}: evaluation and benchmark checkpoint SHA-256 differ"
            )
        validate_evaluation_report(evaluation, expected_checkpoint=evaluation_hash)
        validate_torch_benchmark_report(
            benchmark,
            expected_checkpoint=evaluation_hash,
            require_cuda=True,
            require_samples=True,
        )
        points.append({
            "variant": variant,
            "sap10": float(evaluation["datasets"]["wireframe"]["sap10"]),
            "latency_p50_ms": float(benchmark["latency_ms"]["p50"]),
            "peak_memory_mib": float(benchmark["peak_memory_mib"]),
            "parameters": int(benchmark["parameters"]),
            "checkpoint_sha256": evaluation_hash,
            "evaluation": str(evaluation_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "benchmark": str(benchmark_path),
            "benchmark_sha256": sha256_file(benchmark_path),
        })
    for point in points:
        dominators = []
        for candidate in points:
            no_worse = (
                candidate["sap10"] >= point["sap10"]
                and candidate["latency_p50_ms"] <= point["latency_p50_ms"]
                and candidate["peak_memory_mib"] <= point["peak_memory_mib"]
            )
            strictly_better = (
                candidate["sap10"] > point["sap10"]
                or candidate["latency_p50_ms"] < point["latency_p50_ms"]
                or candidate["peak_memory_mib"] < point["peak_memory_mib"]
            )
            if no_worse and strictly_better:
                dominators.append(candidate["variant"])
        point["pareto"] = not dominators
        point["dominated_by"] = dominators
    return {
        "format": "lineae_pareto_v1",
        "points": points,
        "pareto_variants": [p["variant"] for p in points if p["pareto"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=_model_argument, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
