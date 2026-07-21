"""Generate a LINEAE model card from archived evaluation and benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from util.artifact_validation import (
    validate_evaluation_report,
    validate_tensorrt_report,
    validate_torch_benchmark_report,
)
from util.experiment import sha256_file


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generate(args) -> str:
    evaluation = _read(args.evaluation)
    torch_benchmark = _read(args.torch_benchmark)
    tensorrt_benchmark = _read(args.tensorrt_benchmark)
    checkpoint_hash = sha256_file(args.checkpoint)
    if evaluation.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("evaluation report checkpoint hash does not match the model")
    if torch_benchmark.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("PyTorch benchmark checkpoint hash does not match the model")
    if tensorrt_benchmark.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("TensorRT benchmark checkpoint hash does not match the model")
    validate_evaluation_report(evaluation, expected_checkpoint=checkpoint_hash)
    validate_torch_benchmark_report(
        torch_benchmark,
        expected_checkpoint=checkpoint_hash,
        require_cuda=True,
        require_samples=True,
    )
    validate_tensorrt_report(
        tensorrt_benchmark,
        expected_checkpoint=checkpoint_hash,
    )
    pareto = _read(args.pareto_report)
    if pareto.get("format") != "lineae_pareto_v1":
        raise ValueError("unsupported Pareto report format")
    matching_points = [
        point
        for point in pareto.get("points", [])
        if point.get("variant") == args.variant
        and point.get("checkpoint_sha256") == checkpoint_hash
    ]
    if len(matching_points) != 1 or matching_points[0].get("pareto") is not True:
        raise ValueError("model is not a hash-matched Pareto variant")
    if args.variant not in pareto.get("pareto_variants", []):
        raise ValueError("Pareto report does not retain the requested variant")
    point = matching_points[0]
    if point.get("evaluation_sha256") != sha256_file(args.evaluation):
        raise ValueError("Pareto report does not match the evaluation artifact")
    if point.get("benchmark_sha256") != sha256_file(args.torch_benchmark):
        raise ValueError("Pareto report does not match the benchmark artifact")
    datasets = evaluation["datasets"]
    for dataset in ("wireframe", "york"):
        if dataset not in datasets:
            raise ValueError(f"evaluation report lacks {dataset}")
    complexity = torch_benchmark.get("complexity") or {}
    torch_latency = torch_benchmark["latency_ms"]
    trt_latency = tensorrt_benchmark["latency_ms"]
    initialization = args.initialization or "See the archived run manifest."
    limitations = args.limitations or (
        "Metrics apply only to the named datasets, preprocessing, resolution, and hardware. "
        "Line detections may degrade under domain shift or severe occlusion."
    )
    return f"""# LINEAE-{args.variant} model card

Status: qualified Pareto candidate

## Identity

- Checkpoint: `{args.checkpoint}`
- SHA-256: `{checkpoint_hash}`
- Initialization: {initialization}
- Input: `{torch_benchmark['spatial_size']}x{torch_benchmark['spatial_size']}`, batch 1
- Parameters: `{torch_benchmark['parameters']}`
- FLOPs: `{complexity.get('flops', 'not recorded')}`
- MACs: `{complexity.get('macs', 'not recorded')}`

## Accuracy

| Dataset | sAP5 | sAP10 | sAP15 | Annotation SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Wireframe | {datasets['wireframe']['sap5']:.3f} | {datasets['wireframe']['sap10']:.3f} | {datasets['wireframe']['sap15']:.3f} | `{datasets['wireframe']['annotation_sha256']}` |
| YorkUrban | {datasets['york']['sap5']:.3f} | {datasets['york']['sap10']:.3f} | {datasets['york']['sap15']:.3f} | `{datasets['york']['annotation_sha256']}` |

## Batch-1 latency and memory

| Runtime | Precision | p50 (ms) | p95 (ms) | Peak memory (MiB) | Hardware |
| --- | --- | ---: | ---: | ---: | --- |
| PyTorch | {'AMP' if torch_benchmark['amp'] else 'FP32'} | {torch_latency['p50']:.3f} | {torch_latency['p95']:.3f} | {torch_benchmark.get('peak_memory_mib', float('nan')):.1f} | {torch_benchmark.get('gpu')} |
| TensorRT | {'FP16' if tensorrt_benchmark['fp16'] else 'FP32'} | {trt_latency.get('median') or float('nan'):.3f} | {trt_latency.get('percentile_95') or float('nan'):.3f} | n/a | See TensorRT report |

PyTorch protocol: {torch_benchmark['warmup']} warm-up iterations and {torch_benchmark['iterations']} measured iterations. Raw reports: `{args.evaluation}`, `{args.torch_benchmark}`, `{args.tensorrt_benchmark}`.
Pareto qualification: `{args.pareto_report}`.

## Known limitations

{limitations}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=["A", "F", "P", "N", "S", "M", "L", "X", "XL", "2XL", "3XL"],
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--torch-benchmark", type=Path, required=True)
    parser.add_argument("--tensorrt-benchmark", type=Path, required=True)
    parser.add_argument("--pareto-report", type=Path, required=True)
    parser.add_argument("--initialization")
    parser.add_argument("--limitations")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    card = generate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(card, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
