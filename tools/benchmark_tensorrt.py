"""Build and benchmark a fixed-shape TensorRT engine with ``trtexec``."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from tools.deployment_parity import compare_line_sets
from util.artifact_validation import validate_onnx_export_report
from util.experiment import sha256_file
from util.onnx_runtime import create_ort_session


def _metric(output: str, label: str):
    match = re.search(rf"{re.escape(label)}\s*[=:]\s*([0-9.]+)\s*ms", output)
    return float(match.group(1)) if match else None


def benchmark(args) -> dict:
    executable = shutil.which(args.trtexec)
    if executable is None:
        raise FileNotFoundError(f"TensorRT trtexec executable not found: {args.trtexec}")
    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    onnx_report_path = args.onnx_report or args.onnx.with_suffix(".export.json")
    if not onnx_report_path.is_file():
        raise FileNotFoundError(
            f"ONNX export report not found: {onnx_report_path}; export before TensorRT"
        )
    onnx_report = json.loads(onnx_report_path.read_text(encoding="utf-8"))
    onnx_hash = sha256_file(args.onnx)
    validate_onnx_export_report(
        onnx_report,
        expected_onnx=onnx_hash,
        require_simplified=True,
    )
    checkpoint_hash = onnx_report.get("checkpoint_sha256")
    input_shape = onnx_report["input_shape"]
    seed = int(onnx_report["seed"])
    generator = torch.Generator().manual_seed(seed)
    input_array = torch.randn(*input_shape, generator=generator).numpy()
    reference_session, _, _, _ = create_ort_session(
        ort,
        args.onnx,
        require_cuda=False,
    )
    input_name = reference_session.get_inputs()[0].name
    output_names = [output.name for output in reference_session.get_outputs()]
    reference_values = reference_session.run(None, {input_name: input_array})
    reference = dict(zip(output_names, reference_values, strict=True))

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lineae-trtexec-") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "input.bin"
        output_path = temporary_path / "output.json"
        input_array.tofile(input_path)
        command = [
            executable,
            f"--onnx={args.onnx}",
            f"--saveEngine={args.engine}",
            f"--warmUp={args.warmup_ms}",
            f"--duration={args.duration_seconds}",
            "--useCudaGraph",
            "--noDataTransfers",
            "--noTF32",
            f"--loadInputs={input_name}:{input_path}",
            f"--exportOutput={output_path}",
        ]
        if args.fp16:
            command.append("--fp16")
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        combined = completed.stdout + "\n" + completed.stderr
        if completed.returncode:
            raise RuntimeError(
                f"trtexec failed with exit code {completed.returncode}:\n{combined[-4000:]}"
            )
        exported = json.loads(output_path.read_text(encoding="utf-8"))
    actual = {}
    for output in exported:
        dimensions = tuple(int(value) for value in output["dimensions"].split("x"))
        actual[output["name"]] = np.asarray(
            output["values"],
            dtype=np.float32,
        ).reshape(dimensions)
    missing_outputs = {"pred_logits", "pred_lines"} - actual.keys()
    if missing_outputs:
        raise RuntimeError(f"TensorRT output lacks tensors: {sorted(missing_outputs)}")
    parity = compare_line_sets(
        reference["pred_logits"],
        reference["pred_lines"],
        actual["pred_logits"],
        actual["pred_lines"],
        atol=args.atol,
        rtol=args.rtol,
        max_outlier_fraction=args.max_outlier_fraction,
    )
    latency = {
        "mean": _metric(combined, "mean"),
        "median": _metric(combined, "median"),
        "percentile_90": _metric(combined, "percentile(90%)"),
        "percentile_95": _metric(combined, "percentile(95%)"),
        "percentile_99": _metric(combined, "percentile(99%)"),
    }
    if latency["median"] is None or latency["percentile_95"] is None:
        raise RuntimeError("trtexec output lacks required median/p95 latency metrics")
    result = {
        "format": "lineae_tensorrt_benchmark_v1",
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": onnx_hash,
        "onnx_report": str(onnx_report_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "engine": str(args.engine.resolve()),
        "engine_sha256": sha256_file(args.engine),
        "fp16": args.fp16,
        "tf32_disabled": True,
        "reference_runtime": "onnxruntime-cpu",
        "onnxruntime_version": ort.__version__,
        "input_shape": input_shape,
        "seed": seed,
        "warmup_ms": args.warmup_ms,
        "duration_seconds": args.duration_seconds,
        "latency_ms": latency,
        **parity,
        "command": command,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.log.write_text(combined, encoding="utf-8")
    if not all(parity["parity"].values()):
        raise RuntimeError(f"TensorRT parity failed: {parity['max_abs_error']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-report", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--duration-seconds", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--max-outlier-fraction", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(benchmark(args), indent=2))


if __name__ == "__main__":
    main()
