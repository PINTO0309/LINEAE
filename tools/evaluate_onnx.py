"""Evaluate ONNX on Wireframe/YorkUrban and gate final sAP against PyTorch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, SequentialSampler

from datasets import BatchImageCollateFunction, LineEvaluator, build_dataset
from util.artifact_validation import (
    validate_evaluation_report,
    validate_onnx_export_report,
)
from util.deployment import resolve_num_select
from util.experiment import sha256_file
from util.image_preprocess import validate_image_preprocess_schema
from util.onnx_runtime import create_ort_session
from util.slconfig import SLConfig


def _dataset_argument(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    return name, Path(path)


def compare_ap(onnx_report: dict, torch_report: dict, tolerance: float) -> dict:
    if tolerance < 0:
        raise ValueError("AP tolerance must be non-negative")
    checkpoint_hash = onnx_report.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError("ONNX evaluation lacks a valid checkpoint SHA-256")
    if checkpoint_hash != torch_report.get("checkpoint_sha256"):
        raise ValueError("ONNX and PyTorch evaluations use different checkpoints")
    if onnx_report.get("num_select") != torch_report.get("num_select"):
        raise ValueError("ONNX and PyTorch evaluations use different num_select values")
    if onnx_report.get("image_preprocess_schema") != torch_report.get(
        "image_preprocess_schema"
    ):
        raise ValueError("ONNX and PyTorch evaluations use different preprocessing")
    deltas = {}
    for dataset in ("wireframe", "york"):
        actual = onnx_report.get("datasets", {}).get(dataset)
        expected = torch_report.get("datasets", {}).get(dataset)
        if actual is None or expected is None:
            raise ValueError(f"both evaluations must include {dataset}")
        for field in ("annotation_sha256", "samples"):
            if actual.get(field) != expected.get(field):
                raise ValueError(f"ONNX/PyTorch {dataset}.{field} mismatch")
        deltas[dataset] = {}
        for metric in ("deploy_sap5", "deploy_sap10", "deploy_sap15"):
            actual_value = float(actual[metric])
            expected_value = float(expected[metric])
            if not math.isfinite(actual_value) or not math.isfinite(expected_value):
                raise ValueError(f"non-finite AP value for {dataset}.{metric}")
            deltas[dataset][metric] = abs(actual_value - expected_value)
    maximum = max(delta for values in deltas.values() for delta in values.values())
    result = {"tolerance": tolerance, "absolute_delta": deltas, "maximum_delta": maximum}
    if maximum > tolerance:
        raise RuntimeError(
            f"ONNX final-AP parity failed: maximum delta {maximum:.6f} > {tolerance:.6f}"
        )
    return result


def evaluate(args) -> dict:
    import onnxruntime as ort

    onnx_path = Path(args.onnx)
    parity_path = args.onnx_report or onnx_path.with_suffix(".parity.json")
    if not onnx_path.is_file() or not parity_path.is_file():
        raise FileNotFoundError("ONNX model and export parity report are required")
    export_report = json.loads(parity_path.read_text(encoding="utf-8"))
    validate_onnx_export_report(
        export_report,
        expected_onnx=sha256_file(onnx_path),
        require_simplified=True,
    )

    config = SLConfig.fromfile(args.config)
    validate_image_preprocess_schema(config.image_preprocess_schema)
    expected_config = str(Path(args.config).resolve())
    if export_report.get("config") != expected_config:
        raise ValueError("ONNX model was exported with a different config")
    if export_report.get("image_preprocess_schema") != config.image_preprocess_schema:
        raise ValueError("ONNX model was exported with different image preprocessing")
    num_select = resolve_num_select(
        config.num_select,
        config.num_queries,
        export_report.get("num_select"),
    )
    input_shape = export_report.get("input_shape")
    if not isinstance(input_shape, list) or len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError("ONNX export must have a fixed batch-one input shape")
    spatial_size = int(input_shape[-1])
    if input_shape[-2] != spatial_size:
        raise ValueError("ONNX input must be square")
    configured = config.eval_spatial_size
    configured = configured if isinstance(configured, int) else configured[0]
    if spatial_size != configured:
        config.enforce_variant_input = False
    config.eval_spatial_size = (spatial_size, spatial_size)

    session, available_providers, requested_providers, provider_options = create_ort_session(
        ort,
        onnx_path,
        require_cuda=args.cuda_ort,
    )
    input_name = session.get_inputs()[0].name
    results = {}
    for name, dataset_root in args.dataset:
        config.coco_path = str(dataset_root)
        dataset = build_dataset("val", config)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=SequentialSampler(dataset),
            drop_last=False,
            collate_fn=BatchImageCollateFunction(base_size=spatial_size),
            num_workers=args.num_workers,
        )
        evaluator = LineEvaluator(max_predictions=num_select)
        for samples, targets in loader:
            logits, lines = session.run(None, {input_name: samples.numpy()})
            outputs = {
                "pred_logits": torch.from_numpy(logits),
                "pred_lines": torch.from_numpy(lines),
            }
            evaluator.update(outputs, targets)
        evaluator.accumulate()
        annotation = dataset_root / "annotations" / "lines_val2017.json"
        results[name] = {
            "root": str(dataset_root.resolve()),
            "annotation_sha256": sha256_file(annotation),
            "samples": len(dataset),
            **{
                f"deploy_{metric}": value
                for metric, value in evaluator.sap_results.items()
            },
        }

    report = {
        "format": "lineae_onnx_evaluation_v3",
        "config": expected_config,
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": sha256_file(onnx_path),
        "checkpoint_sha256": export_report.get("checkpoint_sha256"),
        "num_select": num_select,
        "configured_num_select": int(config.num_select),
        "sap_protocol": "deployment_topk",
        "image_preprocess_schema": config.image_preprocess_schema,
        "opencv_version": cv2.__version__,
        "available_providers": available_providers,
        "requested_providers": requested_providers,
        "provider_options": provider_options,
        "cpu_ep_fallback_disabled": bool(args.cuda_ort),
        "providers": session.get_providers(),
        "datasets": results,
    }
    if args.torch_report is not None:
        torch_report = json.loads(args.torch_report.read_text(encoding="utf-8"))
        validate_evaluation_report(
            torch_report,
            expected_checkpoint=export_report.get("checkpoint_sha256"),
        )
        if torch_report.get("config") != expected_config:
            raise ValueError("PyTorch evaluation used a different config")
        report["torch_report"] = str(args.torch_report.resolve())
        report["torch_report_sha256"] = sha256_file(args.torch_report)
        report["ap_parity"] = compare_ap(report, torch_report, args.max_ap_delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-report", type=Path)
    parser.add_argument("--torch-report", type=Path, required=True)
    parser.add_argument("--dataset", type=_dataset_argument, action="append", required=True)
    parser.add_argument("--cuda-ort", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-ap-delta", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
