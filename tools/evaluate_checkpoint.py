"""Evaluate one LINEAE checkpoint on Wireframe and/or YorkUrban with sAP metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, SequentialSampler

from datasets import (
    BatchImageCollateFunction,
    DualLineEvaluator,
    SAP_EVALUATION_PROTOCOL,
    build_dataset,
)
from engine import test
from main import create
from models.lineae.backbones.base import unwrap_state_dict
from util.deployment import resolve_num_select
from util.experiment import sha256_file
from util.image_preprocess import (
    validate_checkpoint_image_preprocess,
    validate_image_preprocess_schema,
)
from util.slconfig import SLConfig


def _dataset_argument(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    return name, Path(path)


def evaluate(args) -> dict:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    config = SLConfig.fromfile(args.config)
    validate_image_preprocess_schema(config.image_preprocess_schema)
    num_select = resolve_num_select(
        config.num_select,
        config.num_queries,
        getattr(args, "num_select", None),
    )
    config.pretrained = False
    config.amp = args.amp
    config.batch_size_val = args.batch_size
    model, postprocessors = create(config, "modelname")
    criterion = create(config, "criterionname")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_image_preprocess(checkpoint)
    model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    device = torch.device(args.device)
    model.to(device)
    criterion.to(device)

    results = {}
    for name, dataset_root in args.dataset:
        config.coco_path = str(dataset_root)
        dataset = build_dataset("val", config)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(dataset),
            drop_last=False,
            collate_fn=BatchImageCollateFunction(base_size=config.eval_spatial_size[0]),
            num_workers=args.num_workers,
        )
        metrics = test(
            model,
            criterion,
            postprocessors,
            DualLineEvaluator(deploy_max_predictions=num_select),
            loader,
            device,
            str(args.output.parent),
            args=config,
        )
        annotation = dataset_root / "annotations" / "lines_val2017.json"
        results[name] = {
            "root": str(dataset_root.resolve()),
            "annotation_sha256": sha256_file(annotation),
            "samples": len(dataset),
            **metrics,
        }
    report = {
        "format": "lineae_evaluation_v3",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": str(device),
        "amp": args.amp,
        "batch_size": args.batch_size,
        "num_select": num_select,
        "configured_num_select": int(config.num_select),
        "num_queries": int(config.num_queries),
        "sap_protocol": SAP_EVALUATION_PROTOCOL,
        "image_preprocess_schema": config.image_preprocess_schema,
        "opencv_version": cv2.__version__,
        "datasets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        type=_dataset_argument,
        action="append",
        required=True,
        help="repeatable NAME=PATH, e.g. wireframe=data/wireframe_processed",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--num-select",
        "--topk",
        dest="num_select",
        type=int,
        help="deployment top-k used in the report; defaults to config.num_select",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
