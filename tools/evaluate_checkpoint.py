"""Evaluate and optionally render one LINEAE checkpoint on validation datasets."""

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
from util.validation_render import save_validation_renders


def _dataset_argument(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    return name, Path(path)


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def _render_directory_name(dataset_name: str) -> str:
    path = Path(dataset_name)
    if dataset_name in {".", ".."} or path.name != dataset_name:
        raise ValueError(
            "rendered dataset names must be single safe path components, "
            f"got {dataset_name!r}"
        )
    return dataset_name


def evaluate(args) -> dict:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    render_only = bool(getattr(args, "render_only", False))
    configured_output = getattr(args, "output", None)
    output_path = Path(configured_output) if configured_output is not None else None
    if not render_only and output_path is None:
        raise ValueError("--output is required unless --render-only is enabled")
    config = SLConfig.fromfile(args.config)
    validate_image_preprocess_schema(config.image_preprocess_schema)
    num_select = resolve_num_select(
        config.num_select,
        config.num_queries,
        getattr(args, "num_select", None),
    )
    render_count = int(getattr(args, "render_count", 0))
    render_score_threshold = float(getattr(args, "render_score_threshold", 0.3))
    render_endpoints = bool(getattr(args, "render_endpoints", False))
    configured_render_line_width = getattr(args, "render_line_width", None)
    render_line_width = (
        None
        if configured_render_line_width is None
        else int(configured_render_line_width)
    )
    requested_render_max = getattr(args, "render_max_predictions", None)
    render_max_predictions = (
        min(100, num_select)
        if requested_render_max is None
        else int(requested_render_max)
    )
    if render_count < 0:
        raise ValueError("render_count must be non-negative")
    if render_only and render_count == 0:
        raise ValueError("--render-only requires --render-count greater than zero")
    if not 0.0 <= render_score_threshold <= 1.0:
        raise ValueError("render_score_threshold must be in [0, 1]")
    if render_line_width is not None and render_line_width <= 0:
        raise ValueError("render_line_width must be positive")
    render_maximum = int(config.num_queries) if render_only else num_select
    if not 0 < render_max_predictions <= render_maximum:
        maximum_name = "num_queries" if render_only else "effective num_select"
        raise ValueError(f"render_max_predictions must be in [1, {maximum_name}]")

    dataset_arguments = list(args.dataset)
    dataset_names = [name for name, _ in dataset_arguments]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("dataset names must be unique")

    render_output_root = None
    render_directories = {}
    if render_count > 0:
        configured_render_root = getattr(args, "render_output_dir", None)
        render_output_root = (
            Path(configured_render_root)
            if configured_render_root is not None
            else (
                output_path.parent / f"{output_path.stem}_renders"
                if output_path is not None
                else checkpoint_path.parent / f"{checkpoint_path.stem}_renders"
            )
        )
        if render_output_root.exists() and (
            not render_output_root.is_dir() or render_output_root.is_symlink()
        ):
            raise FileExistsError(
                "refusing to use a render output root that is not a real "
                f"directory: {render_output_root}"
            )
        for name in dataset_names:
            render_dir = render_output_root / _render_directory_name(name)
            if render_dir.exists() and (
                not render_dir.is_dir() or render_dir.is_symlink()
            ):
                raise FileExistsError(
                    "refusing to replace a render output path that is not a real "
                    f"directory: {render_dir}"
                )
            render_directories[name] = render_dir

    config.pretrained = False
    config.amp = args.amp
    config.batch_size_val = args.batch_size
    model, postprocessors = create(config, "modelname")
    render_candidate_limit = (
        max(num_select, render_max_predictions) if render_only else num_select
    )
    postprocessors.num_select = render_candidate_limit
    criterion = None if render_only else create(config, "criterionname")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_image_preprocess(checkpoint)
    model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    device = torch.device(args.device)
    model.to(device)
    if criterion is not None:
        criterion.to(device)

    validation_datasets = []
    for name, dataset_root in dataset_arguments:
        config.coco_path = str(dataset_root)
        dataset = build_dataset("val", config)
        if render_count > len(dataset):
            raise ValueError(
                f"dataset {name!r} has {len(dataset)} validation samples, "
                f"fewer than --render-count={render_count}"
            )
        validation_datasets.append((name, dataset_root, dataset))

    results = {}
    for name, dataset_root, dataset in validation_datasets:
        metrics = {}
        if not render_only:
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=SequentialSampler(dataset),
                drop_last=False,
                collate_fn=BatchImageCollateFunction(
                    base_size=config.eval_spatial_size[0]
                ),
                num_workers=args.num_workers,
            )
            metrics = test(
                model,
                criterion,
                postprocessors,
                DualLineEvaluator(deploy_max_predictions=num_select),
                loader,
                device,
                str(output_path.parent),
                args=config,
            )
        rendered_path = None
        if render_count > 0:
            rendered_path = save_validation_renders(
                model=model,
                postprocessor=postprocessors,
                dataset=dataset,
                device=device,
                output_dir=render_directories[name],
                image_mean=config.image_mean,
                image_std=config.image_std,
                count=render_count,
                score_threshold=render_score_threshold,
                max_predictions=render_max_predictions,
                batch_size=min(args.batch_size, render_count),
                amp=args.amp,
                draw_endpoints=render_endpoints,
                line_width=render_line_width,
                replace_existing=True,
            )
        annotation = dataset_root / "annotations" / "lines_val2017.json"
        results[name] = {
            "root": str(dataset_root.resolve()),
            "annotation_sha256": sha256_file(annotation),
            "samples": len(dataset),
            **(
                {"render_dir": str(rendered_path.resolve())}
                if rendered_path is not None
                else {}
            ),
            **metrics,
        }
    report = {
        "format": "lineae_render_v1" if render_only else "lineae_evaluation_v3",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "amp": args.amp,
        "batch_size": args.batch_size,
        "num_select": num_select,
        "configured_num_select": int(config.num_select),
        "num_queries": int(config.num_queries),
        "image_preprocess_schema": config.image_preprocess_schema,
        "opencv_version": cv2.__version__,
        "render_only": render_only,
        "rendering": {
            "enabled": render_count > 0,
            "count": render_count,
            "score_threshold": render_score_threshold,
            "max_predictions": render_max_predictions,
            "candidate_limit": render_candidate_limit,
            "endpoints": render_endpoints,
            "line_width": render_line_width,
            "output_root": (
                str(render_output_root.resolve())
                if render_output_root is not None
                else None
            ),
        },
        "datasets": results,
    }
    if not render_only:
        report["checkpoint_sha256"] = sha256_file(checkpoint_path)
        report["sap_protocol"] = SAP_EVALUATION_PROTOCOL
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
    parser.add_argument(
        "--render-count",
        type=_nonnegative_integer,
        default=0,
        help="render the first N validation images per dataset; 0 disables rendering",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="skip full-dataset sAP evaluation and only render --render-count images",
    )
    parser.add_argument(
        "--render-score-threshold",
        type=_probability,
        default=0.3,
        help="minimum class-0 score drawn in prediction renders",
    )
    parser.add_argument(
        "--render-max-predictions",
        type=_positive_integer,
        help=(
            "maximum predictions drawn per image; defaults to min(100, num-select); "
            "render-only permits values up to num_queries"
        ),
    )
    parser.add_argument(
        "--render-endpoints",
        action="store_true",
        help="draw a filled point at both endpoints of every rendered line",
    )
    parser.add_argument(
        "--render-line-width",
        type=_positive_integer,
        help=(
            "line width in pixels for GT and prediction renders; "
            "defaults to automatic scaling based on image size"
        ),
    )
    parser.add_argument(
        "--render-output-dir",
        type=Path,
        help=(
            "render root; defaults to <output-stem>_renders beside the JSON report, "
            "or <checkpoint-stem>_renders in render-only mode without --output"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON evaluation report; required unless --render-only is enabled",
    )
    args = parser.parse_args()
    if args.render_only and args.render_count == 0:
        parser.error("--render-only requires --render-count greater than zero")
    if not args.render_only and args.output is None:
        parser.error("--output is required unless --render-only is enabled")
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
