"""Export and structurally validate a fixed-shape batch-1 LINEAE ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import onnx
import onnxsim
import torch
from torch import nn

from main import create
from models.lineae.backbones.base import unwrap_state_dict
from models.lineae.linea_utils import select_top_line_predictions
from util.deployment import resolve_num_select
from util.experiment import sha256_file
from util.image_preprocess import (
    validate_checkpoint_image_preprocess,
    validate_image_preprocess_schema,
)
from util.slconfig import SLConfig


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_select: int, num_queries: int):
        super().__init__()
        self.model = model.deploy()
        self.num_select = int(num_select)
        self.num_queries = int(num_queries)
        if not 0 < self.num_select <= self.num_queries:
            raise ValueError(
                "num_select must be in "
                f"[1, num_queries={self.num_queries}], got {self.num_select}"
            )
        self.uses_output_topk = self.num_select < self.num_queries

    def forward(self, images):
        outputs = self.model(images)
        if not self.uses_output_topk:
            return outputs["pred_logits"], outputs["pred_lines"]
        return select_top_line_predictions(
            outputs["pred_logits"], outputs["pred_lines"], self.num_select
        )


def export_and_verify(args) -> dict:
    config = SLConfig.fromfile(args.config)
    validate_image_preprocess_schema(config.image_preprocess_schema)
    num_select_override = getattr(args, "num_select", None)
    num_select = resolve_num_select(
        config.num_select,
        config.num_queries,
        num_select_override,
    )
    spatial_size = args.spatial_size
    if spatial_size is None:
        configured = config.eval_spatial_size
        spatial_size = configured if isinstance(configured, int) else configured[0]
    else:
        config.enforce_variant_input = False
    config.eval_spatial_size = (spatial_size, spatial_size)
    torch.manual_seed(args.seed)
    if args.checkpoint:
        config.pretrained = False
    model, _ = create(config, "modelname")
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint_image_preprocess(checkpoint)
        model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    model.eval()
    wrapper = ExportWrapper(model, num_select, config.num_queries).eval()
    generator = torch.Generator().manual_seed(args.seed)
    images = torch.randn(1, 3, spatial_size, spatial_size, generator=generator)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference_logits, reference_lines = wrapper(images)
        torch.onnx.export(
            wrapper,
            (images,),
            args.output,
            input_names=["images"],
            output_names=["pred_logits", "pred_lines"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    simplified = False
    if not getattr(args, "disable_onnxsim", False):
        graph, simplification_succeeded = onnxsim.simplify(graph)
        if not simplification_succeeded:
            raise RuntimeError("onnxsim validation failed")
        onnx.checker.check_model(graph)
        onnx.save(graph, args.output)
        simplified = True
    result = {
        "format": "lineae_onnx_export_v3",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)) if args.checkpoint else None,
        "onnx": str(args.output.resolve()),
        "onnx_sha256": sha256_file(args.output),
        "opset": args.opset,
        "onnx_version": onnx.__version__,
        "onnxsim_version": onnxsim.__version__,
        "onnx_simplified": simplified,
        "deploy_mode": True,
        "seed": args.seed,
        "input_shape": list(images.shape),
        "num_select": num_select,
        "num_queries": int(config.num_queries),
        "configured_num_select": int(config.num_select),
        "output_selection": (
            "class0_topk" if wrapper.uses_output_topk else "all_queries_passthrough"
        ),
        "image_preprocess_schema": config.image_preprocess_schema,
        "opencv_version": cv2.__version__,
        "num_select_source": "cli" if num_select_override is not None else "config",
        "output_shapes": {
            "pred_logits": list(reference_logits.shape),
            "pred_lines": list(reference_lines.shape),
        },
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON report path; no report is written by default",
    )
    parser.add_argument("--spatial-size", type=int)
    parser.add_argument(
        "--num-select",
        "--topk",
        dest="num_select",
        type=int,
        help=(
            "number of line queries in the ONNX outputs; defaults to "
            "config.num_select, and output TopK selection is omitted when this "
            "equals config.num_queries"
        ),
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-onnxsim",
        action="store_true",
        help="skip graph simplification; ONNX checker validation is still required",
    )
    args = parser.parse_args()
    result = export_and_verify(args)
    print(f"Exported ONNX: {result['onnx']}")
    print(f"Input shape: {result['input_shape']}")
    print(f"Output shapes: {result['output_shapes']}")
    print(f"Output selection: {result['output_selection']}")
    print(f"ONNX SHA-256: {result['onnx_sha256']}")
    if args.report is not None:
        print(f"Export report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
