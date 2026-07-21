"""Export a fixed-shape batch-1 LINEAE model and verify ONNX Runtime parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import onnx
import onnxruntime as ort
import onnxsim
import torch
from torch import nn

from main import create
from models.lineae.backbones.base import unwrap_state_dict
from models.lineae.linea_utils import select_top_line_predictions
from tools.deployment_parity import compare_line_sets
from util.deployment import resolve_num_select
from util.experiment import sha256_file
from util.image_preprocess import (
    validate_checkpoint_image_preprocess,
    validate_image_preprocess_schema,
)
from util.onnx_runtime import create_ort_session
from util.slconfig import SLConfig


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_select: int):
        super().__init__()
        self.model = model.deploy()
        self.num_select = int(num_select)

    def forward(self, images):
        outputs = self.model(images)
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
    wrapper = ExportWrapper(model, num_select).eval()
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
    session, available_providers, requested_providers, provider_options = create_ort_session(
        ort,
        args.output,
        require_cuda=args.cuda_ort,
    )
    actual_logits, actual_lines = session.run(None, {"images": images.numpy()})
    expected_logits = reference_logits.detach().cpu().numpy()
    expected_lines = reference_lines.detach().cpu().numpy()
    parity = compare_line_sets(
        expected_logits,
        expected_lines,
        actual_logits,
        actual_lines,
        atol=args.atol,
        rtol=args.rtol,
        max_outlier_fraction=args.max_outlier_fraction,
    )
    result = {
        "format": "lineae_onnx_export_v2",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)) if args.checkpoint else None,
        "onnx": str(args.output.resolve()),
        "onnx_sha256": sha256_file(args.output),
        "opset": args.opset,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "onnxsim_version": onnxsim.__version__,
        "onnx_simplified": simplified,
        "deploy_mode": True,
        "seed": args.seed,
        "input_shape": list(images.shape),
        "num_select": num_select,
        "configured_num_select": int(config.num_select),
        "image_preprocess_schema": config.image_preprocess_schema,
        "opencv_version": cv2.__version__,
        "num_select_source": "cli" if num_select_override is not None else "config",
        "output_shapes": {
            "pred_logits": list(reference_logits.shape),
            "pred_lines": list(reference_lines.shape),
        },
        "available_providers": available_providers,
        "requested_providers": requested_providers,
        "provider_options": provider_options,
        "cpu_ep_fallback_disabled": bool(args.cuda_ort),
        "providers": session.get_providers(),
        **parity,
    }
    report_path = args.report or args.output.with_suffix(".parity.json")
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not all(parity["parity"].values()):
        raise RuntimeError(f"ONNX parity failed: {result['max_abs_error']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--spatial-size", type=int)
    parser.add_argument(
        "--num-select",
        "--topk",
        dest="num_select",
        type=int,
        help="number of top-scoring line queries embedded in the ONNX outputs; defaults to config.num_select",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-outlier-fraction", type=float, default=0.005)
    parser.add_argument("--cuda-ort", action="store_true")
    parser.add_argument(
        "--disable-onnxsim",
        action="store_true",
        help="skip graph simplification; parity is still required",
    )
    args = parser.parse_args()
    print(json.dumps(export_and_verify(args), indent=2))


if __name__ == "__main__":
    main()
