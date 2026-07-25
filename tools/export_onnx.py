"""Export and structurally validate a fixed-shape batch-1 LINEAE ONNX model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
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


def resolve_export_num_select(num_queries: int, override: int | None = None) -> int:
    """Default ONNX outputs to every query; an explicit CLI value filters them."""
    return resolve_num_select(num_queries, num_queries, override)


_LAYOUT_ONLY_OPS = frozenset(
    {"Concat", "Identity", "Reshape", "Squeeze", "Transpose", "Unsqueeze"}
)


def find_redundant_decoder_selection_chains(graph: onnx.ModelProto) -> list[str]:
    """Find legacy ``stack -> permute -> index`` decoder output selection.

    A one-element ``torch.stack`` is exported as an Unsqueeze (and sometimes a
    Concat), ``permute`` as Transpose, and ``[0]`` as Gather.  Walking only
    layout operators avoids treating the encoder proposal gathers or the
    public output TopK gather as this legacy decoder chain.
    """
    producers = {
        output: node
        for node in graph.graph.node
        for output in node.output
        if output
    }

    def layout_ancestors(value: str, visited: set[str]) -> set[str]:
        producer = producers.get(value)
        if producer is None or value in visited:
            return set()
        if producer.op_type not in _LAYOUT_ONLY_OPS:
            return set()
        visited.add(value)
        operations = {producer.op_type}
        for input_name in producer.input:
            operations.update(layout_ancestors(input_name, visited))
        return operations

    redundant = []
    for node in graph.graph.node:
        if node.op_type != "Gather" or "/decoder/" not in node.name:
            continue
        operations = layout_ancestors(node.input[0], set())
        if {"Unsqueeze", "Transpose"}.issubset(operations):
            redundant.append(node.name)
    return redundant


def _sparse_tensors(sparse_tensor: onnx.SparseTensorProto):
    yield sparse_tensor.values
    yield sparse_tensor.indices


def _graph_tensors(graph: onnx.GraphProto | onnx.FunctionProto):
    if isinstance(graph, onnx.GraphProto):
        yield from graph.initializer
        for sparse_initializer in graph.sparse_initializer:
            yield from _sparse_tensors(sparse_initializer)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("sparse_tensor"):
                yield from _sparse_tensors(attribute.sparse_tensor)
            for sparse_tensor in attribute.sparse_tensors:
                yield from _sparse_tensors(sparse_tensor)
            if attribute.HasField("g"):
                yield from _graph_tensors(attribute.g)
            for nested_graph in attribute.graphs:
                yield from _graph_tensors(nested_graph)


def _external_tensors(model: onnx.ModelProto):
    """Yield initializers and node-attribute tensors from the whole model."""
    yield from _graph_tensors(model.graph)
    for function in model.functions:
        yield from _graph_tensors(function)
    for training_info in model.training_info:
        yield from _graph_tensors(training_info.initialization)
        yield from _graph_tensors(training_info.algorithm)


def _external_location(tensor: onnx.TensorProto) -> str | None:
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        return None
    return next(
        (entry.value for entry in tensor.external_data if entry.key == "location"),
        None,
    )


def external_data_paths(
    model_path: Path, graph: onnx.ModelProto | None = None
) -> list[Path]:
    """Return validated external-data files referenced by an ONNX model.

    Exported sidecars must be relative to and remain below the model directory.
    This keeps the model portable and makes subsequent exact-file cleanup safe.
    """
    model_path = model_path.resolve()
    if graph is None:
        graph = onnx.load(str(model_path), load_external_data=False)
    model_directory = model_path.parent
    paths = set()
    for tensor in _external_tensors(graph):
        location = _external_location(tensor)
        if location is None:
            continue
        relative_path = Path(location)
        if relative_path.is_absolute():
            raise RuntimeError(
                f"ONNX external-data location must be relative: {location!r}"
            )
        resolved = (model_directory / relative_path).resolve()
        if resolved == model_directory or not resolved.is_relative_to(model_directory):
            raise RuntimeError(
                "ONNX external-data location escapes the model directory: "
                f"{location!r}"
            )
        paths.add(resolved)
    return sorted(paths)


def _set_external_location(graph: onnx.ModelProto, location: str) -> None:
    for tensor in _external_tensors(graph):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = location
                break
        else:
            entry = tensor.external_data.add()
            entry.key = "location"
            entry.value = location


def _set_external_slice(
    tensor: onnx.TensorProto,
    location: str,
    offset: int,
    length: int,
    checksum: str | None,
) -> None:
    del tensor.external_data[:]
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
        ("checksum", checksum),
    ):
        if value is None:
            continue
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    tensor.data_location = onnx.TensorProto.EXTERNAL


def consolidate_external_onnx(model_path: Path) -> onnx.ModelProto:
    """Combine external tensors without simplifying or inferring the graph."""
    model_path = model_path.resolve()
    graph = onnx.load(str(model_path), load_external_data=False)
    original_external_paths = external_data_paths(model_path, graph)
    if not original_external_paths:
        raise ValueError("consolidate_external_onnx requires an external-data model")

    final_data_path = model_path.with_name(model_path.name + ".data")
    if original_external_paths == [final_data_path]:
        onnx.checker.check_model(str(model_path))
        return graph

    model_directory = model_path.parent
    with tempfile.TemporaryDirectory(
        dir=model_directory, prefix=f".{model_path.stem}-consolidate-"
    ) as temporary_directory:
        temporary_directory_path = Path(temporary_directory)
        temporary_data_path = temporary_directory_path / "model.onnx.data"
        temporary_model_path = temporary_directory_path / model_path.name
        output_offset = 0
        with temporary_data_path.open("wb") as output_file:
            for tensor in _external_tensors(graph):
                location = _external_location(tensor)
                if location is None:
                    continue
                info = {entry.key: entry.value for entry in tensor.external_data}
                source_path = (model_directory / location).resolve()
                if not source_path.is_file():
                    raise RuntimeError(
                        f"ONNX external-data file is missing: {source_path}"
                    )
                source_size = source_path.stat().st_size
                source_offset = int(info.get("offset", 0))
                source_length = int(
                    info.get("length", source_size - source_offset)
                )
                if (
                    source_offset < 0
                    or source_length < 0
                    or source_offset + source_length > source_size
                ):
                    raise RuntimeError(
                        "ONNX external-data slice is out of bounds for "
                        f"{tensor.name!r}: offset={source_offset}, "
                        f"length={source_length}, file_size={source_size}"
                    )
                with source_path.open("rb") as source_file:
                    source_file.seek(source_offset)
                    remaining = source_length
                    while remaining:
                        chunk = source_file.read(min(16 * 1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError(
                                "unexpected end of ONNX external-data file: "
                                f"{source_path}"
                            )
                        output_file.write(chunk)
                        remaining -= len(chunk)
                _set_external_slice(
                    tensor,
                    temporary_data_path.name,
                    output_offset,
                    source_length,
                    info.get("checksum"),
                )
                output_offset += source_length

        onnx.save_model(graph, str(temporary_model_path))
        onnx.checker.check_model(str(temporary_model_path))
        _set_external_location(graph, final_data_path.name)
        onnx.save_model(graph, str(temporary_model_path))
        os.replace(temporary_data_path, final_data_path)
        os.replace(temporary_model_path, model_path)

    onnx.checker.check_model(str(model_path))
    for original_path in original_external_paths:
        if original_path != final_data_path and original_path.is_file():
            original_path.unlink()
    return onnx.load(str(model_path), load_external_data=False)


def simplify_external_onnx(model_path: Path) -> onnx.ModelProto:
    """Simplify an external-data model without serializing it in this process.

    onnxsim's Python API returns an in-memory ModelProto and therefore crosses
    protobuf's 2 GiB serialization boundary for large LINEAE variants.  Its CLI
    handles the large-model path internally, so run that in a child process and
    normalize the result to ``<model>.onnx`` + ``<model>.onnx.data``.

    Shape inference is deliberately disabled here.  It is known to corrupt the
    external-data form of these large fixed-shape decoder graphs.
    """
    model_path = model_path.resolve()
    original_graph = onnx.load(str(model_path), load_external_data=False)
    original_external_paths = external_data_paths(model_path, original_graph)
    if not original_external_paths:
        raise ValueError("simplify_external_onnx requires an external-data model")

    final_data_path = model_path.with_name(model_path.name + ".data")
    with tempfile.TemporaryDirectory(
        dir=model_path.parent, prefix=f".{model_path.stem}-onnxsim-"
    ) as temporary_directory:
        temporary_directory_path = Path(temporary_directory)
        temporary_model_path = temporary_directory_path / model_path.name
        command = [
            sys.executable,
            "-m",
            "onnxsim",
            str(model_path),
            str(temporary_model_path),
            "--save-as-external-data",
            "--skip-shape-inference",
        ]
        completed = subprocess.run(
            command,
            cwd=temporary_directory_path,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part.strip()
            )
            raise RuntimeError(
                f"onnxsim external-data simplification failed: {details}"
            )

        simplified_graph = onnx.load(
            str(temporary_model_path), load_external_data=False
        )
        temporary_external_paths = external_data_paths(
            temporary_model_path, simplified_graph
        )
        if len(temporary_external_paths) != 1:
            raise RuntimeError(
                "onnxsim must produce exactly one external-data file, got "
                f"{temporary_external_paths}"
            )
        temporary_data_path = temporary_external_paths[0]
        if not temporary_data_path.is_file():
            raise RuntimeError(
                f"onnxsim did not write external data: {temporary_data_path}"
            )

        _set_external_location(simplified_graph, final_data_path.name)
        rewritten_model_path = temporary_directory_path / (
            "rewritten-" + model_path.name
        )
        onnx.save_model(simplified_graph, str(rewritten_model_path))
        os.replace(temporary_data_path, final_data_path)
        os.replace(rewritten_model_path, model_path)

    onnx.checker.check_model(str(model_path))
    for original_path in original_external_paths:
        if original_path != final_data_path and original_path.is_file():
            original_path.unlink()
    return onnx.load(str(model_path), load_external_data=False)


def export_and_verify(args) -> dict:
    config = SLConfig.fromfile(args.config)
    validate_image_preprocess_schema(config.image_preprocess_schema)
    num_select_override = getattr(args, "num_select", None)
    num_select = resolve_export_num_select(config.num_queries, num_select_override)
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
    output_path = args.output.resolve()
    with torch.inference_mode():
        reference_logits, reference_lines = wrapper(images)
        # The legacy exporter recognizes only ``str`` as a model file location.
        # Passing a Path leaves that location empty and disables its >2 GiB
        # external-data fallback even though Path works for smaller models.
        torch.onnx.export(
            wrapper,
            (images,),
            str(output_path),
            input_names=["images"],
            output_names=["pred_logits", "pred_lines"],
            opset_version=args.opset,
            do_constant_folding=True,
            external_data=True,
            dynamo=False,
        )
    graph = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(str(output_path))
    exported_external_paths = external_data_paths(output_path, graph)
    simplified = False
    disable_onnxsim = getattr(args, "disable_onnxsim", False)
    if exported_external_paths:
        if disable_onnxsim:
            print(
                "External-data ONNX detected; consolidating without "
                "simplification or shape inference."
            )
            graph = consolidate_external_onnx(output_path)
        else:
            print(
                "External-data ONNX detected; simplifying without shape inference."
            )
            graph = simplify_external_onnx(output_path)
            simplified = True
    elif not disable_onnxsim:
        graph, simplification_succeeded = onnxsim.simplify(graph)
        if not simplification_succeeded:
            raise RuntimeError("onnxsim validation failed")
        onnx.checker.check_model(graph)
        onnx.save(graph, str(output_path))
        simplified = True
    graph = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(str(output_path))
    final_external_paths = external_data_paths(output_path, graph)
    missing_external_paths = [
        path for path in final_external_paths if not path.is_file()
    ]
    if missing_external_paths:
        raise RuntimeError(
            f"ONNX external-data files are missing: {missing_external_paths}"
        )
    nonstandard_domains = sorted(
        {node.domain for node in graph.graph.node if node.domain not in ("", "ai.onnx")}
    )
    if nonstandard_domains:
        raise RuntimeError(
            f"exported graph contains non-standard ONNX domains: {nonstandard_domains}"
        )
    redundant_selection_chains = find_redundant_decoder_selection_chains(graph)
    if redundant_selection_chains:
        raise RuntimeError(
            "exported graph contains legacy decoder stack/permute/gather output "
            f"selection: {redundant_selection_chains}"
        )
    node_counts = Counter(node.op_type for node in graph.graph.node)
    graph_nodes = {
        "total": len(graph.graph.node),
        "Transpose": node_counts["Transpose"],
        "Gather": node_counts["Gather"],
        "Unsqueeze": node_counts["Unsqueeze"],
    }
    result = {
        "format": "lineae_onnx_export_v3",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)) if args.checkpoint else None,
        "onnx": str(output_path),
        "onnx_sha256": sha256_file(output_path),
        "external_data_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in final_external_paths
        ],
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
        "graph_nodes": graph_nodes,
        "image_preprocess_schema": config.image_preprocess_schema,
        "opencv_version": cv2.__version__,
        "num_select_source": (
            "cli" if num_select_override is not None else "variant_num_queries"
        ),
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
            "optionally filter the ONNX outputs to this many top-scoring line "
            "queries; when omitted, all config.num_queries for the selected "
            "variant are exported, and output TopK selection is omitted"
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
    print(f"Graph nodes: {result['graph_nodes']}")
    print(f"ONNX SHA-256: {result['onnx_sha256']}")
    for external_data_file in result["external_data_files"]:
        print(
            "External data: "
            f"{external_data_file['path']} "
            f"({external_data_file['size_bytes']} bytes, "
            f"SHA-256 {external_data_file['sha256']})"
        )
    if args.report is not None:
        print(f"Export report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
