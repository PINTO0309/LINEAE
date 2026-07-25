from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import pytest
import torch

from main import create
from tools.deployment_parity import compare_line_sets
from tools.export_onnx import (
    ExportWrapper,
    consolidate_external_onnx,
    external_data_paths,
    find_redundant_decoder_selection_chains,
    resolve_export_num_select,
    simplify_external_onnx,
)
from util.deployment import resolve_num_select
from util.onnx_runtime import create_ort_session
from util.slconfig import SLConfig


class _ExportStub(torch.nn.Module):
    def __init__(self, num_queries: int):
        super().__init__()
        self.num_queries = num_queries

    def deploy(self):
        return self

    def forward(self, images):
        values = images[:, 0, 0, : self.num_queries]
        logits = torch.stack((values, -values), dim=-1)
        lines = torch.stack((values, values + 1, values + 2, values + 3), dim=-1)
        return {"pred_logits": logits, "pred_lines": lines}


def test_export_tool_does_not_run_onnx_runtime_parity():
    source = Path("tools/export_onnx.py").read_text(encoding="utf-8")
    assert "import onnxruntime" not in source
    assert "compare_line_sets" not in source
    assert "create_ort_session" not in source
    assert ".parity.json" not in source
    assert "--cuda-ort" not in source
    assert 'args.output.with_suffix(".export.json")' not in source
    assert "if args.report is not None:" in source
    assert "str(output_path)" in source
    assert "external_data=True" in source
    assert '"--skip-shape-inference"' in source


def test_external_data_is_simplified_without_shape_inference(tmp_path):
    input_info = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 64]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 64]
    )
    weight_values = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    weight = onnx.numpy_helper.from_array(weight_values, name="weight")
    bias_values = np.linspace(-0.5, 0.5, 64, dtype=np.float32)[None]
    constant_name = "_model_encoder_encoder.0_layers.0_Constant_attr__value"
    bias = onnx.numpy_helper.from_array(bias_values, name=constant_name)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "Constant", [], ["bias"], name="encoder_constant", value=bias
            ),
            onnx.helper.make_node("MatMul", ["input", "weight"], ["projected"]),
            onnx.helper.make_node("Add", ["projected", "bias"], ["output"]),
        ],
        "external-data-test",
        [input_info],
        [output_info],
        [weight],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    model_path = tmp_path / "model.onnx"
    original_weight_path = tmp_path / weight.name
    original_constant_path = tmp_path / constant_name
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=0,
        convert_attribute=True,
    )

    metadata = onnx.load(str(model_path), load_external_data=False)
    assert external_data_paths(model_path, metadata) == sorted(
        [original_constant_path, original_weight_path]
    )

    simplified = simplify_external_onnx(model_path)
    final_data_path = tmp_path / "model.onnx.data"

    onnx.checker.check_model(str(model_path))
    assert external_data_paths(model_path, simplified) == [final_data_path]
    assert final_data_path.is_file()
    assert final_data_path.stat().st_size > 0
    assert not original_constant_path.exists()
    assert not original_weight_path.exists()
    input_values = np.linspace(-1.0, 1.0, 64, dtype=np.float32)[None]
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"input": input_values})[0]
    np.testing.assert_allclose(
        actual, input_values @ weight_values + bias_values, rtol=1e-6
    )


def test_external_data_is_consolidated_without_simplifying_graph(tmp_path):
    input_info = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 4]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 4]
    )
    weight_values = np.arange(16, dtype=np.float32).reshape(4, 4)
    bias_values = np.linspace(-0.5, 0.5, 4, dtype=np.float32)[None]
    weight = onnx.numpy_helper.from_array(weight_values, name="weight")
    constant_name = "_model_encoder_Constant_attr__value"
    bias = onnx.numpy_helper.from_array(bias_values, name=constant_name)
    nodes = [
        onnx.helper.make_node(
            "Constant", [], ["bias"], name="constant", value=bias
        ),
        onnx.helper.make_node("MatMul", ["input", "weight"], ["projected"]),
        onnx.helper.make_node("Add", ["projected", "bias"], ["output"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "external-data-consolidation-test",
        [input_info],
        [output_info],
        [weight],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    model_path = tmp_path / "model.onnx"
    original_paths = [tmp_path / constant_name, tmp_path / weight.name]
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=0,
        convert_attribute=True,
    )
    original_metadata = onnx.load(str(model_path), load_external_data=False)
    original_nodes = [
        (node.name, node.op_type, tuple(node.input), tuple(node.output))
        for node in original_metadata.graph.node
    ]

    consolidated = consolidate_external_onnx(model_path)
    final_data_path = tmp_path / "model.onnx.data"

    onnx.checker.check_model(str(model_path))
    assert external_data_paths(model_path, consolidated) == [final_data_path]
    assert final_data_path.is_file()
    assert all(not path.exists() for path in original_paths)
    assert [
        (node.name, node.op_type, tuple(node.input), tuple(node.output))
        for node in consolidated.graph.node
    ] == original_nodes
    input_values = np.linspace(-1.0, 1.0, 4, dtype=np.float32)[None]
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"input": input_values})[0]
    np.testing.assert_allclose(
        actual, input_values @ weight_values + bias_values, rtol=1e-6
    )


def test_legacy_decoder_stack_permute_gather_chain_is_detected():
    nodes = [
        onnx.helper.make_node(
            "Unsqueeze",
            ["decoder_output", "axes"],
            ["stacked"],
            name="/model/decoder/decoder/Unsqueeze_35",
        ),
        onnx.helper.make_node(
            "Transpose",
            ["stacked"],
            ["permuted"],
            name="/model/decoder/decoder/Transpose_1",
        ),
        onnx.helper.make_node(
            "Gather",
            ["permuted", "index"],
            ["selected"],
            name="/model/decoder/Gather_10",
        ),
    ]
    graph = onnx.helper.make_graph(nodes, "legacy-selection", [], [])
    model = onnx.helper.make_model(graph)

    assert find_redundant_decoder_selection_chains(model) == [
        "/model/decoder/Gather_10"
    ]


def test_deployment_num_select_accepts_cli_override_and_validates_query_count():
    assert resolve_num_select(300, 1100) == 300
    assert resolve_num_select(300, 1100, 500) == 500
    with pytest.raises(ValueError, match="num_select must be in"):
        resolve_num_select(300, 1100, 0)
    with pytest.raises(ValueError, match="num_select must be in"):
        resolve_num_select(300, 1100, 1101)


def test_onnx_export_defaults_to_variant_queries_and_cli_override_filters():
    assert resolve_export_num_select(600) == 600
    assert resolve_export_num_select(600, 300) == 300
    with pytest.raises(ValueError, match="num_select must be in"):
        resolve_export_num_select(600, 601)


@pytest.mark.parametrize(
    ("num_select", "expected_output_topk"),
    [(4, True), (8, False)],
)
@pytest.mark.filterwarnings(
    "ignore:You are using the legacy TorchScript-based ONNX export:DeprecationWarning"
)
def test_export_wrapper_omits_output_topk_when_all_queries_are_selected(
    num_select,
    expected_output_topk,
    tmp_path,
):
    num_queries = 8
    wrapper = ExportWrapper(
        _ExportStub(num_queries),
        num_select,
        num_queries,
    ).eval()
    images = torch.randn(1, 3, 1, num_queries)
    output_path = tmp_path / f"select_{num_select}.onnx"

    with torch.inference_mode():
        logits, lines = wrapper(images)
        torch.onnx.export(
            wrapper,
            (images,),
            output_path,
            input_names=["images"],
            output_names=["pred_logits", "pred_lines"],
            opset_version=17,
            dynamo=False,
        )

    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    output_topk_nodes = [node for node in graph.graph.node if node.op_type == "TopK"]
    assert wrapper.uses_output_topk is expected_output_topk
    assert bool(output_topk_nodes) is expected_output_topk
    if expected_output_topk:
        producers = {
            output: node for node in graph.graph.node for output in node.output
        }
        consumers = {}
        for node in graph.graph.node:
            for input_name in node.input:
                consumers.setdefault(input_name, []).append(node)
        output_topk = output_topk_nodes[0]
        assert producers[output_topk.input[0]].op_type == "Slice"
        assert {
            node.op_type for node in consumers[output_topk.output[1]]
        } == {"Expand"}
    assert logits.shape == (1, num_select, 2)
    assert lines.shape == (1, num_select, 4)


@pytest.mark.parametrize("variant", ["A", "T", "S", "X"])
@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.filterwarnings(
    "ignore:You are using the legacy TorchScript-based ONNX export:DeprecationWarning"
)
@pytest.mark.filterwarnings("ignore:The feature will be removed:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:Constant folding.*:UserWarning")
def test_representative_full_models_simplify_and_match_pinned_onnx_runtime(
    variant,
    tmp_path,
):
    torch.manual_seed(97)
    config = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    model, _ = create(config, "modelname")
    wrapper = ExportWrapper(model.eval(), config.num_select, config.num_queries).eval()
    images = torch.randn(1, 3, 64, 64)
    output_path = tmp_path / f"lineae_{variant.lower()}.onnx"

    with torch.inference_mode():
        expected_logits, expected_lines = wrapper(images)
        torch.onnx.export(
            wrapper,
            (images,),
            output_path,
            input_names=["images"],
            output_names=["pred_logits", "pred_lines"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    simplified_graph, simplification_succeeded = onnxsim.simplify(graph)
    assert simplification_succeeded
    onnx.checker.check_model(simplified_graph)
    onnx.save(simplified_graph, output_path)
    session, _, requested, provider_options = create_ort_session(
        ort,
        output_path,
        require_cuda=False,
    )
    actual_logits, actual_lines = session.run(None, {"images": images.numpy()})
    parity = compare_line_sets(
        expected_logits.detach().cpu().numpy(),
        expected_lines.detach().cpu().numpy(),
        actual_logits,
        actual_lines,
        atol=1e-4,
        rtol=1e-3,
        max_outlier_fraction=0.005,
    )

    assert ort.__version__ == "1.26.0"
    assert onnxsim.__version__.removeprefix("v") == "0.6.5"
    assert requested == ["CPUExecutionProvider"]
    assert provider_options == {}
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert expected_logits.shape == actual_logits.shape == (1, 10, 2)
    assert expected_lines.shape == actual_lines.shape == (1, 10, 4)
    assert parity["parity"] == {"pred_logits": True, "pred_lines": True}
    assert output_path.stat().st_size > 0

    if variant == "A":
        counts = Counter(node.op_type for node in simplified_graph.graph.node)
        domains = {node.domain or "ai.onnx" for node in simplified_graph.graph.node}
        inferred = onnx.shape_inference.infer_shapes(simplified_graph)
        decoder_shapes = []
        for value in inferred.graph.value_info:
            if not value.name.startswith("/model/decoder/"):
                continue
            decoder_shapes.append(tuple(
                dimension.dim_value or dimension.dim_param
                for dimension in value.type.tensor_type.shape.dim
            ))

        assert len(simplified_graph.graph.node) < 724
        assert counts["Transpose"] <= 24
        assert counts["Gather"] <= 2
        assert counts["Unsqueeze"] <= 8
        assert domains == {"ai.onnx"}
        assert not any(shape[:2] == (config.num_queries, 1) for shape in decoder_shapes)
        assert find_redundant_decoder_selection_chains(simplified_graph) == []
        node_names = {node.name for node in simplified_graph.graph.node}
        assert "/Gather" not in node_names
        assert "/Unsqueeze" not in node_names
        assert "/model/decoder/Gather_7" not in node_names
        assert "/model/decoder/Unsqueeze_1" not in node_names
        producers = {
            output: node
            for node in simplified_graph.graph.node
            for output in node.output
        }
        consumers = {}
        for node in simplified_graph.graph.node:
            for input_name in node.input:
                consumers.setdefault(input_name, []).append(node)
        proposal_topk = next(
            node
            for node in simplified_graph.graph.node
            if node.name == "/model/decoder/TopK"
        )
        assert producers[proposal_topk.input[0]].op_type == "Slice"
        assert {
            node.op_type for node in consumers[proposal_topk.output[1]]
        } == {"Expand"}
