from pathlib import Path

import onnx
import onnxruntime as ort
import onnxsim
import pytest
import torch

from main import create
from tools.deployment_parity import compare_line_sets
from tools.export_onnx import ExportWrapper
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


def test_deployment_num_select_accepts_cli_override_and_validates_query_count():
    assert resolve_num_select(300, 1100) == 300
    assert resolve_num_select(300, 1100, 500) == 500
    with pytest.raises(ValueError, match="num_select must be in"):
        resolve_num_select(300, 1100, 0)
    with pytest.raises(ValueError, match="num_select must be in"):
        resolve_num_select(300, 1100, 1101)


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
