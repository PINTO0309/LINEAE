import onnx
import onnxruntime as ort
import onnxsim
import pytest
import torch

from main import create
from tools.deployment_parity import compare_line_sets
from tools.export_onnx import ExportWrapper
from util.onnx_runtime import create_ort_session
from util.slconfig import SLConfig


@pytest.mark.parametrize("variant", ["A", "S", "X"])
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
    wrapper = ExportWrapper(model.eval(), config.num_select).eval()
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
