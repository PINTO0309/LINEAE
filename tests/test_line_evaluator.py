import pytest
import torch

from datasets.line_eval import DualLineEvaluator, LineEvaluator
from models.lineae.linea_utils import select_top_line_predictions


def test_perfect_prediction_reports_all_spatial_ap_metrics():
    evaluator = LineEvaluator()
    predictions = {
        "pred_logits": torch.tensor([[[10.0, -10.0]]]),
        "pred_lines": torch.tensor([[[0.1, 0.2, 0.8, 0.9]]]),
    }
    targets = [{"lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]])}]
    evaluator.update(predictions, targets)
    evaluator.accumulate()
    assert evaluator.sap_results == {"sap5": 100.0, "sap10": 100.0, "sap15": 100.0}


def test_empty_evaluator_and_empty_ground_truth_are_safe():
    evaluator = LineEvaluator()
    evaluator.accumulate()
    assert evaluator.sap_results == {"sap5": 0.0, "sap10": 0.0, "sap15": 0.0}

    evaluator.update(
        {
            "pred_logits": torch.zeros(1, 2, 1),
            "pred_lines": torch.zeros(1, 2, 4),
        },
        [{"lines": torch.empty(0, 4)}],
    )
    evaluator.accumulate()
    assert evaluator.sap_results == {"sap5": 0.0, "sap10": 0.0, "sap15": 0.0}


def test_num_select_uses_only_evaluator_class_zero_and_preserves_line_pairs():
    logits = torch.tensor([[[1.0, 100.0], [4.0, -5.0], [3.0, 200.0], [2.0, 0.0]]])
    lines = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                           [1.0, 1.0, 1.0, 1.0],
                           [2.0, 2.0, 2.0, 2.0],
                           [3.0, 3.0, 3.0, 3.0]]])

    selected_logits, selected_lines = select_top_line_predictions(logits, lines, 2)
    assert torch.equal(selected_logits, logits[:, [1, 2]])
    assert torch.equal(selected_lines, lines[:, [1, 2]])

    evaluator = LineEvaluator(max_predictions=2)
    prepared_lines, prepared_scores = evaluator.prepare(lines, logits)
    assert prepared_lines.shape == (1, 2, 2, 2)
    assert prepared_scores.shape == (1, 2)
    assert torch.equal(prepared_lines[:, :, 0, 0], torch.tensor([[128.0, 256.0]]))


def test_dual_evaluator_separates_official_and_deployment_sap(capsys):
    evaluator = DualLineEvaluator(deploy_max_predictions=2)
    predictions = {
        "pred_logits": torch.tensor([[[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]),
        "pred_lines": torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                                      [1.0, 1.0, 1.0, 1.0],
                                      [0.1, 0.2, 0.8, 0.9]]]),
    }
    targets = [{"lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]])}]

    evaluator.update(predictions, targets)
    evaluator.accumulate()
    evaluator.summarize()

    assert evaluator.sap_results == pytest.approx({
        "sap5": 100 / 3,
        "sap10": 100 / 3,
        "sap15": 100 / 3,
        "official_sap5": 100 / 3,
        "official_sap10": 100 / 3,
        "official_sap15": 100 / 3,
        "deploy_sap5": 0.0,
        "deploy_sap10": 0.0,
        "deploy_sap15": 0.0,
    })
    output = capsys.readouterr().out
    assert "Official sAP (all 3 queries):" in output
    assert "Deployment sAP (top 2 queries):" in output
    assert "deploy_sap10:\t0.0" in output
