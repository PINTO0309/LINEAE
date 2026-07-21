import torch

from main import create
from models.lineae.criterion import LINEACriterion
from models.lineae.linea_utils import endpoint_swap
from models.lineae.matcher import HungarianMatcher
from util.slconfig import SLConfig


def test_supervised_hungarian_matching_prefers_the_same_undirected_line():
    target_line = torch.tensor([0.0, 0.0, 1.0, 1.0])
    outputs = {
        "pred_logits": torch.zeros(1, 1, 2),
        "pred_lines": endpoint_swap(target_line).view(1, 1, 4),
    }
    targets = [{
        "labels": torch.tensor([0, 0]),
        "lines": torch.stack((target_line, torch.tensor([0.9, 0.9, 0.1, 0.1]))),
    }]

    indices = HungarianMatcher(
        cost_class=0.0,
        cost_bbox=1.0,
        endpoint_invariant_lines=True,
    )(outputs, targets)

    assert indices[0][0].tolist() == [0]
    assert indices[0][1].tolist() == [0]


def test_supervised_line_loss_is_endpoint_swap_invariant_and_differentiable():
    criterion = LINEACriterion(
        num_classes=2,
        matcher=None,
        weight_dict={"loss_line": 1.0},
        focal_alpha=0.1,
        losses=["lines"],
        endpoint_invariant_lines=True,
    )
    prediction = torch.tensor(
        [[[0.8, 0.9, 0.1, 0.2]]],
        requires_grad=True,
    )
    target_line = torch.tensor([[0.1, 0.2, 0.8, 0.9]])
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    direct = criterion.loss_lines(
        {"pred_lines": prediction},
        [{"lines": target_line}],
        indices,
        num_boxes=1,
    )["loss_line"]
    swapped = criterion.loss_lines(
        {"pred_lines": prediction},
        [{"lines": endpoint_swap(target_line)}],
        indices,
        num_boxes=1,
    )["loss_line"]

    assert torch.equal(direct, swapped)
    assert direct.item() == 0.0
    direct.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_endpoint_invariant_loss_breaks_zero_length_anchor_symmetry():
    criterion = LINEACriterion(
        num_classes=2,
        matcher=None,
        weight_dict={"loss_line": 1.0},
        focal_alpha=0.1,
        losses=["lines"],
        endpoint_invariant_lines=True,
    )
    prediction = torch.full((1, 1, 4), 0.5, requires_grad=True)
    targets = [{"lines": torch.tensor([[0.2, 0.2, 0.8, 0.8]])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss = criterion.loss_lines(
        {"pred_lines": prediction}, targets, indices, num_boxes=1
    )["loss_line"]
    loss.backward()

    assert torch.equal(
        prediction.grad,
        torch.tensor([[[1.0, 1.0, -1.0, -1.0]]]),
    )


def test_linea_control_retains_the_original_directed_endpoint_semantics():
    criterion = LINEACriterion(
        num_classes=2,
        matcher=None,
        weight_dict={"loss_line": 1.0},
        focal_alpha=0.1,
        losses=["lines"],
        endpoint_invariant_lines=False,
    )
    target_line = torch.tensor([[0.1, 0.2, 0.8, 0.9]])
    prediction = endpoint_swap(target_line).view(1, 1, 4)
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss = criterion.loss_lines(
        {"pred_lines": prediction},
        [{"lines": target_line}],
        indices,
        num_boxes=1,
    )["loss_line"]

    assert loss.item() > 0.0


def test_lineae_enables_undirected_lines_without_changing_the_linea_control():
    lineae = SLConfig.fromfile("configs/lineae/lineae_s.py")
    linea = SLConfig.fromfile("configs/linea/linea_hgnetv2_n.py")
    lineae_criterion = create(lineae, "criterionname")
    linea_criterion = create(linea, "criterionname")

    assert lineae.endpoint_invariant_lines is True
    assert lineae_criterion.endpoint_invariant_lines is True
    assert lineae_criterion.matcher.endpoint_invariant_lines is True
    assert linea.endpoint_invariant_lines is False
    assert linea_criterion.endpoint_invariant_lines is False
    assert linea_criterion.matcher.endpoint_invariant_lines is False
