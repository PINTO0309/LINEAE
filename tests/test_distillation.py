import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from engine import train_one_epoch
from main import create
from models.lineae.distillation import (
    DistillationTeacher,
    LineSetDistillationCriterion,
    TeacherTargetCache,
    _batched_linear_sum_assignment,
    build_distillation_criterion,
    endpoint_swap,
    resolve_distillation_temperature_steps,
)
from models.lineae.lineae import LINEAE
from util.slconfig import SLConfig


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_batched_hungarian_matches_individual_solver_with_one_cpu_transfer(
    device,
    monkeypatch,
):
    torch.manual_seed(43)
    costs = [
        torch.randn(rows, columns, device=device)
        for rows, columns in ((5, 3), (4, 4), (7, 2))
    ]
    expected = [
        linear_sum_assignment(cost.detach().cpu().numpy())
        for cost in costs
    ]
    original_cpu = torch.Tensor.cpu
    transfers = []

    def counted_cpu(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    actual = _batched_linear_sum_assignment(costs)

    assert transfers == [(sum(cost.numel() for cost in costs),)]
    for (actual_rows, actual_columns), (expected_rows, expected_columns) in zip(
        actual,
        expected,
        strict=True,
    ):
        assert (actual_rows == expected_rows).all()
        assert (actual_columns == expected_columns).all()


def test_match_batches_nonempty_costs_once_and_preserves_empty_batch_items(monkeypatch):
    torch.manual_seed(47)
    student = {
        "pred_logits": torch.randn(3, 5, 2),
        "pred_lines": torch.rand(3, 5, 4),
    }
    teacher = {
        "pred_logits": torch.tensor(10.0).expand(3, 4, 2).clone(),
        "pred_lines": torch.rand(3, 4, 4),
    }
    teacher["pred_logits"][1].fill_(-10.0)
    criterion = _criterion(confidence_threshold=0.9, top_k=3)
    original_cpu = torch.Tensor.cpu
    transfers = []

    def counted_cpu(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    matches = criterion.match(student, teacher)

    assert transfers == [(2 * 5 * 3,)]
    assert [student_index.numel() for student_index, _ in matches] == [3, 0, 3]
    assert [teacher_index.numel() for _, teacher_index in matches] == [3, 0, 3]


def _outputs(logits, lines, *, requires_grad=False):
    return {
        "pred_logits": torch.tensor(logits, dtype=torch.float32, requires_grad=requires_grad),
        "pred_lines": torch.tensor(lines, dtype=torch.float32, requires_grad=requires_grad),
    }


def _criterion(**overrides):
    options = dict(
        confidence_threshold=0.0,
        top_k=10,
        match_cost_class=2.0,
        match_cost_line=5.0,
        class_weight=1.0,
        line_weight=5.0,
        total_weight=1.0,
        warmup_steps=0,
        temperature_start=1.0,
        temperature_end=1.0,
        temperature_steps=0,
    )
    options.update(overrides)
    return LineSetDistillationCriterion(**options)


def test_teacher_filter_uses_the_line_evaluator_class_zero_score():
    logits = torch.tensor([
        [-10.0, 20.0],
        [1.0, 100.0],
        [2.0, -100.0],
        [0.5, 200.0],
    ])
    criterion = _criterion(confidence_threshold=0.6, top_k=2)

    selected = criterion._teacher_indices(logits)

    # Channel 1 cannot promote the first/fourth proposals; selection and ranking
    # match LineEvaluator/PostProcess's sigmoid(pred_logits[..., 0]).
    assert selected.tolist() == [2, 1]


def test_output_kd_matching_and_logits_ignore_the_unused_second_class_channel():
    student_logits = torch.tensor(
        [[[-2.0, -20.0], [2.0, 20.0]]],
        requires_grad=True,
    )
    teacher_logits = torch.tensor([[[-1.0, -20.0], [1.0, 20.0]]])
    lines = torch.tensor([[[0.1, 0.2, 0.7, 0.8], [0.2, 0.3, 0.6, 0.9]]])
    student = {"pred_logits": student_logits, "pred_lines": lines.clone()}
    teacher = {"pred_logits": teacher_logits, "pred_lines": lines.clone()}
    criterion = _criterion(match_cost_class=1.0, match_cost_line=0.0)

    reference_matches = criterion.match(student, teacher)
    reference = criterion(student, teacher, global_step=1)
    altered_student_logits = student_logits.detach().clone()
    altered_student_logits[..., 1].mul_(-1)
    altered_student_logits.requires_grad_(True)
    altered_teacher_logits = teacher_logits.clone()
    altered_teacher_logits[..., 1].mul_(-1)
    altered_student = {
        "pred_logits": altered_student_logits,
        "pred_lines": lines.clone(),
    }
    altered_teacher = {
        "pred_logits": altered_teacher_logits,
        "pred_lines": lines.clone(),
    }
    altered_matches = criterion.match(altered_student, altered_teacher)
    altered = criterion(altered_student, altered_teacher, global_step=1)

    for (reference_student, reference_teacher), (actual_student, actual_teacher) in zip(
        reference_matches,
        altered_matches,
        strict=True,
    ):
        assert torch.equal(actual_student, reference_student)
        assert torch.equal(actual_teacher, reference_teacher)
    for name in reference:
        assert torch.equal(altered[name], reference[name])
    altered["loss_kd_logits"].backward()
    assert altered_student_logits.grad[..., 0].abs().sum() > 0
    assert torch.count_nonzero(altered_student_logits.grad[..., 1]) == 0


def _example(requires_grad=True):
    logits = [[[4.0, -2.0], [1.5, 2.5], [-3.0, 0.5]]]
    lines = [[
        [0.1, 0.2, 0.7, 0.8],
        [0.2, 0.7, 0.9, 0.1],
        [0.4, 0.4, 0.6, 0.6],
    ]]
    return _outputs(logits, lines, requires_grad=requires_grad)


def test_identical_predictions_have_zero_kd_and_query_permutations_are_invariant():
    teacher = _example(requires_grad=False)
    student = _example(requires_grad=True)
    criterion = _criterion()
    original = criterion(student, teacher, global_step=1)
    assert criterion.last_match_count == 3

    permutation = torch.tensor([2, 0, 1])
    permuted = {
        "pred_logits": student["pred_logits"][:, permutation],
        "pred_lines": student["pred_lines"][:, permutation],
    }
    reordered = criterion(permuted, teacher, global_step=1)

    for name in ("loss_kd_logits", "loss_kd_line"):
        assert original[name].item() < 1e-7
        assert torch.allclose(original[name], reordered[name], atol=1e-7)


def test_nonzero_kd_is_invariant_to_student_query_permutation():
    teacher = _example(requires_grad=False)
    student = _example(requires_grad=True)
    student["pred_logits"] = student["pred_logits"] + torch.tensor([[[0.2], [-0.1], [0.3]]])
    student["pred_lines"] = student["pred_lines"] + torch.tensor([[[0.01], [0.02], [-0.01]]])
    criterion = _criterion()
    original = criterion(student, teacher, global_step=1)
    permutation = torch.tensor([1, 2, 0])
    permuted = criterion(
        {
            "pred_logits": student["pred_logits"][:, permutation],
            "pred_lines": student["pred_lines"][:, permutation],
        },
        teacher,
        global_step=1,
    )
    assert sum(original.values()).item() > 0
    for name in original:
        assert torch.allclose(original[name], permuted[name], atol=1e-7)


def test_swapping_teacher_endpoints_does_not_change_line_loss():
    teacher = _example(requires_grad=False)
    student = _example(requires_grad=True)
    student["pred_lines"] = student["pred_lines"] + 0.03
    criterion = _criterion(match_cost_class=0.0, match_cost_line=1.0)
    direct = criterion(student, teacher, global_step=1)
    swapped_teacher = dict(teacher)
    swapped_teacher["pred_lines"] = endpoint_swap(teacher["pred_lines"])
    swapped = criterion(student, swapped_teacher, global_step=1)
    assert torch.allclose(direct["loss_kd_line"], swapped["loss_kd_line"], atol=1e-7)


def test_empty_teacher_selection_returns_finite_graph_connected_zeros():
    student = _example(requires_grad=True)
    teacher = _outputs(
        [[[-20.0, -20.0], [-30.0, -30.0]]],
        [[[0.0, 0.0, 0.1, 0.1], [0.3, 0.3, 0.4, 0.4]]],
    )
    losses = _criterion(confidence_threshold=0.9)(student, teacher, global_step=1)
    total = sum(losses.values())
    assert torch.isfinite(total)
    assert total.item() == 0.0
    total.backward()
    assert student["pred_logits"].grad is not None


def test_schedule_ramps_weight_and_cosine_anneals_temperature():
    criterion = _criterion(
        total_weight=2.0,
        warmup_steps=10,
        temperature_start=4.0,
        temperature_end=1.0,
        temperature_steps=20,
    )
    assert criterion.schedule(0).weight == 0.0
    assert criterion.schedule(5).weight == 1.0
    assert criterion.schedule(10).weight == 2.0
    assert criterion.schedule(0).temperature == 4.0
    assert criterion.schedule(20).temperature == 1.0


def test_gazelle_temperature_schedule_resolves_to_the_last_optimizer_step():
    resolved = resolve_distillation_temperature_steps(
        -1,
        optimizer_steps_per_epoch=5,
        epochs=4,
    )
    assert resolved == 19
    criterion = _criterion(
        temperature_start=1.0,
        temperature_end=4.0,
        temperature_steps=-1,
    )
    with pytest.raises(RuntimeError, match="were not resolved"):
        criterion.schedule(0)
    criterion.set_temperature_steps(resolved)
    assert criterion.schedule(0).temperature == 1.0
    assert criterion.schedule(19).temperature == 4.0
    assert criterion.schedule(100).temperature == 4.0

    assert resolve_distillation_temperature_steps(
        12,
        optimizer_steps_per_epoch=5,
        epochs=4,
    ) == 12
    with pytest.raises(ValueError, match="must be -1"):
        resolve_distillation_temperature_steps(
            -2,
            optimizer_steps_per_epoch=5,
            epochs=4,
        )


def test_optional_feature_kd_is_normalized_switchable_and_differentiable():
    student = _example(requires_grad=True)
    teacher = _example(requires_grad=False)
    student_features = [
        torch.randn(1, 5, size, size, requires_grad=True) for size in (8, 4, 2)
    ]
    teacher_features = [feature.detach().clone() for feature in student_features]
    student["distill_features"] = student_features
    teacher["distill_features"] = teacher_features
    criterion = _criterion(feature_weight=2.0, feature_loss="cosine")
    identical = criterion(student, teacher, global_step=1)
    assert identical["loss_kd_feature"].abs().item() < 1e-6

    teacher["distill_features"] = [feature.roll(1, dims=1) for feature in teacher_features]
    losses = criterion(student, teacher, global_step=1)
    assert losses["loss_kd_feature"].item() > 0
    sum(losses.values()).backward()
    assert all(feature.grad is not None for feature in student_features)

    disabled = _criterion(feature_weight=0.0)(
        _example(requires_grad=True), _example(requires_grad=False), global_step=1
    )
    assert set(disabled) == {"loss_kd_logits", "loss_kd_line"}


def test_feature_kd_aligns_teacher_spatial_shapes_but_rejects_channel_mismatch():
    student = _example(requires_grad=True)
    teacher = _example(requires_grad=False)
    student["distill_features"] = [
        torch.randn(1, 4, 8, 8, requires_grad=True) for _ in range(3)
    ]
    teacher["distill_features"] = [torch.randn(1, 4, 4, 4)] * 3
    losses = _criterion(feature_weight=1.0)(student, teacher, global_step=1)
    assert torch.isfinite(losses["loss_kd_feature"])
    sum(losses.values()).backward()
    assert all(feature.grad is not None for feature in student["distill_features"])

    teacher["distill_features"] = [torch.randn(1, 5, 4, 4)] * 3
    with pytest.raises(ValueError, match="batch/channel mismatch"):
        _criterion(feature_weight=1.0)(student, teacher, global_step=1)


class _FeatureBackbone(nn.Module):
    out_channels = (2, 3, 4)

    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleList([
            nn.Conv2d(3, 2, 1),
            nn.Conv2d(3, 3, 1, stride=2),
            nn.Conv2d(3, 4, 1, stride=4),
        ])

    def forward(self, samples):
        return [level(samples) for level in self.levels]


class _IdentityEncoder(nn.Module):
    def forward(self, features):
        return features


class _FeatureDecoder(nn.Module):
    def forward(self, features, targets=None):
        batch = features[0].shape[0]
        return {
            "pred_logits": features[0].mean((2, 3))[:, :1, None],
            "pred_lines": features[0].new_zeros(batch, 1, 4),
        }


def test_lineae_owns_learned_student_feature_projections():
    model = LINEAE(
        _FeatureBackbone(),
        _IdentityEncoder(),
        _FeatureDecoder(),
        return_distill_features=True,
        distill_projection_channels=(5, 5, 5),
    )
    output = model(torch.randn(2, 3, 8, 8))
    assert [tuple(feature.shape) for feature in output["distill_features"]] == [
        (2, 5, 8, 8),
        (2, 5, 4, 4),
        (2, 5, 2, 2),
    ]
    sum(feature.square().mean() for feature in output["distill_features"]).backward()
    assert all(
        projection.weight.grad is not None
        for projection in model.distill_feature_projections
    )


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.2))

    def forward(self, samples, targets=None):
        batch = samples.shape[0]
        logits = self.value.expand(batch, 2, 1)
        base_lines = torch.tensor(
            [[0.1, 0.2, 0.7, 0.8], [0.2, 0.3, 0.6, 0.9]],
            device=samples.device,
        )
        lines = (base_lines + self.value).expand(batch, -1, -1)
        return {"pred_logits": logits, "pred_lines": lines}


class _SupervisedCriterion(nn.Module):
    def forward(self, outputs, targets):
        return {"loss_supervised": outputs["pred_logits"].square().mean()}


def test_training_engine_keeps_teacher_eval_and_gradient_free():
    student = _TinyModel()
    teacher = _TinyModel()
    teacher.train()
    teacher.requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    args = SimpleNamespace(
        amp=False,
        gradient_accumulation_steps=1,
        verify_optimizer_step=False,
        output_dir="",
        use_ema=False,
        ema_epoch=0,
    )
    loader = [(torch.ones(1, 3), [{"labels": torch.empty(0)}])]
    stats, steps, epoch_complete = train_one_epoch(
        student,
        _SupervisedCriterion(),
        loader,
        optimizer,
        torch.device("cpu"),
        epoch=0,
        args=args,
        teacher_model=teacher,
        distillation_criterion=_criterion(),
    )
    assert steps == 1
    assert epoch_complete is True
    assert not teacher.training
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert stats["kd_matches"] == 2
    assert stats["kd_temperature"] == 1.0
    assert stats["kd_overhead_ms"] >= 0.0


@pytest.fixture(scope="module")
def untrained_xl_teacher_for_integration():
    config = SLConfig.fromfile("configs/lineae/lineae_xl.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    config.return_distill_features = True
    config.distill_feature_weight = 0.0
    model, _ = create(config, "modelname")
    model.requires_grad_(False).eval()
    return model, config


@pytest.fixture(scope="module")
def untrained_x_teacher_for_integration():
    config = SLConfig.fromfile("configs/lineae/lineae_x.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    model, _ = create(config, "modelname")
    model.requires_grad_(False).eval()
    return model, config


@pytest.mark.parametrize("variant", ["X", "L", "M", "S", "N", "P", "F", "A"])
def test_xl_output_distillation_updates_every_smaller_variant(
    variant,
    untrained_xl_teacher_for_integration,
):
    torch.manual_seed(59)
    teacher_core, teacher_config = untrained_xl_teacher_for_integration
    config = SLConfig.fromfile(f"configs/lineae/distill/lineae_{variant.lower()}.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    config.dn_number = 4
    config.distill_confidence_threshold = 0.0
    config.distill_top_k = 10
    config.distill_warmup_steps = 0
    config.distill_temperature_steps = 0
    student, _ = create(config, "modelname")
    supervised_criterion = create(config, "criterionname")
    distillation_criterion = build_distillation_criterion(config)
    student.train()
    supervised_criterion.train()
    teacher = DistillationTeacher(
        teacher_core,
        source_mean=config.image_mean,
        source_std=config.image_std,
        target_mean=teacher_config.image_mean,
        target_std=teacher_config.image_std,
        target_spatial_size=(64, 64),
    ).eval()

    pixels = torch.rand(1, 3, 64, 64)
    source_mean = torch.tensor(config.image_mean).view(1, 3, 1, 1)
    source_std = torch.tensor(config.image_std).view(1, 3, 1, 1)
    images = (pixels - source_mean) / source_std
    targets = [{
        "labels": torch.tensor([0]),
        "lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]]),
    }]
    tracked = [
        parameter
        for name, parameter in student.named_parameters()
        if "decoder.class_embed" in name
        and name.endswith(".weight")
        and parameter.requires_grad
    ][-1]
    optimizer = torch.optim.SGD([tracked], lr=1e-3)
    before = tracked.detach().clone()

    student_outputs = student(images, targets)
    with torch.no_grad():
        teacher_outputs = teacher(images)
    supervised_losses = supervised_criterion(student_outputs, targets)
    kd_losses = distillation_criterion(
        student_outputs,
        teacher_outputs,
        global_step=0,
    )
    kd_total = sum(kd_losses.values())
    kd_gradient = torch.autograd.grad(kd_total, tracked, retain_graph=True)[0]
    total = sum(supervised_losses.values()) + kd_total
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    assert distillation_criterion.last_match_count == 10
    assert kd_losses["loss_kd_logits"] > 0
    assert kd_losses["loss_kd_line"] > 0
    assert torch.isfinite(kd_gradient).all()
    assert torch.count_nonzero(kd_gradient)
    assert torch.isfinite(total)
    assert not torch.equal(before, tracked.detach())
    assert not teacher_outputs["pred_logits"].requires_grad
    assert not teacher_outputs["pred_lines"].requires_grad
    assert all(parameter.grad is None for parameter in teacher_core.parameters())


@pytest.mark.parametrize("variant", ["L", "M", "S", "N", "P", "F", "A"])
def test_x_cascade_distillation_updates_every_lower_variant(
    variant,
    untrained_x_teacher_for_integration,
):
    torch.manual_seed(61)
    teacher_core, teacher_config = untrained_x_teacher_for_integration
    config = SLConfig.fromfile(f"configs/lineae/cascade/lineae_{variant.lower()}.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    config.dn_number = 4
    config.distill_confidence_threshold = 0.0
    config.distill_top_k = 10
    config.distill_warmup_steps = 0
    config.distill_temperature_steps = 0
    assert config.distill_teacher_config == "configs/lineae/lineae_x.py"
    assert config.distill_teacher_checkpoint == "ckpts/lineae_x_teacher.pth"
    student, _ = create(config, "modelname")
    supervised_criterion = create(config, "criterionname")
    distillation_criterion = build_distillation_criterion(config)
    student.train()
    supervised_criterion.train()
    teacher = DistillationTeacher(
        teacher_core,
        source_mean=config.image_mean,
        source_std=config.image_std,
        target_mean=teacher_config.image_mean,
        target_std=teacher_config.image_std,
        target_spatial_size=(64, 64),
    ).eval()

    pixels = torch.rand(1, 3, 64, 64)
    source_mean = torch.tensor(config.image_mean).view(1, 3, 1, 1)
    source_std = torch.tensor(config.image_std).view(1, 3, 1, 1)
    images = (pixels - source_mean) / source_std
    targets = [{
        "labels": torch.tensor([0]),
        "lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]]),
    }]
    tracked = [
        parameter
        for name, parameter in student.named_parameters()
        if "decoder.class_embed" in name
        and name.endswith(".weight")
        and parameter.requires_grad
    ][-1]
    optimizer = torch.optim.SGD([tracked], lr=1e-3)
    before = tracked.detach().clone()

    student_outputs = student(images, targets)
    with torch.no_grad():
        teacher_outputs = teacher(images)
    supervised_losses = supervised_criterion(student_outputs, targets)
    kd_losses = distillation_criterion(
        student_outputs,
        teacher_outputs,
        global_step=0,
    )
    kd_total = sum(kd_losses.values())
    kd_gradient = torch.autograd.grad(kd_total, tracked, retain_graph=True)[0]
    total = sum(supervised_losses.values()) + kd_total
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    assert distillation_criterion.last_match_count == 10
    assert kd_losses["loss_kd_logits"] > 0
    assert kd_losses["loss_kd_line"] > 0
    assert torch.isfinite(kd_gradient).all()
    assert torch.count_nonzero(kd_gradient)
    assert torch.isfinite(total)
    assert not torch.equal(before, tracked.detach())
    assert not teacher_outputs["pred_logits"].requires_grad
    assert not teacher_outputs["pred_lines"].requires_grad
    assert all(parameter.grad is None for parameter in teacher_core.parameters())


def test_xl_to_x_real_feature_distillation_updates_all_pyramid_projections(
    untrained_xl_teacher_for_integration,
):
    torch.manual_seed(67)
    teacher_core, teacher_config = untrained_xl_teacher_for_integration
    config = SLConfig.fromfile("configs/lineae/ablations/lineae_x_feature_kd.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    config.dn_number = 4
    config.distill_confidence_threshold = 0.0
    config.distill_top_k = 10
    config.distill_warmup_steps = 0
    config.distill_temperature_steps = 0
    student, _ = create(config, "modelname")
    student.train()
    teacher = DistillationTeacher(
        teacher_core,
        source_mean=config.image_mean,
        source_std=config.image_std,
        target_mean=teacher_config.image_mean,
        target_std=teacher_config.image_std,
        target_spatial_size=(64, 64),
    ).eval()
    criterion = build_distillation_criterion(config)

    pixels = torch.rand(1, 3, 64, 64)
    source_mean = torch.tensor(config.image_mean).view(1, 3, 1, 1)
    source_std = torch.tensor(config.image_std).view(1, 3, 1, 1)
    images = (pixels - source_mean) / source_std
    targets = [{
        "labels": torch.tensor([0]),
        "lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]]),
    }]
    projection_weights = [
        projection.weight for projection in student.distill_feature_projections
    ]
    before = [weight.detach().clone() for weight in projection_weights]
    optimizer = torch.optim.SGD(
        student.distill_feature_projections.parameters(),
        lr=1e-3,
    )

    student_outputs = student(images, targets)
    with torch.no_grad():
        teacher_outputs = teacher(images)
    losses = criterion(student_outputs, teacher_outputs, global_step=0)
    gradients = torch.autograd.grad(
        losses["loss_kd_feature"],
        projection_weights,
        retain_graph=True,
    )
    optimizer.zero_grad()
    sum(losses.values()).backward()
    optimizer.step()

    expected_shapes = [(1, 256, 8, 8), (1, 256, 4, 4), (1, 256, 2, 2)]
    assert [tuple(feature.shape) for feature in student_outputs["distill_features"]] == expected_shapes
    assert [tuple(feature.shape) for feature in teacher_outputs["distill_features"]] == expected_shapes
    assert losses["loss_kd_feature"] > 0
    assert losses["loss_kd_logits"] > 0
    assert losses["loss_kd_line"] > 0
    assert criterion.last_match_count == 10
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) for gradient in gradients)
    assert all(
        not torch.equal(previous, weight.detach())
        for previous, weight in zip(before, projection_weights, strict=True)
    )
    assert all(parameter.grad is None for parameter in teacher_core.parameters())


def test_real_xl_teacher_cache_skips_full_hits_and_batches_only_partial_misses(
    tmp_path,
    untrained_xl_teacher_for_integration,
):
    teacher_core, teacher_config = untrained_xl_teacher_for_integration
    student_config = SLConfig.fromfile("configs/lineae/lineae_a.py")
    cache = TeacherTargetCache(
        tmp_path,
        namespace={"checkpoint": "real-xl-integration", "schema": 3},
    )
    teacher = DistillationTeacher(
        teacher_core,
        source_mean=student_config.image_mean,
        source_std=student_config.image_std,
        target_mean=teacher_config.image_mean,
        target_std=teacher_config.image_std,
        target_spatial_size=(64, 64),
        cache=cache,
    ).eval()
    forward_batch_sizes = []

    def capture_batch_size(_module, inputs):
        forward_batch_sizes.append(inputs[0].shape[0])

    handle = teacher_core.register_forward_pre_hook(capture_batch_size)
    try:
        torch.manual_seed(83)
        pixels = torch.rand(2, 3, 32, 32)
        source_mean = torch.tensor(student_config.image_mean).view(1, 3, 1, 1)
        source_std = torch.tensor(student_config.image_std).view(1, 3, 1, 1)
        samples = (pixels - source_mean) / source_std

        first = teacher(samples)
        second = teacher(samples.clone())
        changed = samples.clone()
        changed[0, 0, 0, 0] += 1.0
        partial = teacher(changed)
    finally:
        handle.remove()

    assert forward_batch_sizes == [2, 1]
    assert teacher.cache_stats == {"hits": 3, "misses": 3, "writes": 3}
    for key in ("pred_logits", "pred_lines"):
        assert torch.equal(first[key], second[key])
        assert torch.equal(first[key][1], partial[key][1])
    assert [tuple(feature.shape) for feature in first["distill_features"]] == [
        (2, 256, 8, 8),
        (2, 256, 4, 4),
        (2, 256, 2, 2),
    ]
    for first_feature, second_feature, partial_feature in zip(
        first["distill_features"],
        second["distill_features"],
        partial["distill_features"],
        strict=True,
    ):
        assert torch.equal(first_feature, second_feature)
        assert torch.equal(first_feature[1], partial_feature[1])
    assert all(parameter.grad is None for parameter in teacher_core.parameters())


class _CaptureTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, samples, targets=None):
        self.seen = samples
        return {"pred_logits": samples.mean((2, 3))[:, None], "pred_lines": samples.new_zeros(1, 1, 4)}


def test_teacher_receives_same_pixels_with_its_own_normalization():
    core = _CaptureTeacher()
    teacher = DistillationTeacher(
        core,
        source_mean=[0.5, 0.4, 0.3],
        source_std=[0.2, 0.25, 0.5],
        target_mean=[0.1, 0.2, 0.3],
        target_std=[0.5, 0.4, 0.25],
    )
    raw = torch.tensor([0.7, 0.3, 0.8]).view(1, 3, 1, 1)
    source = (raw - torch.tensor([0.5, 0.4, 0.3]).view(1, 3, 1, 1)) / torch.tensor(
        [0.2, 0.25, 0.5]
    ).view(1, 3, 1, 1)
    teacher(source)
    expected = (raw - torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1)) / torch.tensor(
        [0.5, 0.4, 0.25]
    ).view(1, 3, 1, 1)
    assert torch.allclose(core.seen, expected)


def test_teacher_resizes_the_augmented_tensor_to_its_canonical_input():
    core = _CaptureTeacher()
    teacher = DistillationTeacher(
        core,
        source_mean=[0.0, 0.0, 0.0],
        source_std=[1.0, 1.0, 1.0],
        target_mean=[0.0, 0.0, 0.0],
        target_std=[1.0, 1.0, 1.0],
        target_spatial_size=(8, 8),
    )
    samples = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)

    teacher(samples)

    expected = torch.nn.functional.interpolate(
        samples,
        size=(8, 8),
        mode="bilinear",
        align_corners=False,
    )
    assert core.seen.shape == (1, 3, 8, 8)
    assert torch.equal(core.seen, expected)


def test_teacher_rejects_invalid_canonical_input_size():
    with pytest.raises(ValueError, match="two positive integers"):
        DistillationTeacher(
            _CaptureTeacher(),
            source_mean=[0.0, 0.0, 0.0],
            source_std=[1.0, 1.0, 1.0],
            target_mean=[0.0, 0.0, 0.0],
            target_std=[1.0, 1.0, 1.0],
            target_spatial_size=(640, 0),
        )


class _CacheableTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.batch_sizes = []

    def forward(self, samples, targets=None):
        self.calls += 1
        self.batch_sizes.append(samples.shape[0])
        average = samples.mean((2, 3))
        logits = average[:, :2, None].transpose(1, 2)
        lines = average[:, :1, None].repeat(1, 2, 4)
        return {
            "pred_logits": logits,
            "pred_lines": lines,
            "distill_features": [
                samples,
                torch.nn.functional.avg_pool2d(samples, 2),
                torch.nn.functional.avg_pool2d(samples, 4),
            ],
        }


def _cached_teacher(core, tmp_path):
    cache = TeacherTargetCache(
        tmp_path,
        namespace={"checkpoint": "abc", "normalization": "identity"},
    )
    return DistillationTeacher(
        core,
        source_mean=[0.0, 0.0, 0.0],
        source_std=[1.0, 1.0, 1.0],
        target_mean=[0.0, 0.0, 0.0],
        target_std=[1.0, 1.0, 1.0],
        cache=cache,
    )


def test_teacher_cache_reuses_only_byte_identical_augmented_inputs(tmp_path):
    core = _CacheableTeacher()
    teacher = _cached_teacher(core, tmp_path)
    samples = torch.randn(2, 3, 8, 8)
    first = teacher(samples)
    second = teacher(samples.clone())
    assert core.calls == 1
    assert teacher.cache_stats == {"hits": 2, "misses": 2, "writes": 2}
    for key in ("pred_logits", "pred_lines"):
        assert torch.equal(first[key], second[key])
    for expected, actual in zip(
        first["distill_features"], second["distill_features"], strict=True
    ):
        assert torch.equal(expected, actual)

    changed = samples.clone()
    changed[0, 0, 0, 0] += 1.0
    teacher(changed)
    assert core.calls == 2
    assert core.batch_sizes == [2, 1]

    fresh_core = _CacheableTeacher()
    fresh_teacher = _cached_teacher(fresh_core, tmp_path)
    reloaded = fresh_teacher(samples)
    assert fresh_core.calls == 0
    assert torch.equal(first["pred_logits"], reloaded["pred_logits"])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_teacher_cache_zero_copy_hash_preserves_the_legacy_key(tmp_path, device):
    cache = TeacherTargetCache(tmp_path, namespace={"checkpoint": "hash-test"})
    sample = torch.randn(3, 7, 5, device=device).transpose(1, 2)
    value = sample.detach().contiguous().cpu()
    legacy = hashlib.sha256()
    legacy.update(cache.namespace.encode())
    legacy.update(str(value.dtype).encode())
    legacy.update(json.dumps(list(value.shape)).encode())
    legacy.update(value.view(torch.uint8).numpy().tobytes())

    assert cache.key(sample) == legacy.hexdigest()
    assert cache.key(sample, context="preprocess-a") != cache.key(
        sample,
        context="preprocess-b",
    )


def test_teacher_cache_keys_source_and_skips_preprocessing_on_full_hits(
    tmp_path,
    monkeypatch,
):
    core = _CacheableTeacher()
    cache = TeacherTargetCache(tmp_path, namespace={"checkpoint": "resize-test"})
    teacher = DistillationTeacher(
        core,
        source_mean=[0.0, 0.0, 0.0],
        source_std=[1.0, 1.0, 1.0],
        target_mean=[0.1, 0.2, 0.3],
        target_std=[0.5, 0.5, 0.5],
        target_spatial_size=(16, 16),
        cache=cache,
    )
    samples = torch.randn(2, 3, 8, 8)
    key_shapes = []
    resize_calls = []
    original_key = cache.key
    original_interpolate = torch.nn.functional.interpolate

    def captured_key(sample, **kwargs):
        key_shapes.append(tuple(sample.shape))
        return original_key(sample, **kwargs)

    def captured_interpolate(*args, **kwargs):
        resize_calls.append(tuple(args[0].shape))
        return original_interpolate(*args, **kwargs)

    monkeypatch.setattr(cache, "key", captured_key)
    monkeypatch.setattr(torch.nn.functional, "interpolate", captured_interpolate)
    first = teacher(samples)
    second = teacher(samples.clone())

    assert key_shapes == [(3, 8, 8)] * 4
    assert resize_calls == [(2, 3, 8, 8)]
    assert core.calls == 1
    assert teacher.cache_stats == {"hits": 2, "misses": 2, "writes": 2}
    for key in ("pred_logits", "pred_lines"):
        assert torch.equal(first[key], second[key])


def test_teacher_cache_preprocessing_fingerprint_prevents_cross_transform_hits(tmp_path):
    samples = torch.randn(1, 3, 8, 8)
    namespace = {"checkpoint": "shared", "test": "transform-fingerprint"}
    first_core = _CacheableTeacher()
    first = DistillationTeacher(
        first_core,
        source_mean=[0.0, 0.0, 0.0],
        source_std=[1.0, 1.0, 1.0],
        target_mean=[0.0, 0.0, 0.0],
        target_std=[1.0, 1.0, 1.0],
        cache=TeacherTargetCache(tmp_path, namespace=namespace),
    )
    second_core = _CacheableTeacher()
    second = DistillationTeacher(
        second_core,
        source_mean=[0.0, 0.0, 0.0],
        source_std=[1.0, 1.0, 1.0],
        target_mean=[0.5, 0.5, 0.5],
        target_std=[1.0, 1.0, 1.0],
        cache=TeacherTargetCache(tmp_path, namespace=namespace),
    )

    first(samples)
    second(samples)

    assert first_core.calls == 1
    assert second_core.calls == 1
    assert second.cache_stats == {"hits": 0, "misses": 1, "writes": 1}
