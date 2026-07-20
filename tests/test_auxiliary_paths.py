import pytest
import torch

from main import create
from models.lineae.attention_mechanism import MSDeformAttn
from models.lineae.criterion import DFINESetCriterion
from util.slconfig import SLConfig


class _Matcher(torch.nn.Module):
    def forward(self, outputs, targets):
        return [
            (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
            for _ in targets
        ]


def test_deformable_attention_non_power_of_two_head_width_warns():
    with pytest.warns(UserWarning, match="power of 2"):
        MSDeformAttn(d_model=120, n_levels=3, n_heads=8, n_points=4)


def test_dfine_auxiliary_line_map_loss_path_is_finite():
    criterion = DFINESetCriterion(
        num_classes=2,
        matcher=_Matcher(),
        weight_dict={"loss_lmap": 1.0},
        focal_alpha=0.1,
        reg_max=16,
        losses=[],
    )
    outputs = {
        "pred_logits": torch.zeros(1, 2, 2),
        "pred_lines": torch.zeros(1, 2, 4),
        "aux_lmap": [
            torch.zeros(1, 1, 8, 8),
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 1, 2, 2),
        ],
        "dn_meta": None,
    }
    targets = [
        {
            "labels": torch.empty(0, dtype=torch.long),
            "lines": torch.empty(0, 4),
            "lmap": [
                torch.zeros(1, 8, 8),
                torch.zeros(1, 4, 4),
                torch.zeros(1, 2, 2),
            ],
        }
    ]

    losses = criterion(outputs, targets)

    assert set(losses) == {"loss_lmap"}
    assert torch.isfinite(losses["loss_lmap"])


def test_fused_deploy_model_preserves_lineae_outputs():
    torch.manual_seed(8)
    config = SLConfig.fromfile("configs/lineae/lineae_s.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    config.distill_feature_weight = 1.0
    model, _ = create(config, "modelname")
    model.eval()
    images = torch.randn(1, 3, 64, 64)
    projection_parameters = sum(
        parameter.numel()
        for parameter in model.distill_feature_projections.parameters()
    )
    parameters_before_deploy = sum(parameter.numel() for parameter in model.parameters())

    with torch.inference_mode():
        reference = model(images)
        assert "distill_features" in reference
        model.deploy()
        deployed = model(images)

    assert "distill_features" not in deployed
    assert model.distill_feature_projections is None
    assert sum(parameter.numel() for parameter in model.parameters()) <= (
        parameters_before_deploy - projection_parameters
    )
    for output in ("pred_logits", "pred_lines"):
        torch.testing.assert_close(
            deployed[output],
            reference[output],
            rtol=1e-4,
            atol=1e-5,
        )


def test_real_x_intermediate_fusion_trains_and_survives_deploy():
    torch.manual_seed(71)
    config = SLConfig.fromfile("configs/lineae/ablations/lineae_x_intermediate.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    config.dn_number = 4
    model, _ = create(config, "modelname")
    criterion = create(config, "criterionname")
    model.train()
    criterion.train()
    fusion = model.backbone.intermediate_fusion
    assert model.backbone.intermediate_layers == (3, 7, 11)
    assert fusion is not None
    optimizer = torch.optim.SGD(fusion.parameters(), lr=1e-3)
    level_weights_before = fusion.level_weights.detach().clone()
    projection_before = fusion.projection.weight.detach().clone()
    images = torch.randn(1, 3, 64, 64)
    targets = [{
        "labels": torch.tensor([0]),
        "lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]]),
    }]

    outputs = model(images, targets)
    total = sum(criterion(outputs, targets).values())
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    assert torch.isfinite(total)
    assert torch.isfinite(fusion.level_weights.grad).all()
    assert torch.count_nonzero(fusion.level_weights.grad)
    assert torch.isfinite(fusion.projection.weight.grad).all()
    assert torch.count_nonzero(fusion.projection.weight.grad)
    assert not torch.equal(level_weights_before, fusion.level_weights.detach())
    assert not torch.equal(projection_before, fusion.projection.weight.detach())
    assert model.backbone.last_feature_shapes == [
        (1, 256, 8, 8),
        (1, 256, 4, 4),
        (1, 256, 2, 2),
    ]

    model.eval()
    with torch.inference_mode():
        reference = model(images)
        model.deploy()
        deployed = model(images)

    assert model.backbone.intermediate_fusion is fusion
    for output in ("pred_logits", "pred_lines"):
        torch.testing.assert_close(
            deployed[output],
            reference[output],
            rtol=1e-4,
            atol=1e-5,
        )
