import pytest
import torch

from main import create
from util.slconfig import SLConfig


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available",
            ),
        ),
    ],
)
def test_s_forward_loss_backward_and_update(device):
    config = SLConfig.fromfile("configs/lineae/lineae_s.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    config.dn_number = 4
    model, _ = create(config, "modelname")
    criterion = create(config, "criterionname")
    model.to(device).train()
    criterion.to(device).train()

    images = torch.randn(1, 3, 64, 64, device=device)
    targets = [{
        "labels": torch.tensor([0], device=device),
        "lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]], device=device),
    }]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    tracked = model.backbone.pyramid.p3[0].weight
    before = tracked.detach().clone()
    encoder_line_head = model.decoder.enc_out_bbox_embed.layers[-1].weight
    assert torch.equal(encoder_line_head[0], encoder_line_head[2])
    assert torch.equal(encoder_line_head[1], encoder_line_head[3])
    frozen = model.backbone.core.blocks[0].attn.qkv.weight
    frozen_before = frozen.detach().clone()

    outputs = model(images, targets)
    assert len(model.decoder._dynamic_anchor_cache) == 1
    cached_anchors = next(iter(model.decoder._dynamic_anchor_cache.values()))
    assert len(model.encoder._position_embedding_cache) == 1
    cached_position = next(iter(model.encoder._position_embedding_cache.values()))
    losses = criterion(outputs, targets)
    total = sum(losses.values())
    assert torch.isfinite(total)
    assert outputs["pred_logits"].shape == (1, 20, 2)
    assert outputs["pred_lines"].shape == (1, 20, 4)
    assert model.backbone.last_feature_shapes == [
        (1, 192, 8, 8),
        (1, 192, 4, 4),
        (1, 192, 2, 2),
    ]

    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    assert not torch.equal(before, tracked.detach())
    assert not torch.equal(encoder_line_head[0], encoder_line_head[2])
    assert not torch.equal(encoder_line_head[1], encoder_line_head[3])
    assert torch.equal(frozen_before, frozen.detach())

    empty_targets = [{
        "labels": torch.empty(0, dtype=torch.long, device=device),
        "lines": torch.empty(0, 4, device=device),
    }]
    empty_outputs = model(images, empty_targets)
    predicted_lengths = (
        empty_outputs["pred_lines"][..., :2]
        - empty_outputs["pred_lines"][..., 2:]
    ).square().sum(dim=-1).sqrt()
    assert torch.count_nonzero(predicted_lengths) > 0
    reloaded_anchors = next(iter(model.decoder._dynamic_anchor_cache.values()))
    assert reloaded_anchors[0] is cached_anchors[0]
    assert reloaded_anchors[1] is cached_anchors[1]
    assert next(iter(model.encoder._position_embedding_cache.values())) is cached_position
    empty_loss = sum(criterion(empty_outputs, empty_targets).values())
    assert torch.isfinite(empty_loss)
