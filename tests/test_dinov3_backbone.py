import copy
from types import MethodType

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import models.lineae.backbones.dinov3 as dinov3_module
from models.lineae.backbones.dinov3 import (
    Attention,
    CompactDinoV3,
    CompactDinoV3Backbone,
    DinoFinalResidualFusion,
    OfficialDinoV3,
    OfficialDinoV3Backbone,
    OfficialAttention,
    RopePositionEmbedding,
    _apply_rope,
    _rotate_half,
)


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
def test_half_width_rope_matches_legacy_full_width_values_and_gradients(device):
    rope = RopePositionEmbedding(embed_dim=24, num_heads=3).to(device).eval()
    sin, cos = rope(height=4, width=5)
    assert sin.shape == cos.shape == (1, 1, 20, 4)

    value = torch.randn(2, 3, 20, 8, device=device, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    full_sin = sin.repeat(1, 1, 1, 2)
    full_cos = cos.repeat(1, 1, 1, 2)
    gradient = torch.randn_like(value)

    actual = _apply_rope(value, sin, cos)
    reference = reference_value * full_cos + _rotate_half(reference_value) * full_sin
    (actual * gradient).sum().backward()
    (reference * gradient).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, reference_value.grad, rtol=0, atol=0)
    assert sin.numel() * 2 == full_sin.numel()


@pytest.mark.parametrize("official", [False, True])
def test_dino_half_width_rope_matches_legacy_full_width_model_output(official):
    torch.manual_seed(37)
    if official:
        optimized = OfficialDinoV3(
            embed_dim=24,
            num_heads=3,
            ffn_ratio=4.0,
            swiglu=False,
            depth=2,
            storage_tokens=2,
        )
        with torch.no_grad():
            optimized.cls_token.zero_()
            optimized.storage_tokens.zero_()
            optimized.mask_token.zero_()
    else:
        optimized = CompactDinoV3(embed_dim=24, num_heads=3, depth=2)
    reference = copy.deepcopy(optimized)
    half_forward = reference.rope_embed.forward

    def full_width_forward(_module, *, height, width):
        sin, cos = half_forward(height=height, width=width)
        return sin.repeat(1, 1, 1, 2), cos.repeat(1, 1, 1, 2)

    reference.rope_embed.forward = MethodType(full_width_forward, reference.rope_embed)
    optimized.eval()
    reference.eval()
    images = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        actual = optimized(images)
        expected = reference(images)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rope_eval_cache_reuses_values_and_invalidates_on_buffer_change():
    rope = RopePositionEmbedding(embed_dim=24, num_heads=3, rescale_coords=2.0).eval()
    state_keys = tuple(rope.state_dict())

    first_sin, first_cos = rope(height=4, width=5)
    second_sin, second_cos = rope(height=4, width=5)

    assert second_sin.data_ptr() == first_sin.data_ptr()
    assert second_cos.data_ptr() == first_cos.data_ptr()
    assert tuple(rope.state_dict()) == state_keys

    with torch.no_grad():
        rope.periods.mul_(1.01)
    changed_sin, changed_cos = rope(height=4, width=5)

    assert changed_sin.data_ptr() != first_sin.data_ptr()
    assert changed_cos.data_ptr() != first_cos.data_ptr()
    assert not torch.equal(changed_sin, first_sin)
    assert not torch.equal(changed_cos, first_cos)


def test_rope_cache_supports_inference_created_buffers_and_load_invalidation():
    with torch.inference_mode():
        rope = RopePositionEmbedding(embed_dim=24, num_heads=3).eval()
        first_sin, first_cos = rope(height=4, width=5)
        second_sin, second_cos = rope(height=4, width=5)
        state = {key: value.clone() for key, value in rope.state_dict().items()}
        state["periods"].mul_(1.01)
        rope.load_state_dict(state, strict=True)
        changed_sin, changed_cos = rope(height=4, width=5)

    assert rope.periods.is_inference()
    assert second_sin.data_ptr() == first_sin.data_ptr()
    assert second_cos.data_ptr() == first_cos.data_ptr()
    assert changed_sin.data_ptr() != first_sin.data_ptr()
    assert changed_cos.data_ptr() != first_cos.data_ptr()
    assert not torch.equal(changed_sin, first_sin)
    assert not torch.equal(changed_cos, first_cos)


def test_rope_cache_is_bypassed_during_training_and_onnx_export(monkeypatch):
    rope = RopePositionEmbedding(embed_dim=24, num_heads=3, rescale_coords=2.0)
    rope.train()
    torch.manual_seed(5)
    first_sin, first_cos = rope(height=4, width=5)
    second_sin, second_cos = rope(height=4, width=5)

    assert first_sin.data_ptr() != second_sin.data_ptr()
    assert first_cos.data_ptr() != second_cos.data_ptr()
    assert not torch.equal(first_sin, second_sin)
    assert not torch.equal(first_cos, second_cos)
    assert rope._eval_cache is None

    rope.eval()
    cached_sin, cached_cos = rope(height=4, width=5)
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    export_sin, export_cos = rope(height=4, width=5)

    assert export_sin.data_ptr() != cached_sin.data_ptr()
    assert export_cos.data_ptr() != cached_cos.data_ptr()
    torch.testing.assert_close(export_sin, cached_sin)
    torch.testing.assert_close(export_cos, cached_cos)


def test_compact_attention_sdpa_matches_materialized_attention_and_gradients():
    torch.manual_seed(4)
    attention = Attention(dim=24, num_heads=3)
    value = torch.randn(2, 5, 24, requires_grad=True)
    sin = torch.zeros(1, 1, 4, 8)
    cos = torch.ones(1, 1, 4, 8)

    actual = attention(value, sin, cos)
    actual.square().sum().backward()
    actual_input_grad = value.grad.detach().clone()
    actual_parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in attention.named_parameters()
    }

    attention.zero_grad(set_to_none=True)
    reference_value = value.detach().clone().requires_grad_(True)
    batch, tokens, channels = reference_value.shape
    qkv = attention.qkv(reference_value).reshape(
        batch, tokens, 3, attention.num_heads, channels // attention.num_heads
    ).permute(2, 0, 3, 1, 4)
    query, key, val = qkv.unbind(0)
    scale = (channels // attention.num_heads) ** -0.5
    weights = (query @ key.transpose(-2, -1) * scale).softmax(dim=-1)
    reference = (weights @ val).transpose(1, 2).reshape(batch, tokens, channels)
    reference = attention.proj_drop(attention.proj(reference))
    reference.square().sum().backward()

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_input_grad, reference_value.grad, rtol=1e-5, atol=1e-6)
    for name, parameter in attention.named_parameters():
        torch.testing.assert_close(
            actual_parameter_grads[name],
            parameter.grad,
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("official", [False, True])
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
def test_no_grad_dino_attention_reuses_qk_storage_with_bit_exact_outputs(
    device,
    official,
    monkeypatch,
):
    torch.manual_seed(53)
    attention = (OfficialAttention if official else Attention)(dim=24, num_heads=3).to(device)
    tokens = 7 if official else 5
    value = torch.randn(2, tokens, 24, device=device)
    rope = RopePositionEmbedding(embed_dim=24, num_heads=3).to(device).eval()
    sin, cos = rope(height=2, width=2)
    original_in_place = dinov3_module._apply_rope_in_place
    reused_pointers = []

    def counted_in_place(tensor, rope_sin, rope_cos):
        reused_pointers.append(tensor.data_ptr())
        return original_in_place(tensor, rope_sin, rope_cos)

    monkeypatch.setattr(dinov3_module, "_apply_rope_in_place", counted_in_place)
    with torch.inference_mode():
        actual = attention(value, sin, cos)
    with torch.no_grad():
        no_grad_actual = attention(value, sin, cos)
    assert len(reused_pointers) == 4

    monkeypatch.setattr(dinov3_module, "_can_reuse_qk_storage", lambda: False)
    with torch.inference_mode():
        expected = attention(value, sin, cos)

    assert torch.equal(actual, expected)
    assert torch.equal(no_grad_actual, expected)


@pytest.mark.parametrize("official", [False, True])
def test_dino_checkpoint_without_rng_preservation_matches_gradients(official):
    torch.manual_seed(31)
    if official:
        reference = OfficialDinoV3(
            embed_dim=24,
            num_heads=3,
            ffn_ratio=4.0,
            swiglu=False,
            depth=2,
            storage_tokens=2,
            use_checkpoint=False,
        )
        with torch.no_grad():
            reference.cls_token.zero_()
            reference.storage_tokens.zero_()
            reference.mask_token.zero_()
        checkpointed = OfficialDinoV3(
            embed_dim=24,
            num_heads=3,
            ffn_ratio=4.0,
            swiglu=False,
            depth=2,
            storage_tokens=2,
            use_checkpoint=True,
        )
    else:
        reference = CompactDinoV3(
            embed_dim=24,
            num_heads=3,
            depth=2,
            use_checkpoint=False,
        )
        checkpointed = CompactDinoV3(
            embed_dim=24,
            num_heads=3,
            depth=2,
            use_checkpoint=True,
        )
    checkpointed.load_state_dict(reference.state_dict(), strict=True)
    reference.train()
    checkpointed.train()
    reference_input = torch.randn(1, 3, 32, 32, requires_grad=True)
    checkpointed_input = reference_input.detach().clone().requires_grad_(True)

    torch.manual_seed(77)
    reference_output = reference(reference_input)
    reference_output.square().mean().backward()
    torch.manual_seed(77)
    checkpointed_output = checkpointed(checkpointed_input)
    checkpointed_output.square().mean().backward()

    torch.testing.assert_close(checkpointed_output, reference_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        checkpointed_input.grad,
        reference_input.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    for (reference_name, reference_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(),
        checkpointed.named_parameters(),
        strict=True,
    ):
        assert actual_name == reference_name
        torch.testing.assert_close(
            actual_parameter.grad,
            reference_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
        )


def test_s_checkpoint_load_and_feature_contract():
    backbone = CompactDinoV3Backbone(
        embed_dim=192,
        num_heads=3,
        weights_path="ckpts/vitt_distill.pt",
        trainable_depth=2,
    )
    assert backbone.checkpoint_report is not None
    assert backbone.checkpoint_report.strict
    assert backbone.checkpoint_report.tensor_count == 148
    assert backbone.out_channels == (192, 192, 192)
    assert backbone.out_strides == (8, 16, 32)
    assert backbone.num_blocks == 12

    images = torch.randn(1, 3, 64, 64)
    features = backbone(images)
    assert [feature.shape for feature in features] == [
        (1, 192, 8, 8),
        (1, 192, 4, 4),
        (1, 192, 2, 2),
    ]

    sum(feature.square().mean() for feature in features).backward()
    assert all(
        parameter.grad is not None
        for parameter in backbone.pyramid.parameters()
        if parameter.requires_grad
    )
    trainable_blocks = [
        any(parameter.requires_grad for parameter in block.parameters())
        for block in backbone.core.blocks
    ]
    assert trainable_blocks == [False] * 10 + [True, True]


def test_dino_input_must_be_divisible_by_32():
    backbone = CompactDinoV3Backbone(
        embed_dim=192,
        num_heads=3,
        weights_path=None,
        trainable_depth=2,
    )
    try:
        backbone(torch.randn(1, 3, 48, 48))
    except ValueError as error:
        assert "divisible by 32" in str(error)
    else:
        raise AssertionError("invalid DINO input shape was accepted")


@pytest.mark.parametrize(
    "embed_dim,num_heads,ffn_ratio,swiglu",
    [
        (384, 6, 4.0, False),
        (384, 6, 6.0, True),
        (768, 12, 4.0, False),
    ],
)
def test_official_dinov3_random_initialization_is_finite(
    embed_dim,
    num_heads,
    ffn_ratio,
    swiglu,
):
    torch.manual_seed(43)
    core = OfficialDinoV3(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        swiglu=swiglu,
        depth=2,
    )

    assert all(torch.isfinite(parameter).all() for parameter in core.parameters())
    assert torch.count_nonzero(core.cls_token)
    assert torch.count_nonzero(core.storage_tokens)
    assert torch.count_nonzero(core.mask_token) == 0

    output = core(torch.randn(1, 3, 32, 32))
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in core.parameters()
    )


def test_intermediate_block_fusion_is_an_independent_trainable_adapter():
    backbone = CompactDinoV3Backbone(
        embed_dim=192,
        num_heads=3,
        weights_path=None,
        pyramid_channels=64,
        trainable_depth=2,
        intermediate_layers=(3, 7, 11),
    )
    features = backbone(torch.randn(1, 3, 64, 64))
    assert [tuple(feature.shape) for feature in features] == [
        (1, 64, 8, 8),
        (1, 64, 4, 4),
        (1, 64, 2, 2),
    ]
    sum(feature.square().mean() for feature in features).backward()
    assert backbone.intermediate_fusion.level_weights.grad is not None
    assert backbone.intermediate_fusion.projection.weight.grad is not None


def test_final_residual_fusion_is_identity_initialized_and_trains_gates():
    torch.manual_seed(101)
    fusion = DinoFinalResidualFusion(channels=8, levels=4)
    features = [
        torch.randn(2, 8, 3, 3, requires_grad=True)
        for _ in range(4)
    ]

    output = fusion(features)
    torch.testing.assert_close(output, features[-1], rtol=0, atol=0)
    output.square().mean().backward()

    assert torch.isfinite(fusion.residual_gates.grad).all()
    assert torch.count_nonzero(fusion.residual_gates.grad)
    assert torch.isfinite(fusion.projection.weight.grad).all()


def test_official_backbone_selects_final_residual_fusion_schema():
    backbone = OfficialDinoV3Backbone(
        embed_dim=32,
        num_heads=4,
        ffn_ratio=2.0,
        swiglu=False,
        depth=4,
        weights_path=None,
        pyramid_channels=16,
        intermediate_layers=(0, 1, 2, 3),
        intermediate_fusion_schema="residual_final_v1",
    )
    assert isinstance(backbone.intermediate_fusion, DinoFinalResidualFusion)
    features = backbone(torch.randn(2, 3, 32, 32))
    assert [tuple(feature.shape) for feature in features] == [
        (2, 16, 4, 4),
        (2, 16, 2, 2),
        (2, 16, 1, 1),
    ]


def test_intermediate_layers_must_be_sorted_unique_and_have_multiple_levels():
    with pytest.raises(ValueError, match="sorted and unique"):
        CompactDinoV3Backbone(
            embed_dim=192,
            num_heads=3,
            weights_path=None,
            intermediate_layers=(7, 3, 11),
        )
    with pytest.raises(ValueError, match="at least two"):
        CompactDinoV3Backbone(
            embed_dim=192,
            num_heads=3,
            weights_path=None,
            intermediate_layers=(11,),
        )


@pytest.mark.parametrize(
    "filename,embed_dim,num_heads,ffn_ratio,swiglu,tensor_count",
    [
        ("dinov3_vits16_pretrain_lvd1689m-08c60483.pth", 384, 6, 4.0, False, 188),
        ("dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth", 384, 6, 6.0, True, 212),
        ("dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", 768, 12, 4.0, False, 188),
    ],
)
def test_official_dinov3_checkpoints_load_strictly(
    filename, embed_dim, num_heads, ffn_ratio, swiglu, tensor_count
):
    backbone = OfficialDinoV3Backbone(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        swiglu=swiglu,
        weights_path=f"ckpts/{filename}",
        pyramid_channels=192,
        trainable_depth=2,
    )
    assert backbone.checkpoint_report is not None
    assert backbone.checkpoint_report.strict
    assert backbone.checkpoint_report.tensor_count == tensor_count
    with torch.inference_mode():
        features = backbone(torch.randn(1, 3, 64, 64))
    assert [tuple(feature.shape) for feature in features] == [
        (1, 192, 8, 8),
        (1, 192, 4, 4),
        (1, 192, 2, 2),
    ]


@pytest.mark.parametrize(
    "filename,embed_dim,num_heads,ffn_ratio,swiglu,depth,tensor_count",
    [
        (
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            1024,
            16,
            4.0,
            False,
            24,
            368,
        ),
        (
            "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
            1280,
            20,
            6.0,
            True,
            32,
            552,
        ),
    ],
)
def test_large_official_dinov3_checkpoint_shapes_match_meta_architecture(
    filename,
    embed_dim,
    num_heads,
    ffn_ratio,
    swiglu,
    depth,
    tensor_count,
):
    with FakeTensorMode():
        state = torch.load(
            f"ckpts/{filename}",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    with torch.device("meta"):
        core = OfficialDinoV3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            swiglu=swiglu,
            depth=depth,
        )

    expected_shapes = {
        key: tuple(value.shape) for key, value in core.state_dict().items()
    }
    actual_shapes = {key: tuple(value.shape) for key, value in state.items()}
    assert len(state) == tensor_count
    assert actual_shapes == expected_shapes
    assert core.rope_embed.head_dim == 64
    assert len(core.blocks) == depth
    if swiglu:
        assert core.blocks[0].mlp.w1.out_features == int(embed_dim * ffn_ratio * 2 / 3)
    else:
        assert core.blocks[0].mlp.fc1.out_features == int(embed_dim * ffn_ratio)
