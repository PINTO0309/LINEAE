import copy

import pytest
import torch

from models.lineae.hybrid_encoder import HybridEncoderAsymConv, TransformerEncoderLayer


def _encoder():
    return HybridEncoderAsymConv(
        in_channels=[32, 32, 32],
        feat_strides=[8, 16, 32],
        hidden_dim=32,
        nhead=4,
        dim_feedforward=64,
        use_encoder_idx=[2],
        depth_mult=0.34,
        eval_spatial_size=(64, 64),
    )


def _materialized_attention_forward(layer, src, pos_embed):
    residual = src
    if layer.normalize_before:
        src = layer.norm1(src)
    query = key = layer.with_pos_embed(src, pos_embed)
    src = layer.self_attn(
        query,
        key,
        value=src,
        need_weights=True,
    )[0]
    src = residual + layer.dropout1(src)
    if not layer.normalize_before:
        src = layer.norm1(src)
    residual = src
    if layer.normalize_before:
        src = layer.norm2(src)
    src = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
    src = residual + layer.dropout2(src)
    if not layer.normalize_before:
        src = layer.norm2(src)
    return src


@pytest.mark.parametrize("normalize_before", [False, True])
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
def test_encoder_sdpa_matches_materialized_attention_outputs_and_gradients(
    device,
    normalize_before,
):
    torch.manual_seed(53)
    optimized = TransformerEncoderLayer(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        activation="gelu",
        normalize_before=normalize_before,
    ).to(device)
    reference = copy.deepcopy(optimized)
    optimized.train()
    reference.train()
    value = torch.randn(2, 17, 32, device=device, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    position = torch.randn(1, 17, 32, device=device)
    output_gradient = torch.randn_like(value)

    actual = optimized(value, pos_embed=position)
    expected = _materialized_attention_forward(reference, reference_value, position)
    (actual * output_gradient).sum().backward()
    (expected * output_gradient).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(value.grad, reference_value.grad, rtol=2e-5, atol=3e-6)
    for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
        optimized.named_parameters(),
        reference.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=2e-5,
            atol=3e-6,
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
def test_position_embedding_cache_is_exact_bounded_and_nonpersistent(device):
    encoder = _encoder().to(device)
    encoder._position_embedding_cache_limit = 2
    state_keys = tuple(encoder.state_dict())
    model_device = next(encoder.parameters()).device
    assert encoder.pos_embed2.device == model_device
    assert not any("pos_embed" in key for key in state_keys)

    encoder.train()
    expected = encoder.create_sinehw_position_embedding(3, 3, 16, device=device)
    first = encoder._position_embedding_for(2, 3, 3, device)
    second = encoder._position_embedding_for(2, 3, 3, device)

    assert torch.equal(first, expected)
    assert second is first
    assert tuple(encoder.state_dict()) == state_keys

    encoder._position_embedding_for(2, 4, 4, device)
    encoder._position_embedding_for(2, 5, 5, device)
    assert len(encoder._position_embedding_cache) == 2
    assert all(key[1:3] != (3, 3) for key in encoder._position_embedding_cache)

    reloaded = encoder._position_embedding_for(2, 3, 3, device)
    assert reloaded is not first
    assert reloaded.data_ptr() != first.data_ptr()

    encoder.eval()
    fixed = encoder._position_embedding_for(2, 2, 2, device)
    assert fixed is encoder.pos_embed2
    assert fixed.device == model_device

    encoder.to(device)
    assert encoder._position_embedding_cache == {}
    assert encoder.pos_embed2.device == model_device
    assert tuple(encoder.state_dict()) == state_keys
