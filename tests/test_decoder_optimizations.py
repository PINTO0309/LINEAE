import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from main import create
from models.lineae.attention_mechanism import (
    ms_deform_attn_core_batch_first,
    ms_deform_attn_core_pytorchv2,
    packed_batch_first_self_attention,
)
from models.lineae.decoder import (
    DeformableTransformerDecoderLayer,
    Integral,
    LINEATransformer,
    TransformerDecoder,
    _distance2bbox_batch_first,
    _topk_line_proposals,
)
from models.lineae.hybrid_encoder import TransformerEncoderLayer
from models.lineae.linea_utils import distance2bbox
from util.slconfig import SLConfig


def _out_of_place_line_attention_core(
    value,
    spatial_shapes,
    sampling_locations,
    attention_weights,
    num_points_list,
):
    _, channels, _ = value[0].shape
    batch, queries, heads, _, _ = sampling_locations.shape
    sampling_grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    grid_levels = sampling_grids.split(num_points_list, dim=-2)
    sampled = []
    for level, (height, width) in enumerate(spatial_shapes):
        feature = value[level].unflatten(2, (height, width))
        sampled.append(
            F.grid_sample(
                feature,
                grid_levels[level],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads,
        1,
        queries,
        sum(num_points_list),
    )
    output = (torch.cat(sampled, dim=-1) * weights).sum(-1)
    return output.view(batch, heads * channels, queries).transpose(1, 2).contiguous()


def test_encoder_topk_uses_the_evaluator_line_class_only():
    logits = torch.tensor([[
        [0.1, 100.0],
        [3.0, -100.0],
        [2.0, 200.0],
        [1.0, 300.0],
    ]])

    selected = _topk_line_proposals(logits, topk=2)

    assert selected.shape == (1, 2, 1)
    assert selected[..., 0].tolist() == [[1, 2]]


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        pytest.param(
            "cuda",
            torch.float16,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_no_grad_line_attention_inplace_weighting_is_bit_exact_and_keeps_gradients(
    device,
    dtype,
):
    torch.manual_seed(71)
    batch, queries, heads, channels = 2, 5, 2, 4
    shapes = [(4, 4), (2, 2), (1, 1)]
    points = [2, 1, 1]
    inference_values = [
        torch.randn(batch * heads, channels, height * width, device=device, dtype=dtype)
        for height, width in shapes
    ]
    inference_locations = torch.rand(
        batch,
        queries,
        heads,
        sum(points),
        2,
        device=device,
        dtype=dtype,
    )
    inference_weights = torch.rand(
        batch,
        queries,
        heads,
        sum(points),
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        expected = _out_of_place_line_attention_core(
            inference_values,
            shapes,
            inference_locations,
            inference_weights,
            points,
        )
        actual = ms_deform_attn_core_pytorchv2(
            inference_values,
            shapes,
            inference_locations,
            inference_weights,
            points,
        )
    assert torch.equal(actual, expected)

    actual_values = [value.detach().clone().requires_grad_(True) for value in inference_values]
    expected_values = [value.detach().clone().requires_grad_(True) for value in inference_values]
    actual_locations = inference_locations.detach().clone().requires_grad_(True)
    expected_locations = inference_locations.detach().clone().requires_grad_(True)
    actual_weights = inference_weights.detach().clone().requires_grad_(True)
    expected_weights = inference_weights.detach().clone().requires_grad_(True)
    output_gradient = torch.randn_like(actual)
    actual = ms_deform_attn_core_pytorchv2(
        actual_values,
        shapes,
        actual_locations,
        actual_weights,
        points,
    )
    expected = _out_of_place_line_attention_core(
        expected_values,
        shapes,
        expected_locations,
        expected_weights,
        points,
    )
    (actual * output_gradient).sum().backward()
    (expected * output_gradient).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_value, expected_value in zip(actual_values, expected_values, strict=True):
        torch.testing.assert_close(actual_value.grad, expected_value.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        actual_locations.grad,
        expected_locations.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(actual_weights.grad, expected_weights.grad, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 3])
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
def test_batch_first_line_attention_keeps_batch_axis_and_matches_legacy(batch, device):
    torch.manual_seed(73)
    queries, heads, channels = 5, 2, 4
    shapes = [(4, 4), (2, 2), (1, 1)]
    points = [2, 1, 1]
    explicit_values = [
        torch.randn(batch, heads, channels, height * width, device=device)
        for height, width in shapes
    ]
    flattened_values = [value.flatten(0, 1).clone() for value in explicit_values]
    locations = torch.rand(batch, queries, heads, sum(points), 2, device=device)
    weights = torch.rand(batch, queries, heads, sum(points), device=device)

    with torch.inference_mode():
        expected = ms_deform_attn_core_pytorchv2(
            flattened_values,
            shapes,
            locations,
            weights,
            points,
        )
        actual = ms_deform_attn_core_batch_first(
            explicit_values,
            shapes,
            locations,
            weights,
            points,
        )

    assert actual.shape == (batch, queries, heads * channels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 3])
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
def test_packed_batch_first_attention_matches_multihead_attention(batch, device):
    torch.manual_seed(79)
    attention = nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True).to(device)
    attention.eval()
    value = torch.randn(batch, 11, 32, device=device)
    position = torch.randn_like(value)
    query = value + position

    with torch.inference_mode():
        expected = attention(query, query, value, need_weights=False)[0]
        actual = packed_batch_first_self_attention(attention, query, value)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


class _ZeroCrossAttention(nn.Module):
    def forward(self, query, reference_points, memory, spatial_shapes):
        return torch.zeros_like(query)


def _materialized_decoder_layer_forward(layer, tgt, position, reference_points, mask):
    query = key = layer.with_pos_embed(tgt, position)
    update = layer.self_attn(
        query,
        key,
        tgt,
        attn_mask=mask,
        need_weights=True,
    )[0]
    tgt = layer.norm2(tgt + layer.dropout2(update))
    update = layer.cross_attn(
        layer.with_pos_embed(tgt, position).transpose(0, 1),
        reference_points.transpose(0, 1),
        None,
        None,
    ).transpose(0, 1)
    tgt = layer.norm1(tgt + layer.dropout1(update))
    update = layer.linear2(layer.dropout3(layer.activation(layer.linear1(tgt))))
    return layer.norm3(tgt + layer.dropout4(update))


@pytest.mark.parametrize("use_mask", [False, True])
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
def test_decoder_sdpa_matches_materialized_self_attention_and_gradients(device, use_mask):
    torch.manual_seed(61)
    optimized = DeformableTransformerDecoderLayer(
        d_model=32,
        d_ffn=64,
        dropout=0.0,
        activation="relu",
        n_levels=3,
        n_heads=4,
        n_points=[2, 1, 1],
    ).to(device)
    optimized.cross_attn = _ZeroCrossAttention()
    reference = copy.deepcopy(optimized)
    value = torch.randn(13, 2, 32, device=device, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    position = torch.randn(13, 2, 32, device=device)
    reference_points = torch.rand(13, 2, 4, device=device)
    mask = (
        torch.triu(torch.ones(13, 13, dtype=torch.bool, device=device), diagonal=1)
        if use_mask
        else None
    )
    output_gradient = torch.randn_like(value)

    actual = optimized(
        value,
        tgt_query_pos=position,
        tgt_reference_points=reference_points,
        self_attn_mask=mask,
    )
    expected = _materialized_decoder_layer_forward(
        reference,
        reference_value,
        position,
        reference_points,
        mask,
    )
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


def _legacy_sine_embedding(tensor, hidden_dim):
    hidden_dim_ = hidden_dim // 2
    dim_t = torch.arange(hidden_dim_, dtype=torch.float32, device=tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / hidden_dim_)
    embeddings = []
    for coordinate in (1, 0, 2, 3):
        value = tensor[:, :, coordinate, None] * (2 * math.pi) / dim_t
        embeddings.append(
            torch.stack((value[:, :, 0::2].sin(), value[:, :, 1::2].cos()), dim=3)
            .flatten(2)
        )
    return torch.cat(embeddings, dim=2)


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
def test_decoder_sine_frequency_cache_is_device_exact_and_not_checkpoint_state(device):
    decoder = TransformerDecoder(nn.Identity(), 0, d_model=32).to(device)
    reference_points = torch.rand(7, 2, 4, device=device)
    state_keys = tuple(decoder.state_dict())

    expected = _legacy_sine_embedding(reference_points, decoder.d_model)
    first = decoder.sine_embedding(reference_points, decoder.d_model)
    cached_frequency = decoder._sine_dim_t_cache[1]
    second = decoder.sine_embedding(reference_points, decoder.d_model)

    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    assert decoder._sine_dim_t_cache[1].data_ptr() == cached_frequency.data_ptr()
    assert tuple(decoder.state_dict()) == state_keys
    decoder.to(device)
    assert decoder._sine_dim_t_cache is None


@pytest.mark.parametrize("batch", [1, 3])
def test_deploy_coordinate_math_is_batch_first_and_matches_legacy(batch):
    torch.manual_seed(83)
    queries = 7
    decoder = TransformerDecoder(nn.Identity(), 0, d_model=32)
    reference_points = torch.rand(batch, queries, 4)
    expected_sine = _legacy_sine_embedding(reference_points, decoder.d_model)
    decoder._deploy_batch_first = True
    actual_sine = decoder.sine_embedding(reference_points, decoder.d_model)

    integral = Integral(reg_max=16)
    corners = torch.randn(batch, queries, 4 * 17)
    project = torch.randn(17)
    expected_distance = integral(corners, project)
    actual_distance = integral.forward_batch_first(corners, project)
    scale = torch.tensor([4.0])
    expected_lines = distance2bbox(reference_points, expected_distance, scale)
    actual_lines = _distance2bbox_batch_first(
        reference_points,
        actual_distance,
        scale,
    )

    torch.testing.assert_close(actual_sine, expected_sine, rtol=0, atol=0)
    torch.testing.assert_close(actual_distance, expected_distance, rtol=0, atol=0)
    torch.testing.assert_close(actual_lines, expected_lines, rtol=0, atol=0)


def test_deploy_layout_conversion_preserves_checkpoint_schema():
    config = SLConfig.fromfile("configs/lineae/lineae_a.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    model, _ = create(config, "modelname")
    checkpoint = model.state_dict()
    state_keys = tuple(checkpoint)

    optimized_types = (
        DeformableTransformerDecoderLayer,
        LINEATransformer,
        TransformerDecoder,
        TransformerEncoderLayer,
    )
    for module in model.modules():
        if isinstance(module, optimized_types):
            module.convert_to_deploy()

    assert tuple(model.state_dict()) == state_keys
    fresh_model, _ = create(config, "modelname")
    incompatible = fresh_model.load_state_dict(model.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_expanded_decoder_broadcasts_match_repeated_values_and_gradients():
    torch.manual_seed(23)
    batch, positions, queries, channels = 3, 19, 7, 11
    topk = torch.randint(positions, (batch, queries))

    repeated_memory = torch.randn(batch, positions, channels, requires_grad=True)
    expanded_memory = repeated_memory.detach().clone().requires_grad_(True)
    repeated_proposals = torch.randn(1, positions, 4, requires_grad=True)
    expanded_proposals = repeated_proposals.detach().clone().requires_grad_(True)
    repeated_queries = torch.randn(queries, 1, channels, requires_grad=True)
    expanded_queries = repeated_queries.detach().clone().requires_grad_(True)

    repeated_index = topk.unsqueeze(-1).repeat(1, 1, channels)
    expanded_index = topk.unsqueeze(-1).expand(-1, -1, channels)
    repeated = (
        torch.gather(repeated_memory, 1, repeated_index),
        torch.gather(
            repeated_proposals.repeat(batch, 1, 1),
            1,
            topk.unsqueeze(-1).repeat(1, 1, 4),
        ),
        repeated_queries.repeat(1, batch, 1),
    )
    expanded = (
        torch.gather(expanded_memory, 1, expanded_index),
        torch.gather(
            expanded_proposals.expand(batch, -1, -1),
            1,
            topk.unsqueeze(-1).expand(-1, -1, 4),
        ),
        expanded_queries.expand(-1, batch, -1),
    )

    for actual, expected in zip(expanded, repeated, strict=True):
        assert torch.equal(actual, expected)
    sum(value.square().sum() for value in repeated).backward()
    sum(value.square().sum() for value in expanded).backward()
    torch.testing.assert_close(expanded_memory.grad, repeated_memory.grad, rtol=0, atol=0)
    torch.testing.assert_close(expanded_proposals.grad, repeated_proposals.grad, rtol=0, atol=0)
    torch.testing.assert_close(expanded_queries.grad, repeated_queries.grad, rtol=0, atol=0)

    assert expanded_index.untyped_storage().data_ptr() == topk.untyped_storage().data_ptr()
    assert repeated_index.untyped_storage().data_ptr() != topk.untyped_storage().data_ptr()


def test_training_reuses_selected_encoder_memory_for_both_auxiliary_heads():
    config = SLConfig.fromfile("configs/lineae/lineae_s.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    model, _ = create(config, "modelname")
    model.train()
    box_inputs = []
    class_inputs = []

    box_hook = model.decoder.enc_out_bbox_embed.register_forward_pre_hook(
        lambda _module, inputs: box_inputs.append(inputs[0])
    )
    class_hook = model.decoder.enc_out_class_embed.register_forward_pre_hook(
        lambda _module, inputs: class_inputs.append(inputs[0])
    )
    try:
        outputs = model(torch.randn(2, 3, 64, 64), targets=None)
    finally:
        box_hook.remove()
        class_hook.remove()

    assert outputs["aux_interm_outputs"]["pred_logits"].shape == (2, 20, 2)
    assert len(box_inputs) == 1
    assert len(class_inputs) == 2
    assert class_inputs[1] is box_inputs[0]


def test_s_eval_batch_uses_optimized_decoder_without_changing_checkpoint_schema():
    torch.manual_seed(29)
    config = SLConfig.fromfile("configs/lineae/lineae_s.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    model, postprocessor = create(config, "modelname")
    model.eval()
    state_keys = tuple(model.state_dict())
    image = torch.randn(1, 3, 64, 64)
    images = image.expand(3, -1, -1, -1).clone()

    with torch.inference_mode():
        first = model(images)
        cached_frequency = model.decoder.decoder._sine_dim_t_cache[1]
        second = model(images)
        dynamic_image = torch.randn(1, 3, 96, 96)
        dynamic_first = model(dynamic_image)
        cached_anchors = next(iter(model.decoder._dynamic_anchor_cache.values()))
        cached_position = next(iter(model.encoder._position_embedding_cache.values()))
        dynamic_second = model(dynamic_image)

    assert first["pred_logits"].shape == (3, 20, 2)
    assert first["pred_lines"].shape == (3, 20, 4)
    processed = postprocessor(first, torch.ones(3, 2))
    assert all(result["lines"].shape == (10, 4) for result in processed)
    assert all(result["scores"].shape == (10,) for result in processed)
    for key in ("pred_logits", "pred_lines"):
        torch.testing.assert_close(second[key], first[key], rtol=0, atol=0)
        torch.testing.assert_close(first[key][1], first[key][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(first[key][2], first[key][0], rtol=1e-5, atol=1e-6)
    assert model.decoder.decoder._sine_dim_t_cache[1].data_ptr() == cached_frequency.data_ptr()
    for key in ("pred_logits", "pred_lines"):
        torch.testing.assert_close(dynamic_second[key], dynamic_first[key], rtol=0, atol=0)
    assert len(model.decoder._dynamic_anchor_cache) == 1
    assert next(iter(model.decoder._dynamic_anchor_cache.values()))[0] is cached_anchors[0]
    assert next(iter(model.decoder._dynamic_anchor_cache.values()))[1] is cached_anchors[1]
    assert len(model.encoder._position_embedding_cache) == 1
    assert next(iter(model.encoder._position_embedding_cache.values())) is cached_position
    assert tuple(model.state_dict()) == state_keys


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
def test_dynamic_anchor_cache_is_device_exact_bounded_and_not_checkpoint_state(device):
    config = SLConfig.fromfile("configs/lineae/lineae_s.py")
    config.pretrained = False
    model, _ = create(config, "modelname")
    decoder = model.decoder
    decoder._dynamic_anchor_cache_limit = 2
    state_keys = tuple(decoder.state_dict())
    shapes = [(8, 8), (4, 4), (2, 2)]

    expected = tuple(value.to(device) for value in decoder.generate_anchors(shapes))
    first = decoder._anchors_for_device(shapes, device)
    second = decoder._anchors_for_device(shapes, device)

    for actual, reference in zip(first, expected, strict=True):
        assert torch.equal(actual, reference)
    assert second[0] is first[0]
    assert second[1] is first[1]
    assert tuple(decoder.state_dict()) == state_keys

    decoder._anchors_for_device([(10, 10), (5, 5), (3, 3)], device)
    decoder._anchors_for_device([(12, 12), (6, 6), (3, 3)], device)
    assert len(decoder._dynamic_anchor_cache) == 2
    assert all(key[0] != tuple(shapes) for key in decoder._dynamic_anchor_cache)

    reloaded = decoder._anchors_for_device(shapes, device)
    assert reloaded[0] is not first[0]
    assert reloaded[1] is not first[1]
    assert reloaded[0].data_ptr() != first[0].data_ptr()
    assert reloaded[1].data_ptr() != first[1].data_ptr()
    decoder.to(device)
    assert decoder._dynamic_anchor_cache == {}
