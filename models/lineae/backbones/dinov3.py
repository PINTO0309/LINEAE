"""Self-contained DINOv3 backbones for LINEAE."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .base import CheckpointLoadReport, LINEAEBackbone, unwrap_state_dict
from .simple_feature_pyramid import SimpleFeaturePyramid


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
    ) -> None:
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim % 4:
            raise ValueError("DINOv3 RoPE head dimension must be divisible by four")
        if (base is None) == (min_period is None or max_period is None):
            raise ValueError("provide base, or both min_period and max_period")
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.head_dim = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.register_buffer("periods", torch.empty(head_dim // 4), persistent=True)
        self._eval_cache: tuple[tuple[object, ...], tuple[Tensor, Tensor]] | None = None
        self._init_weights()

    def _init_weights(self) -> None:
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.head_dim // 4, device=self.periods.device) / (self.head_dim // 2)
            )
        else:
            assert self.min_period is not None and self.max_period is not None
            ratio = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.head_dim // 4, device=self.periods.device)
            periods = self.max_period * ratio ** (exponents - 1)
        self.periods.data.copy_(periods)
        self._eval_cache = None

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._eval_cache = None
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._eval_cache = None
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._eval_cache = None

    def _eval_cache_key(self, *, height: int, width: int) -> tuple[object, ...]:
        device = self.periods.device
        return (
            height,
            width,
            device.type,
            device.index,
            self.periods.dtype,
            self.normalize_coords,
            None if self.periods.is_inference() else self.periods._version,
        )

    def _eval_cache_enabled(self) -> bool:
        return (
            not self.training
            and not torch.jit.is_tracing()
            and not torch.onnx.is_in_onnx_export()
        )

    def forward(self, *, height: int, width: int) -> tuple[Tensor, Tensor]:
        cache_enabled = self._eval_cache_enabled()
        cache_key = self._eval_cache_key(height=height, width=width)
        if cache_enabled and self._eval_cache is not None and self._eval_cache[0] == cache_key:
            return self._eval_cache[1]

        kwargs = {"device": self.periods.device, "dtype": torch.float32}
        if self.normalize_coords == "max":
            divisor_h = divisor_w = max(height, width)
        elif self.normalize_coords == "min":
            divisor_h = divisor_w = min(height, width)
        else:
            divisor_h, divisor_w = height, width
        coords_h = torch.arange(0.5, height, **kwargs) / divisor_h
        coords_w = torch.arange(0.5, width, **kwargs) / divisor_w
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **kwargs).uniform_(-self.shift_coords, self.shift_coords)
        if self.training and self.jitter_coords is not None:
            jitter = torch.empty(2, **kwargs).uniform_(
                -np.log(self.jitter_coords), np.log(self.jitter_coords)
            ).exp()
            coords *= jitter
        if self.training and self.rescale_coords is not None:
            rescale = torch.empty(1, **kwargs).uniform_(
                -np.log(self.rescale_coords), np.log(self.rescale_coords)
            ).exp()
            coords *= rescale

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)
        rope = torch.sin(angles)[None, None], torch.cos(angles)[None, None]
        if cache_enabled:
            # Keep only the most recent shape so multiscale evaluation cannot
            # accumulate one device allocation per encountered resolution.
            self._eval_cache = cache_key, rope
        return rope


def _rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(value: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    if sin.shape != cos.shape:
        raise ValueError("RoPE sin and cos tensors must have identical shapes")
    if sin.shape[-1] == value.shape[-1]:
        # Compatibility with the original full-width representation.
        return value * cos + _rotate_half(value) * sin
    if sin.shape[-1] * 2 != value.shape[-1]:
        raise ValueError(
            f"RoPE width {sin.shape[-1]} is incompatible with head width {value.shape[-1]}"
        )
    first, second = value.chunk(2, dim=-1)
    return torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
        ),
        dim=-1,
    )


def _apply_rope_in_place(value: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Apply half-width RoPE into disposable Q/K storage during inference."""
    if sin.shape != cos.shape:
        raise ValueError("RoPE sin and cos tensors must have identical shapes")
    if sin.shape[-1] * 2 != value.shape[-1]:
        # Full-width inputs are retained for compatibility tests and external
        # callers, but the model-owned embeddings always use half width.
        value.copy_(_apply_rope(value, sin, cos))
        return value
    first, second = value.chunk(2, dim=-1)
    rotated_first = first * cos - second * sin
    rotated_second = second * cos + first * sin
    first.copy_(rotated_first)
    second.copy_(rotated_second)
    return value


def _can_reuse_qk_storage() -> bool:
    return (
        not torch.is_grad_enabled()
        and not torch.jit.is_tracing()
        and not torch.onnx.is_in_onnx_export()
    )


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: Tensor) -> Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)

    def reset_parameters(self) -> None:
        # Match the official DINOv3 PatchEmbed initialization. ``Conv2d`` uses
        # the same fan-in uniform rule at construction, but exposing the reset
        # keeps the complete ViT initialization sequence explicit and reusable.
        self.proj.reset_parameters()


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * ratio), dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, value: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(value)))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, value: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, val = qkv.unbind(0)
        query_cls, query_patch = query[:, :, :1], query[:, :, 1:]
        key_cls, key_patch = key[:, :, :1], key[:, :, 1:]
        if _can_reuse_qk_storage():
            # Q/K are disposable views into this forward's fresh QKV projection.
            # Reuse them during Torch inference instead of allocating both a
            # rotated patch tensor and a prefix-concatenated tensor per Q/K.
            _apply_rope_in_place(query_patch, sin, cos)
            _apply_rope_in_place(key_patch, sin, cos)
        else:
            query = torch.cat((query_cls, _apply_rope(query_patch, sin, cos)), dim=2)
            key = torch.cat((key_cls, _apply_rope(key_patch, sin, cos)), dim=2)
        # Dropout is fixed to zero for the pretrained Tiny/Tiny+ architecture.
        # SDPA preserves the same scaled-softmax attention while allowing PyTorch
        # to select memory-efficient CUDA kernels instead of materializing the
        # complete [B, H, tokens, tokens] attention matrix at 640x640.
        output = F.scaled_dot_product_attention(
            query,
            key,
            val,
            dropout_p=0.0,
        ).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = norm(dim)
        self.mlp = Mlp(dim)

    def forward(self, value: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        value = value + self.drop_path(self.attn(self.norm1(value), sin, cos))
        return value + self.drop_path(self.mlp(self.norm2(value)))


def _validate_intermediate_layers(layers, depth: int) -> tuple[int, ...]:
    layers = tuple(int(layer) for layer in layers)
    if layers != tuple(sorted(set(layers))):
        raise ValueError("DINO intermediate layers must be sorted and unique")
    if any(layer < 0 or layer >= depth for layer in layers):
        raise ValueError(f"DINO intermediate layers must be in [0, {depth - 1}]")
    return layers


class DinoIntermediateFusion(nn.Module):
    """Low-overhead learned fusion of same-resolution transformer block maps."""

    def __init__(self, channels: int, levels: int) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("intermediate fusion requires at least two block outputs")
        self.level_weights = nn.Parameter(torch.zeros(levels))
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.dirac_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != self.level_weights.numel():
            raise ValueError("intermediate feature count does not match fusion adapter")
        reference_shape = features[0].shape
        if any(feature.shape != reference_shape for feature in features):
            raise ValueError("all intermediate DINO features must have identical shapes")
        weights = self.level_weights.softmax(dim=0)
        fused = sum(weight * feature for weight, feature in zip(weights, features, strict=True))
        return self.projection(fused)


class CompactDinoV3(nn.Module):
    """The exact checkpoint-owned Tiny/Tiny+ module (148 tensor keys)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int = 12,
        patch_size: int = 16,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads) for _ in range(depth))
        self.rope_embed = RopePositionEmbedding(embed_dim, num_heads=num_heads)

    def forward(self, images: Tensor, return_layers=()) -> Tensor | list[Tensor]:
        batch, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"input {(height, width)} must be divisible by patch size {self.patch_size}"
            )
        value = self.patch_embed(images)
        value = torch.cat((self.cls_token.expand(batch, -1, -1), value), dim=1)
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        sin, cos = self.rope_embed(height=grid_h, width=grid_w)
        sin, cos = sin.to(value.dtype), cos.to(value.dtype)
        return_layers = _validate_intermediate_layers(return_layers, len(self.blocks))
        requested = set(return_layers)
        intermediate = []
        for index, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training and value.requires_grad:
                # The compact blocks have zero dropout/stochastic depth and RoPE
                # randomness is resolved before entering the block, so preserving
                # RNG state only adds checkpoint overhead.
                value = activation_checkpoint(
                    block,
                    value,
                    sin,
                    cos,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                value = block(value, sin, cos)
            if index in requested:
                patches = value[:, 1:].transpose(1, 2).contiguous()
                intermediate.append(patches.reshape(batch, self.embed_dim, grid_h, grid_w))
        if return_layers:
            return intermediate
        patches = value[:, 1:].transpose(1, 2).contiguous()
        return patches.reshape(batch, self.embed_dim, grid_h, grid_w)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.init_value = init_value
        self.gamma = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.gamma, self.init_value)

    def forward(self, value: Tensor) -> Tensor:
        return value * self.gamma


class LinearKMaskedBias(nn.Linear):
    """Official DINOv3 QKV projection with the key bias masked to zero."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)
        mask = torch.ones(out_features)
        third = out_features // 3
        mask[third:2 * third] = 0
        self.register_buffer("bias_mask", mask, persistent=True)

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, self.weight, self.bias * self.bias_mask.to(self.bias.dtype))


class OfficialAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = LinearKMaskedBias(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def _apply_rope_with_prefix(
        self, query: Tensor, key: Tensor, sin: Tensor, cos: Tensor
    ) -> tuple[Tensor, Tensor]:
        query_dtype, key_dtype = query.dtype, key.dtype
        query, key = query.float(), key.float()
        prefix = query.shape[-2] - sin.shape[-2]
        if prefix < 0:
            raise RuntimeError("RoPE patch count exceeds attention token count")
        if _can_reuse_qk_storage():
            _apply_rope_in_place(query[:, :, prefix:], sin, cos)
            _apply_rope_in_place(key[:, :, prefix:], sin, cos)
        else:
            query = torch.cat(
                (query[:, :, :prefix], _apply_rope(query[:, :, prefix:], sin, cos)),
                dim=2,
            )
            key = torch.cat(
                (key[:, :, :prefix], _apply_rope(key[:, :, prefix:], sin, cos)),
                dim=2,
            )
        return query.to(query_dtype), key.to(key_dtype)

    def forward(self, value: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        query, key, val = torch.unbind(qkv, dim=2)
        query, key, val = (item.transpose(1, 2) for item in (query, key, val))
        query, key = self._apply_rope_with_prefix(query, key, sin, cos)
        output = F.scaled_dot_product_attention(query, key, val)
        output = output.transpose(1, 2).contiguous().reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class OfficialMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, value: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(value)))))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, align_to: int = 8) -> None:
        super().__init__()
        width = int(hidden_dim * 2 / 3)
        width += -width % align_to
        self.w1 = nn.Linear(dim, width)
        self.w2 = nn.Linear(dim, width)
        self.w3 = nn.Linear(width, dim)

    def forward(self, value: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(value)) * self.w2(value))


class OfficialBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float, swiglu: bool) -> None:
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-5)
        self.norm1 = norm(dim)
        self.attn = OfficialAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = norm(dim)
        hidden_dim = int(dim * ffn_ratio)
        self.mlp = SwiGLUFFN(dim, hidden_dim) if swiglu else OfficialMlp(dim, hidden_dim)
        self.ls2 = LayerScale(dim)

    def forward(self, value: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        value = value + self.ls1(self.attn(self.norm1(value), sin, cos))
        return value + self.ls2(self.mlp(self.norm2(value)))


class OfficialDinoV3(nn.Module):
    """Local minimal DINOv3 ViT implementation for official checkpoints."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        ffn_ratio: float,
        swiglu: bool,
        depth: int = 12,
        patch_size: int = 16,
        storage_tokens: int = 4,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_storage_tokens = storage_tokens
        self.use_checkpoint = use_checkpoint
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.empty(1, storage_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim,
            num_heads=num_heads,
            rescale_coords=2.0,
        )
        self.blocks = nn.ModuleList(
            OfficialBlock(embed_dim, num_heads, ffn_ratio, swiglu) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.init_weights()

    @staticmethod
    def _init_module_weights(module: nn.Module) -> None:
        """Apply the upstream DINOv3 ViT initialization to local modules."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, LinearKMaskedBias):
                third = module.out_features // 3
                module.bias_mask.fill_(1)
                module.bias_mask[third:2 * third].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        elif isinstance(module, LayerScale):
            module.reset_parameters()
        elif isinstance(module, PatchEmbed):
            module.reset_parameters()

    def init_weights(self) -> None:
        """Initialize a usable official ViT before optional checkpoint loading."""
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        self.apply(self._init_module_weights)

    def forward(self, images: Tensor, return_layers=()) -> Tensor | list[Tensor]:
        batch, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"input {(height, width)} must be divisible by patch size {self.patch_size}"
            )
        patches = self.patch_embed(images)
        cls = self.cls_token + 0 * self.mask_token
        value = torch.cat((
            cls.expand(batch, -1, -1),
            self.storage_tokens.expand(batch, -1, -1),
            patches,
        ), dim=1)
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        sin, cos = self.rope_embed(height=grid_h, width=grid_w)
        return_layers = _validate_intermediate_layers(return_layers, len(self.blocks))
        requested = set(return_layers)
        intermediate = []
        for index, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training and value.requires_grad:
                # Official S/S+/B checkpoints also use zero stochastic depth and
                # dropout. Coordinate rescaling happens once outside each block.
                value = activation_checkpoint(
                    block,
                    value,
                    sin,
                    cos,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                value = block(value, sin, cos)
            if index in requested:
                normalized = self.norm(value[:, self.n_storage_tokens + 1:])
                intermediate.append(normalized.transpose(1, 2).contiguous().reshape(
                    batch, self.embed_dim, grid_h, grid_w
                ))
        if return_layers:
            return intermediate
        value = self.norm(value[:, self.n_storage_tokens + 1:])
        return value.transpose(1, 2).contiguous().reshape(
            batch, self.embed_dim, grid_h, grid_w
        )


class CompactDinoV3Backbone(LINEAEBackbone):
    preprocess_profile = "imagenet"

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        weights_path: str | Path | None,
        pyramid_channels: int | None = None,
        trainable_depth: int = 2,
        use_checkpoint: bool = False,
        intermediate_layers=(),
    ) -> None:
        super().__init__()
        self.core = CompactDinoV3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )
        if weights_path is not None:
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"DINOv3 checkpoint not found: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = unwrap_state_dict(checkpoint)
            incompatible = self.core.load_state_dict(state, strict=True)
            self.checkpoint_report = CheckpointLoadReport(
                path=path,
                tensor_count=len(state),
                missing_keys=tuple(incompatible.missing_keys),
                unexpected_keys=tuple(incompatible.unexpected_keys),
            )
        channels = pyramid_channels or embed_dim
        self.intermediate_layers = _validate_intermediate_layers(
            intermediate_layers, len(self.core.blocks)
        )
        self.intermediate_fusion = (
            DinoIntermediateFusion(embed_dim, len(self.intermediate_layers))
            if self.intermediate_layers else None
        )
        self.pyramid = SimpleFeaturePyramid(embed_dim, channels)
        self.out_channels = (channels, channels, channels)
        self.set_trainable_depth(trainable_depth)

    @property
    def num_blocks(self) -> int:
        return len(self.core.blocks)

    def set_trainable_depth(self, depth: int) -> Sequence[nn.Parameter]:
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        if depth < 0:
            modules = []
        elif depth == 0 or depth >= self.num_blocks:
            modules: Sequence[nn.Module] = [self.core]
        else:
            modules = list(self.core.blocks[-depth:])
        for module in modules:
            module.requires_grad_(True)
        if depth > 0 and depth < self.num_blocks:
            self.core.cls_token.requires_grad_(True)
        return [parameter for parameter in self.core.parameters() if parameter.requires_grad]

    def forward(self, images: Tensor) -> list[Tensor]:
        if images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError(f"LINEAE DINO inputs must be divisible by 32, got {images.shape[-2:]}")
        feature = self.core(images, return_layers=self.intermediate_layers)
        if self.intermediate_fusion is not None:
            feature = self.intermediate_fusion(feature)
        features = self.pyramid(feature)
        self.validate_features(images, features)
        self.last_feature_shapes = [tuple(feature.shape) for feature in features]
        return features


class OfficialDinoV3Backbone(LINEAEBackbone):
    preprocess_profile = "imagenet"

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        ffn_ratio: float,
        swiglu: bool,
        weights_path: str | Path | None,
        depth: int = 12,
        pyramid_channels: int | None = None,
        trainable_depth: int = 2,
        use_checkpoint: bool = False,
        intermediate_layers=(),
    ) -> None:
        super().__init__()
        self.core = OfficialDinoV3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            swiglu=swiglu,
            depth=depth,
            use_checkpoint=use_checkpoint,
        )
        if weights_path is not None:
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"DINOv3 checkpoint not found: {path}")
            state = unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            incompatible = self.core.load_state_dict(state, strict=True)
            self.checkpoint_report = CheckpointLoadReport(
                path=path,
                tensor_count=len(state),
                missing_keys=tuple(incompatible.missing_keys),
                unexpected_keys=tuple(incompatible.unexpected_keys),
            )
        channels = pyramid_channels or embed_dim
        self.intermediate_layers = _validate_intermediate_layers(
            intermediate_layers, len(self.core.blocks)
        )
        self.intermediate_fusion = (
            DinoIntermediateFusion(embed_dim, len(self.intermediate_layers))
            if self.intermediate_layers else None
        )
        self.pyramid = SimpleFeaturePyramid(embed_dim, channels)
        self.out_channels = (channels, channels, channels)
        self.set_trainable_depth(trainable_depth)

    @property
    def num_blocks(self) -> int:
        return len(self.core.blocks)

    def set_trainable_depth(self, depth: int) -> Sequence[nn.Parameter]:
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        if depth < 0:
            modules = []
        elif depth == 0 or depth >= self.num_blocks:
            modules: Sequence[nn.Module] = [self.core]
        else:
            modules = [*self.core.blocks[-depth:], self.core.norm]
        for module in modules:
            module.requires_grad_(True)
        if 0 < depth < self.num_blocks:
            self.core.cls_token.requires_grad_(True)
            self.core.storage_tokens.requires_grad_(True)
        return [parameter for parameter in self.core.parameters() if parameter.requires_grad]

    def forward(self, images: Tensor) -> list[Tensor]:
        if images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError(f"LINEAE DINO inputs must be divisible by 32, got {images.shape[-2:]}")
        feature = self.core(images, return_layers=self.intermediate_layers)
        if self.intermediate_fusion is not None:
            feature = self.intermediate_fusion(feature)
        features = self.pyramid(feature)
        self.validate_features(images, features)
        self.last_feature_shapes = [tuple(feature.shape) for feature in features]
        return features
