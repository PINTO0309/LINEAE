"""Single authoritative mapping for LINEAE model variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    backbone: str
    checkpoint: str
    input_size: int
    preprocess_profile: str
    pyramid_channels: int | None


VARIANTS = {
    "A": VariantSpec("A", "hgnetv2_atto", "ckpts/PPHGNetV2_B0_stage1.pth", 320, "linea", None),
    "F": VariantSpec("F", "hgnetv2_femto", "ckpts/PPHGNetV2_B0_stage1.pth", 416, "linea", None),
    "P": VariantSpec("P", "hgnetv2_pico", "ckpts/PPHGNetV2_B0_stage1.pth", 640, "linea", None),
    "N": VariantSpec("N", "hgnetv2_n", "ckpts/PPHGNetV2_B0_stage1.pth", 640, "linea", None),
    "T": VariantSpec("T", "hgnetv2_t", "ckpts/PPHGNetV2_B1_stage1.pth", 640, "linea", None),
    "S": VariantSpec("S", "dinov3_vitt", "ckpts/vitt_distill.pt", 640, "imagenet", 192),
    "M": VariantSpec("M", "dinov3_vittplus", "ckpts/vittplus_distill.pt", 640, "imagenet", 256),
    "L": VariantSpec("L", "dinov3_vits16", "ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth", 640, "imagenet", 256),
    "X": VariantSpec("X", "dinov3_vits16plus", "ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth", 640, "imagenet", 256),
    "XL": VariantSpec("XL", "dinov3_vitb16", "ckpts/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", 640, "imagenet", 256),
    "2XL": VariantSpec("2XL", "dinov3_vitl16", "ckpts/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", 640, "imagenet", 512),
    "3XL": VariantSpec("3XL", "dinov3_vith16plus", "ckpts/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", 640, "imagenet", 640),
}

PREPROCESS_PROFILES = {
    "linea": {
        "mean": [0.538, 0.494, 0.453],
        "std": [0.257, 0.263, 0.273],
    },
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name.upper()]
    except KeyError as error:
        raise ValueError(f"unknown LINEAE variant {name!r}; expected one of {tuple(VARIANTS)}") from error


def validate_variant_config(args) -> VariantSpec | None:
    name = getattr(args, "variant", None)
    if name is None:
        return None
    spec = get_variant(name)
    checks = {
        "backbone": spec.backbone,
        "backbone_weights": spec.checkpoint,
    }
    profile = PREPROCESS_PROFILES[spec.preprocess_profile]
    checks.update({
        "image_mean": profile["mean"],
        "image_std": profile["std"],
    })
    if getattr(args, "enforce_variant_pyramid", True):
        checks["backbone_pyramid_channels"] = spec.pyramid_channels
    actual_overrides = {}
    if getattr(args, "enforce_variant_input", True):
        spatial_size = getattr(args, "eval_spatial_size", None)
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        if spatial_size is not None:
            spatial_size = tuple(spatial_size)
        checks["eval_spatial_size"] = (spec.input_size, spec.input_size)
        actual_overrides["eval_spatial_size"] = spatial_size
    for field, expected in checks.items():
        actual = actual_overrides.get(field, getattr(args, field, None))
        if actual != expected:
            raise ValueError(
                f"variant {spec.name} requires {field}={expected!r}, got {actual!r}"
            )
    return spec


__all__ = [
    "PREPROCESS_PROFILES",
    "VARIANTS",
    "VariantSpec",
    "get_variant",
    "validate_variant_config",
]
