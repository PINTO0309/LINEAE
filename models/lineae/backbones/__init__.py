"""Backbone factory for LINEAE variants."""

from __future__ import annotations

from .dinov3 import CompactDinoV3Backbone, OfficialDinoV3Backbone
from .hgnetv2 import HGNetV2Backbone
from ..variants import validate_variant_config


def build_backbone(args):
    validate_variant_config(args)
    name = args.backbone.lower()
    if name == "dinov3_vitt":
        return CompactDinoV3Backbone(
            embed_dim=192,
            num_heads=3,
            weights_path=args.backbone_weights if args.pretrained else None,
            pyramid_channels=getattr(args, "backbone_pyramid_channels", None),
            trainable_depth=getattr(args, "backbone_trainable_layers", 2),
            use_checkpoint=getattr(args, "use_checkpoint", False),
            intermediate_layers=getattr(args, "dino_intermediate_layers", ()),
        )
    if name == "dinov3_vittplus":
        return CompactDinoV3Backbone(
            embed_dim=256,
            num_heads=4,
            weights_path=args.backbone_weights if args.pretrained else None,
            pyramid_channels=getattr(args, "backbone_pyramid_channels", None),
            trainable_depth=getattr(args, "backbone_trainable_layers", 2),
            use_checkpoint=getattr(args, "use_checkpoint", False),
            intermediate_layers=getattr(args, "dino_intermediate_layers", ()),
        )
    official_specs = {
        "dinov3_vits16": dict(embed_dim=384, num_heads=6, ffn_ratio=4.0, swiglu=False),
        "dinov3_vits16plus": dict(embed_dim=384, num_heads=6, ffn_ratio=6.0, swiglu=True),
        "dinov3_vitb16": dict(embed_dim=768, num_heads=12, ffn_ratio=4.0, swiglu=False),
    }
    if name in official_specs:
        return OfficialDinoV3Backbone(
            **official_specs[name],
            weights_path=args.backbone_weights if args.pretrained else None,
            pyramid_channels=getattr(args, "backbone_pyramid_channels", None),
            trainable_depth=getattr(args, "backbone_trainable_layers", 2),
            use_checkpoint=getattr(args, "use_checkpoint", False),
            intermediate_layers=getattr(args, "dino_intermediate_layers", ()),
        )

    if name.startswith("hgnetv2_"):
        return HGNetV2Backbone(
            name=name,
            weights_path=args.backbone_weights if args.pretrained else None,
            use_lab=getattr(args, "use_lab", True),
            freeze_norm=getattr(args, "freeze_norm", False),
            trainable_depth=getattr(args, "backbone_trainable_layers", 0),
        )
    raise ValueError(f"unsupported backbone: {args.backbone}")


__all__ = [
    "CompactDinoV3Backbone",
    "HGNetV2Backbone",
    "OfficialDinoV3Backbone",
    "build_backbone",
]
