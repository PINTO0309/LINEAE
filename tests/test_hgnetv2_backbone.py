from types import SimpleNamespace

import pytest
import torch

from models.lineae.backbones import build_backbone


@pytest.mark.parametrize(
    ("name", "channels", "loaded", "strict"),
    [
        ("hgnetv2_atto", (256, 256, 256), 152, False),
        ("hgnetv2_femto", (256, 512, 512), 165, False),
        ("hgnetv2_pico", (256, 512, 512), 215, False),
        ("hgnetv2_n", (256, 512, 1024), 270, True),
    ],
)
def test_hgnet_variants_load_report_and_feature_contract(name, channels, loaded, strict):
    args = SimpleNamespace(
        backbone=name,
        backbone_weights="ckpts/PPHGNetV2_B0_stage1.pth",
        pretrained=True,
        use_lab=True,
        freeze_norm=False,
        backbone_trainable_layers=0,
    )
    backbone = build_backbone(args).eval()
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        features = backbone(images)

    assert backbone.out_channels == channels
    assert backbone.out_strides == (8, 16, 32)
    assert [feature.shape[1] for feature in features] == list(channels)
    assert [feature.shape[-2:] for feature in features] == [(8, 8), (4, 4), (2, 2)]
    report = backbone.checkpoint_report
    assert report.source_tensor_count == 270
    assert len(report.loaded_keys) == loaded
    assert report.strict is strict
    assert len(report.loaded_keys) + len(report.unexpected_keys) + len(report.shape_mismatches) == 270


def test_hgnet_derivative_p5_is_learned_and_progressive_depth_freezes_earlier_stages():
    args = SimpleNamespace(
        backbone="hgnetv2_atto",
        backbone_weights=None,
        pretrained=False,
        use_lab=True,
        freeze_norm=False,
        backbone_trainable_layers=1,
    )
    backbone = build_backbone(args)
    assert not any(parameter.requires_grad for parameter in backbone.core.stages[0].parameters())
    assert any(parameter.requires_grad for parameter in backbone.core.stages[-1].parameters())
    assert all(parameter.requires_grad for parameter in backbone.p5.parameters())
    assert isinstance(backbone.p5[0], torch.nn.AvgPool2d)
    assert backbone.p5[0].kernel_size == 2
    assert backbone.p5[0].stride == 2
    assert backbone.p5[1].kernel_size == (1, 1)
    assert backbone.p5[1].groups == 1

    features = backbone(torch.randn(2, 3, 64, 64))
    sum(feature.mean() for feature in features).backward()
    assert backbone.p5[1].weight.grad is not None
