import torch

from models.lineae.dn_components import prepare_for_cdn


def test_prepare_for_cdn_uses_target_device_on_cpu():
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.long),
            "lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]], dtype=torch.float32),
        }
    ]
    label_enc = torch.nn.Embedding(1, 16)

    query_labels, query_lines, attention_mask, metadata = prepare_for_cdn(
        (targets, 2, 0.2, 0.4),
        training=True,
        num_queries=10,
        num_classes=1,
        hidden_dim=16,
        label_enc=label_enc,
    )

    assert query_labels.device.type == "cpu"
    assert query_lines.device.type == "cpu"
    assert attention_mask.device.type == "cpu"
    assert torch.isfinite(query_labels).all()
    assert torch.isfinite(query_lines).all()
    assert metadata["pad_size"] == 8


def test_prepare_for_cdn_inference_returns_no_queries():
    result = prepare_for_cdn(
        None,
        training=False,
        num_queries=10,
        num_classes=1,
        hidden_dim=16,
        label_enc=torch.nn.Embedding(1, 16),
    )
    assert result == (None, None, None, None)
