import hashlib
from pathlib import Path

import pytest
import torch

from tools.checkpoint_preflight import DEFAULT_MANIFEST, verify_checkpoint, verify_manifest


def test_all_bootstrap_checkpoints_match_manifest():
    assert DEFAULT_MANIFEST.is_file()
    reports = verify_manifest(DEFAULT_MANIFEST)
    assert len(reports) == 6
    assert all(report["status"] == "ok" for report in reports)
    assert all(Path(report["path"]).is_file() for report in reports)


def test_preflight_rejects_a_hash_mismatch(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"value": torch.ones(1)}, path)
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint(path, {
            "sha256": "0" * 64,
            "tensor_count": 1,
            "family": "test",
        })
    report = verify_checkpoint(path, {
        "sha256": actual_hash,
        "tensor_count": 1,
        "family": "test",
    })
    assert report["status"] == "ok"


def test_preflight_reports_architecture_keys_and_shapes(tmp_path):
    path = tmp_path / "malformed.pt"
    state = torch.load("ckpts/vitt_distill.pt", map_location="cpu", weights_only=True)
    state["patch_embed.proj.weight"] = state["patch_embed.proj.weight"][:1]
    torch.save(state, path)
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="architecture mismatch.*shape_mismatches"):
        verify_checkpoint(path, {
            "sha256": actual_hash,
            "tensor_count": 148,
            "family": "dinov3_vitt",
        })
