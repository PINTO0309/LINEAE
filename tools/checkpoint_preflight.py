"""Validate all immutable bootstrap checkpoints before a training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from models.lineae.backbones.base import unwrap_state_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "ckpts" / "MANIFEST.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_depth(keys: list[str]) -> int:
    indexes = []
    for key in keys:
        match = re.match(r"blocks\.(\d+)\.", key)
        if match:
            indexes.append(int(match.group(1)))
    return max(indexes) + 1 if indexes else 0


def _expected_shapes(family: str) -> dict[str, tuple[int, ...]] | None:
    with torch.device("meta"):
        if family == "hgnetv2_b0":
            from models.lineae.hgnetv2 import HGNetv2

            model = HGNetv2(
                "B0", use_lab=True, return_idx=[1, 2, 3], freeze_at=-1,
                freeze_norm=False, pretrained=False,
            )
        elif family in {"dinov3_vitt", "dinov3_vittplus"}:
            from models.lineae.backbones.dinov3 import CompactDinoV3

            width, heads = (192, 3) if family == "dinov3_vitt" else (256, 4)
            model = CompactDinoV3(embed_dim=width, num_heads=heads)
        elif family in {
            "dinov3_vits16",
            "dinov3_vits16plus",
            "dinov3_vitb16",
            "dinov3_vitl16",
            "dinov3_vith16plus",
        }:
            from models.lineae.backbones.dinov3 import OfficialDinoV3

            specs = {
                "dinov3_vits16": (384, 6, 4.0, False, 12),
                "dinov3_vits16plus": (384, 6, 6.0, True, 12),
                "dinov3_vitb16": (768, 12, 4.0, False, 12),
                "dinov3_vitl16": (1024, 16, 4.0, False, 24),
                "dinov3_vith16plus": (1280, 20, 6.0, True, 32),
            }
            width, heads, ratio, swiglu, depth = specs[family]
            model = OfficialDinoV3(
                embed_dim=width,
                num_heads=heads,
                ffn_ratio=ratio,
                swiglu=swiglu,
                depth=depth,
            )
        else:
            return None
    return {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
        if not key.endswith(".num_batches_tracked")
    }


def verify_checkpoint(path: Path, specification: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != specification["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {specification['sha256']}, got {actual_hash}"
        )
    with FakeTensorMode():
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    state = unwrap_state_dict(checkpoint)
    if len(state) != specification["tensor_count"]:
        raise ValueError(
            f"tensor-count mismatch for {path.name}: expected {specification['tensor_count']}, "
            f"got {len(state)}"
        )
    expected_shapes = _expected_shapes(specification["family"])
    if expected_shapes is not None:
        actual_shapes = {key: tuple(value.shape) for key, value in state.items()}
        missing = sorted(set(expected_shapes) - set(actual_shapes))
        unexpected = sorted(set(actual_shapes) - set(expected_shapes))
        mismatched = sorted(
            key for key in set(expected_shapes) & set(actual_shapes)
            if expected_shapes[key] != actual_shapes[key]
        )
        if missing or unexpected or mismatched:
            mismatch_details = [
                f"{key}: expected={expected_shapes[key]}, got={actual_shapes[key]}"
                for key in mismatched[:5]
            ]
            raise ValueError(
                f"architecture mismatch for {path.name}: missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}, shape_mismatches={mismatch_details}"
            )
    if "embed_dim" in specification:
        cls_token = state.get("cls_token")
        if cls_token is None or cls_token.shape[-1] != specification["embed_dim"]:
            raise ValueError(f"embedding-width mismatch for {path.name}")
    if "depth" in specification:
        actual_depth = _infer_depth(list(state))
        if actual_depth != specification["depth"]:
            raise ValueError(
                f"depth mismatch for {path.name}: expected {specification['depth']}, got {actual_depth}"
            )
    return {
        "path": str(path),
        "sha256": actual_hash,
        "tensor_count": len(state),
        "family": specification["family"],
        "status": "ok",
    }


def verify_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_dir = manifest_path.parent
    return [
        verify_checkpoint(checkpoint_dir / filename, specification)
        for filename, specification in manifest.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(verify_manifest(args.manifest), indent=2))


if __name__ == "__main__":
    main()
