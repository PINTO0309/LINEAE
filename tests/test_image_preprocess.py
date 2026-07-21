from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from util.image_preprocess import (
    IMAGE_PREPROCESS_SCHEMA,
    preprocess_image_file,
    read_rgb_image,
    resize_rgb_image,
    rgb_image_to_normalized_tensor,
    validate_checkpoint_image_preprocess,
)


def test_read_rgb_image_converts_bgr_and_disables_exif_rotation(monkeypatch):
    captured = {}
    bgr = np.array([[[3, 2, 1], [30, 20, 10]]], dtype=np.uint8)

    def fake_imread(path, flags):
        captured["path"] = path
        captured["flags"] = flags
        return bgr.copy()

    monkeypatch.setattr(cv2, "imread", fake_imread)
    image = read_rgb_image("sample.jpg")

    assert captured["path"] == "sample.jpg"
    assert captured["flags"] & cv2.IMREAD_IGNORE_ORIENTATION
    assert np.array_equal(image, bgr[..., ::-1])
    assert image.flags.c_contiguous


def test_read_rgb_image_reports_the_failed_path(tmp_path):
    missing = tmp_path / "missing.jpg"
    with pytest.raises(RuntimeError, match=str(missing)):
        read_rgb_image(missing)


@pytest.mark.parametrize("source_hw,target_hw", [((3, 5), (7, 4)), ((7, 4), (3, 5))])
def test_resize_is_standard_opencv_linear(source_hw, target_hw):
    image = np.arange(source_hw[0] * source_hw[1] * 3, dtype=np.uint8).reshape(
        *source_hw, 3
    )
    actual = resize_rgb_image(image, target_hw)
    expected = cv2.resize(
        image,
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    nearest = cv2.resize(
        image,
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, nearest)


def test_rgb_normalization_and_file_preprocessing(tmp_path):
    rgb = np.array([[[255, 128, 0], [0, 64, 255]]], dtype=np.uint8)
    path = tmp_path / "pixels.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    tensor = rgb_image_to_normalized_tensor(
        rgb,
        mean=[0.5, 0.25, 0.0],
        std=[0.5, 0.25, 1.0],
    )
    expected = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
    expected = (
        expected - torch.tensor([0.5, 0.25, 0.0]).view(3, 1, 1)
    ) / torch.tensor([0.5, 0.25, 1.0]).view(3, 1, 1)
    torch.testing.assert_close(tensor, expected)

    preprocessed = preprocess_image_file(
        path,
        (2, 4),
        mean=[0.5, 0.25, 0.0],
        std=[0.5, 0.25, 1.0],
    )
    assert preprocessed.shape == (3, 2, 4)


def test_detector_checkpoints_require_the_opencv_schema():
    validate_checkpoint_image_preprocess(
        {"config": {"image_preprocess_schema": IMAGE_PREPROCESS_SCHEMA}}
    )
    with pytest.raises(ValueError, match="predates"):
        validate_checkpoint_image_preprocess({"config": {}})
    with pytest.raises(ValueError, match="unsupported"):
        validate_checkpoint_image_preprocess(
            {"config": {"image_preprocess_schema": "pillow_bilinear_v0"}}
        )
    with pytest.raises(ValueError, match="unsupported"):
        validate_checkpoint_image_preprocess(
            {"config": {"image_preprocess_schema": "opencv_rgb_inter_nearest_v1"}}
        )


def test_project_image_pipeline_has_no_pillow_or_torchvision_imports():
    root = Path(__file__).resolve().parents[1]
    for path in [*root.glob("**/*.py"), root / "pyproject.toml"]:
        if any(part in {".venv", ".git"} for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8")
        assert "from " + "PIL" not in source
        assert "import " + "PIL" not in source
    project = (root / "pyproject.toml").read_text(encoding="utf-8").lower()
    lock = (root / "uv.lock").read_text(encoding="utf-8").lower()
    assert '"pillow==' not in project
    assert '"torchvision==' not in project
    assert '\nname = "torchvision"\n' not in lock


def test_tensorboard_viewer_and_tensorboardx_writer_are_compatibly_pinned():
    root = Path(__file__).resolve().parents[1]
    project = (root / "pyproject.toml").read_text(encoding="utf-8").lower()
    lock = (root / "uv.lock").read_text(encoding="utf-8").lower()
    assert '"tensorboard==2.21.0"' in project
    assert '"tensorboardx==2.6.5"' in project
    assert '\nname = "tensorboard"\n' in lock
    assert '\nname = "tensorboardx"\n' in lock
