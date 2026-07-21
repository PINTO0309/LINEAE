from pathlib import Path

import cv2
import numpy as np
import pytest

from demo_lineae import (
    PREPROCESS_PROFILES,
    annotate_image,
    build_providers,
    infer_variant_from_model,
    preprocess_bgr_image,
    resolve_variant,
    sigmoid,
)


def test_demo_does_not_depend_on_an_export_report():
    source = Path("demo_lineae.py").read_text(encoding="utf-8")
    assert "export_report" not in source
    assert ".export.json" not in source
    assert "onnx_sha256" not in source


def test_variant_is_inferred_from_model_name_or_explicitly_selected():
    assert infer_variant_from_model(Path("lineae_n.onnx")) == "N"
    assert infer_variant_from_model(Path("optimized_lineae_x_speed.onnx")) == "X"
    assert infer_variant_from_model(Path("lineae_xl_1x3x640x640.onnx")) == "XL"
    assert infer_variant_from_model(Path("custom.onnx")) is None
    assert resolve_variant(None, Path("lineae_xl.onnx")) == "XL"
    assert resolve_variant("A", Path("custom.onnx")) == "A"
    assert resolve_variant("N", Path("lineae_xl.onnx")) == "N"
    with pytest.raises(ValueError, match="specify --variant"):
        resolve_variant(None, Path("custom.onnx"))


def test_tensorrt_cache_is_written_beside_the_onnx(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "demo_lineae.ort.get_available_providers",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    )
    model_path = tmp_path / "deployment" / "lineae_xl.onnx"

    _, providers = build_providers("tensorrt", model_path)

    provider_name, provider_options = providers[0]
    assert provider_name == "TensorrtExecutionProvider"
    assert provider_options["trt_engine_cache_enable"] is True
    assert provider_options["trt_engine_cache_path"] == str(model_path.parent)


def test_demo_preprocessing_matches_opencv_linear_rgb_normalization():
    bgr = np.array(
        [
            [[0, 10, 255], [30, 20, 10], [50, 100, 150]],
            [[255, 128, 0], [70, 60, 50], [90, 80, 70]],
        ],
        dtype=np.uint8,
    )
    mean, std = PREPROCESS_PROFILES["imagenet"]

    actual = preprocess_bgr_image(
        bgr,
        size_hw=(3, 5),
        mean=mean,
        std=std,
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (5, 3), interpolation=cv2.INTER_LINEAR)
    expected = resized.astype(np.float32) / np.float32(255.0)
    expected = (expected - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    expected = np.ascontiguousarray(expected.transpose(2, 0, 1)[None])

    assert actual.shape == (1, 3, 3, 5)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert np.array_equal(actual, expected)


def test_sigmoid_is_stable_and_render_filter_is_bounded():
    actual = sigmoid(np.asarray([-1000.0, 0.0, 1000.0], dtype=np.float32))
    assert np.array_equal(actual, np.asarray([0.0, 0.5, 1.0], dtype=np.float32))

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    lines = np.asarray(
        [[0, 0, 31, 31], [0, 31, 31, 0], [0, 16, 31, 16]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    rendered, count = annotate_image(
        image,
        lines,
        scores,
        threshold=0.75,
        max_lines=1,
    )

    assert count == 1
    assert rendered.shape == image.shape
    assert np.any(rendered != image)
