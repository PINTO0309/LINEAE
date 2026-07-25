from pathlib import Path

import cv2
import numpy as np
import pytest

import demo_lineae
from demo_lineae import (
    LINEA_VARIANTS,
    PREPROCESS_PROFILES,
    annotate_image,
    build_providers,
    infer_variant_from_model,
    preprocess_bgr_image,
    process_video_source,
    resolve_variant,
    select_camera_recording_fps,
    sigmoid,
)


def test_demo_does_not_depend_on_an_export_report():
    source = Path("demo_lineae.py").read_text(encoding="utf-8")
    assert "export_report" not in source
    assert ".export.json" not in source
    assert "onnx_sha256" not in source


def test_variant_is_inferred_from_model_name_or_explicitly_selected():
    assert infer_variant_from_model(Path("lineae_n.onnx")) == "N"
    assert infer_variant_from_model(Path("lineae_t.onnx")) == "T"
    assert infer_variant_from_model(Path("optimized_lineae_x_speed.onnx")) == "X"
    assert infer_variant_from_model(Path("lineae_xl_1x3x640x640.onnx")) == "XL"
    assert infer_variant_from_model(Path("lineae_2xl_1x3x640x640.onnx")) == "2XL"
    assert infer_variant_from_model(Path("lineae_3xl.onnx")) == "3XL"
    assert infer_variant_from_model(Path("custom.onnx")) is None
    assert resolve_variant(None, Path("lineae_xl.onnx")) == "XL"
    assert resolve_variant("A", Path("custom.onnx")) == "A"
    assert resolve_variant("N", Path("lineae_xl.onnx")) == "N"
    assert "T" in LINEA_VARIANTS
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


@pytest.mark.parametrize(
    ("durations", "expected_recording_fps", "expected_processing_fps"),
    [
        ([0.075] * 10, 1.0 / 0.075, 1.0 / 0.075),
        ([0.020] * 10, 30.0, 50.0),
        ([1.0] + [0.075] * 9, 1.0 / 0.075, 1.0 / 0.075),
        ([0.075], 30.0, None),
    ],
)
def test_camera_recording_fps_uses_median_and_does_not_exceed_camera_rate(
    durations,
    expected_recording_fps,
    expected_processing_fps,
):
    recording_fps, processing_fps = select_camera_recording_fps(30.0, durations)

    assert recording_fps == pytest.approx(expected_recording_fps)
    if expected_processing_fps is None:
        assert processing_fps is None
    else:
        assert processing_fps == pytest.approx(expected_processing_fps)


class FakeCapture:
    def __init__(self, frames, fps=30.0):
        self.frames = [frame.copy() for frame in frames]
        self.fps = fps
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        assert property_id == cv2.CAP_PROP_FPS
        return self.fps

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FakeDisplay:
    def show(self, image, delay):
        return False


def fake_model(frame):
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        1.0,
    )


def install_video_fakes(monkeypatch, frames, durations):
    capture = FakeCapture(frames)
    writer = FakeWriter()
    writer_calls = []
    ticks = []
    current_time = 0.0
    for duration in durations:
        ticks.extend((current_time, current_time + duration))
        current_time += duration
    ticks.append(current_time)
    tick_iterator = iter(ticks)

    monkeypatch.setattr(demo_lineae.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(demo_lineae.time, "perf_counter", lambda: next(tick_iterator))
    monkeypatch.setattr(
        demo_lineae,
        "annotate_image",
        lambda frame, *args, **kwargs: (frame.copy(), 0),
    )

    def fake_open_video_writer(path, fps, frame):
        writer_calls.append((path, fps, frame.copy()))
        return writer

    monkeypatch.setattr(demo_lineae, "open_video_writer", fake_open_video_writer)
    return capture, writer, writer_calls


def test_camera_recording_buffers_calibration_frames_without_loss(monkeypatch, tmp_path):
    frames = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(12)]
    capture, writer, writer_calls = install_video_fakes(
        monkeypatch,
        frames,
        [0.075] * 10,
    )

    process_video_source(
        model=fake_model,
        capture_source=0,
        output_path=tmp_path / "camera.mp4",
        threshold=0.3,
        max_lines=100,
        display=FakeDisplay(),
        save_result=True,
        adjust_recording_fps=True,
    )

    assert len(writer_calls) == 1
    assert writer_calls[0][1] == pytest.approx(1.0 / 0.075)
    assert [int(frame[0, 0, 0]) for frame in writer.frames] == list(range(12))
    assert capture.released
    assert writer.released


def test_short_camera_recording_flushes_available_frames(monkeypatch, tmp_path):
    frames = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(3)]
    _, writer, writer_calls = install_video_fakes(
        monkeypatch,
        frames,
        [0.10, 0.08, 0.09],
    )

    process_video_source(
        model=fake_model,
        capture_source=0,
        output_path=tmp_path / "short-camera.mp4",
        threshold=0.3,
        max_lines=100,
        display=FakeDisplay(),
        save_result=True,
        adjust_recording_fps=True,
    )

    assert writer_calls[0][1] == pytest.approx(1.0 / 0.09)
    assert [int(frame[0, 0, 0]) for frame in writer.frames] == [0, 1, 2]


def test_prerecorded_video_keeps_source_fps_without_calibration(monkeypatch, tmp_path):
    frames = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(2)]
    capture = FakeCapture(frames, fps=24.0)
    writer = FakeWriter()
    writer_calls = []

    monkeypatch.setattr(demo_lineae.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(
        demo_lineae,
        "annotate_image",
        lambda frame, *args, **kwargs: (frame.copy(), 0),
    )
    monkeypatch.setattr(
        demo_lineae.time,
        "perf_counter",
        lambda: pytest.fail("video-file recording must not calibrate FPS"),
    )

    def fake_open_video_writer(path, fps, frame):
        writer_calls.append((path, fps, frame.copy()))
        return writer

    monkeypatch.setattr(demo_lineae, "open_video_writer", fake_open_video_writer)

    process_video_source(
        model=fake_model,
        capture_source="input.mp4",
        output_path=tmp_path / "video.mp4",
        threshold=0.3,
        max_lines=100,
        display=FakeDisplay(),
        save_result=True,
    )

    assert writer_calls[0][1] == 24.0
    assert [int(frame[0, 0, 0]) for frame in writer.frames] == [0, 1]


def test_disabled_camera_save_does_not_calibrate_or_open_writer(monkeypatch, tmp_path):
    capture = FakeCapture([np.zeros((2, 3, 3), dtype=np.uint8)])
    monkeypatch.setattr(demo_lineae.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(
        demo_lineae,
        "annotate_image",
        lambda frame, *args, **kwargs: (frame.copy(), 0),
    )
    monkeypatch.setattr(
        demo_lineae.time,
        "perf_counter",
        lambda: pytest.fail("disabled recording must not calibrate FPS"),
    )
    monkeypatch.setattr(
        demo_lineae,
        "open_video_writer",
        lambda *args, **kwargs: pytest.fail("disabled recording must not open a writer"),
    )

    process_video_source(
        model=fake_model,
        capture_source=0,
        output_path=tmp_path / "unused.mp4",
        threshold=0.3,
        max_lines=100,
        display=FakeDisplay(),
        save_result=False,
        adjust_recording_fps=True,
    )

    assert capture.released
