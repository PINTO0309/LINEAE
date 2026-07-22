#!/usr/bin/env python

"""Run a structurally validated LINEAE ONNX model on images or video."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "onnx" / "lineae_xl.onnx"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "demo_lineae"
DEFAULT_SCORE_THRESHOLD = 0.4
DEFAULT_MAX_LINES = 100
DEFAULT_VIDEO_FPS = 30.0
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VARIANTS = ("A", "F", "P", "N", "T", "S", "M", "L", "X", "XL", "2XL", "3XL")
LINEA_VARIANTS = frozenset(("A", "F", "P", "N", "T"))
PREPROCESS_PROFILES = {
    "linea": (
        np.asarray([0.538, 0.494, 0.453], dtype=np.float32),
        np.asarray([0.257, 0.263, 0.273], dtype=np.float32),
    ),
    "imagenet": (
        np.asarray([0.485, 0.456, 0.406], dtype=np.float32),
        np.asarray([0.229, 0.224, 0.225], dtype=np.float32),
    ),
}


@dataclass(frozen=True)
class InputSource:
    kind: str
    path: Path | None = None
    camera_index: int | None = None


class DisplayWindow:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        if (
            self.enabled
            and sys.platform.startswith("linux")
            and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        ):
            warnings.warn(
                "No graphical display was detected; continuing with display disabled.",
                stacklevel=2,
            )
            self.enabled = False

    def show(self, image: np.ndarray, delay: int) -> bool:
        if not self.enabled:
            return False
        try:
            cv2.imshow("LINEAE", image)
            key = cv2.waitKey(delay) & 0xFF
        except cv2.error as error:
            warnings.warn(
                f"OpenCV display is unavailable; continuing without a window: {error}",
                stacklevel=2,
            )
            self.enabled = False
            return False
        return key in (ord("q"), 27)

    def close(self) -> None:
        if self.enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


class LineaeOnnxModel:
    def __init__(
        self,
        model_path: Path,
        variant: str | None,
        execution_provider: str,
    ) -> None:
        self.model_path = resolve_existing_file(model_path, "ONNX model")
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(f"model must have an .onnx extension: {self.model_path}")

        self.variant = resolve_variant(variant, self.model_path)
        profile = "linea" if self.variant in LINEA_VARIANTS else "imagenet"
        self.mean, self.std = PREPROCESS_PROFILES[profile]

        session_options, providers = build_providers(
            execution_provider,
            self.model_path,
        )
        print("Requested ONNX Runtime providers:")
        pprint(providers)

        ort.set_default_logger_severity(3)
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers,
        )
        self.enabled_providers = self.session.get_providers()
        print("Enabled ONNX Runtime providers:")
        pprint(self.enabled_providers)

        required_provider = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "tensorrt": "TensorrtExecutionProvider",
        }[execution_provider]
        if required_provider not in self.enabled_providers:
            raise RuntimeError(
                f"{required_provider} was requested but did not activate; "
                f"active providers: {self.enabled_providers}"
            )

        self.input_height, self.input_width, self.num_select = (
            self._validate_model_schema()
        )
        print(
            f"LINEAE variant {self.variant}: input "
            f"{self.input_width}x{self.input_height}, top-k {self.num_select}"
        )

    def _validate_model_schema(self) -> tuple[int, int, int]:
        inputs = {item.name: item for item in self.session.get_inputs()}
        outputs = {item.name: item for item in self.session.get_outputs()}
        if set(inputs) != {"images"}:
            raise ValueError(
                "LINEAE model input must be 'images'; "
                f"got: {sorted(inputs)}"
            )
        if set(outputs) != {"pred_logits", "pred_lines"}:
            raise ValueError(
                "LINEAE model outputs must be 'pred_logits' and 'pred_lines'; "
                f"got: {sorted(outputs)}"
            )

        images = inputs["images"]
        logits = outputs["pred_logits"]
        lines = outputs["pred_lines"]
        require_type(images, "tensor(float)")
        require_type(logits, "tensor(float)")
        require_type(lines, "tensor(float)")
        require_rank(images, 4)
        require_rank(logits, 3)
        require_rank(lines, 3)
        if images.shape[0] != 1 or images.shape[1] != 3:
            raise ValueError(f"images must have fixed shape [1,3,H,W]; got: {images.shape}")
        if logits.shape[0] != 1 or logits.shape[2] != 2:
            raise ValueError(
                "pred_logits must have fixed shape [1,K,2]; "
                f"got: {logits.shape}"
            )
        if lines.shape[0] != 1 or lines.shape[2] != 4:
            raise ValueError(
                "pred_lines must have fixed shape [1,K,4]; "
                f"got: {lines.shape}"
            )
        input_height = require_static_positive_dimension(
            images.shape[2], "image height"
        )
        input_width = require_static_positive_dimension(images.shape[3], "image width")
        num_select = require_static_positive_dimension(logits.shape[1], "output top-k")
        if lines.shape[1] != num_select:
            raise ValueError(
                "pred_logits and pred_lines must have the same fixed top-k dimension; "
                f"got: {logits.shape} and {lines.shape}"
            )
        return input_height, input_width, num_select

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        input_image = preprocess_bgr_image(
            image,
            size_hw=(self.input_height, self.input_width),
            mean=self.mean,
            std=self.std,
        )
        original_height, original_width = image.shape[:2]

        started_at = time.perf_counter()
        logits, normalized_lines = self.session.run(
            ["pred_logits", "pred_lines"],
            {"images": input_image},
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        if logits.shape != (1, self.num_select, 2):
            raise RuntimeError(f"unexpected pred_logits output shape: {logits.shape}")
        if normalized_lines.shape != (1, self.num_select, 4):
            raise RuntimeError(
                f"unexpected pred_lines output shape: {normalized_lines.shape}"
            )
        if not np.isfinite(logits).all() or not np.isfinite(normalized_lines).all():
            raise RuntimeError("LINEAE produced non-finite output values")

        scores = sigmoid(logits[0, :, 0])
        scale = np.asarray(
            [original_width, original_height, original_width, original_height],
            dtype=np.float32,
        )
        lines = normalized_lines[0] * scale
        return lines, scores, elapsed_ms


def infer_variant_from_model(model_path: Path) -> str | None:
    match = re.search(
        r"(?:^|_)lineae_(2xl|3xl|xl|[afpntsmlx])(?:_|$)", model_path.stem.lower()
    )
    if match is None:
        return None
    return match.group(1).upper()


def resolve_variant(requested: str | None, model_path: Path) -> str:
    if requested is not None:
        return requested.upper()
    inferred = infer_variant_from_model(model_path)
    if inferred is None:
        raise ValueError(
            "could not infer the LINEAE variant from the ONNX filename; "
            "specify --variant"
        )
    return inferred


def preprocess_bgr_image(
    image: np.ndarray,
    *,
    size_hw: tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("input image is empty")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"input image must be BGR/HWC/uint8, got shape={image.shape}, "
            f"dtype={image.dtype}"
        )
    height, width = size_hw
    if height <= 0 or width <= 0:
        raise ValueError("model input size must be positive")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb_image,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    pixels = resized.astype(np.float32) / np.float32(255.0)
    pixels = (pixels - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def score_threshold(value: str) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("score threshold must be in the range [0, 1]")
    return threshold


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Image, image directory, video, or camera index.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"LINEAE ONNX model path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--variant",
        type=str.upper,
        choices=VARIANTS,
        help="LINEAE variant; inferred from the ONNX filename when omitted.",
    )
    parser.add_argument(
        "--execution-provider",
        "--execution_provider",
        "-ep",
        dest="execution_provider",
        choices=("cpu", "cuda", "tensorrt"),
        default="cuda",
        help="ONNX Runtime execution provider (default: cuda).",
    )
    parser.add_argument(
        "--score-threshold",
        "-t",
        type=score_threshold,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Class-0 score threshold (default: {DEFAULT_SCORE_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-lines",
        type=positive_integer,
        default=DEFAULT_MAX_LINES,
        help=f"Maximum rendered lines per frame (default: {DEFAULT_MAX_LINES}).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Result directory (default: {DEFAULT_OUTPUT_DIRECTORY}).",
    )
    parser.add_argument(
        "--disable-display",
        action="store_true",
        help="Disable the OpenCV result window.",
    )
    parser.add_argument(
        "--disable-save",
        action="store_true",
        help="Disable result image and video writing.",
    )
    parser.add_argument(
        "--disable-wait-key",
        action="store_true",
        help="Do not wait for a key press between still images.",
    )
    return parser.parse_args()


def resolve_existing_file(path: Path, label: str) -> Path:
    resolved_path = path.expanduser()
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path
    resolved_path = resolved_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved_path}")
    return resolved_path


def resolve_output_directory(path: Path) -> Path:
    resolved_path = path.expanduser()
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path
    return resolved_path.resolve()


def resolve_input_source(value: str) -> InputSource:
    candidate = Path(value).expanduser()
    if candidate.exists():
        candidate = candidate.resolve()
        if candidate.is_dir():
            return InputSource(kind="image_directory", path=candidate)
        if candidate.is_file():
            kind = "image" if candidate.suffix.lower() in IMAGE_SUFFIXES else "video"
            return InputSource(kind=kind, path=candidate)
        raise ValueError(f"unsupported input path: {candidate}")

    try:
        camera_index = int(value)
    except ValueError as error:
        raise FileNotFoundError(f"input does not exist: {candidate.resolve()}") from error
    if camera_index < 0:
        raise ValueError(f"camera index must be non-negative: {camera_index}")
    return InputSource(kind="camera", camera_index=camera_index)


def build_providers(
    execution_provider: str,
    model_path: Path,
) -> tuple[ort.SessionOptions, list[Any]]:
    available = ort.get_available_providers()
    required = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }[execution_provider]
    if required not in available:
        raise RuntimeError(
            f"{required} is unavailable; available providers: {available}"
        )

    options = ort.SessionOptions()
    if execution_provider == "cpu":
        return options, ["CPUExecutionProvider"]

    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    cuda = ("CUDAExecutionProvider", {"use_tf32": "0"})
    if execution_provider == "cuda":
        return options, [cuda]

    if "CUDAExecutionProvider" not in available:
        raise RuntimeError("TensorRT execution requires CUDAExecutionProvider fallback")
    tensorrt = (
        "TensorrtExecutionProvider",
        {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(model_path.parent),
            "trt_fp16_enable": True,
            "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
        },
    )
    return options, [tensorrt, cuda]


def require_type(node: Any, expected_type: str) -> None:
    if node.type != expected_type:
        raise ValueError(f"{node.name} must be {expected_type}; got: {node.type}")


def require_rank(node: Any, expected_rank: int) -> None:
    if len(node.shape) != expected_rank:
        raise ValueError(
            f"{node.name} must have rank {expected_rank}; got: {node.shape}"
        )


def require_static_positive_dimension(value: Any, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a static positive integer; got: {value!r}")
    return value


def annotate_image(
    image: np.ndarray,
    lines: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    max_lines: int,
    elapsed_ms: float | None = None,
) -> tuple[np.ndarray, int]:
    result = image.copy()
    height, width = result.shape[:2]
    order = np.argsort(-scores, kind="stable")
    selected_indices = order[scores[order] >= threshold][:max_lines]

    for index in selected_indices:
        line = lines[index]
        score = scores[index]
        x1, y1, x2, y2 = np.rint(line).astype(np.int64)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        point1 = int(x1), int(y1)
        point2 = int(x2), int(y2)
        cv2.line(result, point1, point2, (0, 0, 255), 2, cv2.LINE_AA)
        label_origin = point1[0], max(point1[1] - 4, 12)
        label = f"{float(score):.2f}"
        cv2.putText(
            result,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if elapsed_ms is not None:
        latency_text = f"{elapsed_ms:.2f} ms"
        cv2.putText(
            result,
            latency_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            latency_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return result, len(selected_indices)


def read_image(path: Path) -> np.ndarray:
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")
    print(f"Saved: {path}")


def result_image_path(input_path: Path, output_directory: Path) -> Path:
    suffix = input_path.suffix.lower()
    return output_directory / f"{input_path.stem}_result{suffix}"


def process_still_image(
    model: LineaeOnnxModel,
    input_path: Path,
    output_path: Path,
    threshold: float,
    max_lines: int,
    display: DisplayWindow,
    save_result: bool,
    disable_wait_key: bool,
) -> bool:
    image = read_image(input_path)
    lines, scores, elapsed_ms = model(image)
    result, count = annotate_image(
        image,
        lines,
        scores,
        threshold,
        max_lines,
        elapsed_ms=elapsed_ms,
    )
    print(f"{input_path}: {elapsed_ms:.2f} ms, {count} lines")
    if save_result:
        write_image(output_path, result)
    delay = 1 if disable_wait_key else 0
    return display.show(result, delay)


def process_image_directory(
    model: LineaeOnnxModel,
    input_directory: Path,
    output_directory: Path,
    threshold: float,
    max_lines: int,
    display: DisplayWindow,
    save_result: bool,
    disable_wait_key: bool,
) -> None:
    image_paths = sorted(
        path
        for path in input_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"no supported images found in: {input_directory}")

    result_root = output_directory / input_directory.name
    for input_path in image_paths:
        relative_path = input_path.relative_to(input_directory)
        output_path = result_root / relative_path.parent / (
            f"{relative_path.stem}_result{relative_path.suffix.lower()}"
        )
        should_stop = process_still_image(
            model=model,
            input_path=input_path,
            output_path=output_path,
            threshold=threshold,
            max_lines=max_lines,
            display=display,
            save_result=save_result,
            disable_wait_key=disable_wait_key,
        )
        if should_stop:
            break


def open_video_writer(path: Path, fps: float, frame: np.ndarray) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def process_video_source(
    model: LineaeOnnxModel,
    capture_source: str | int,
    output_path: Path,
    threshold: float,
    max_lines: int,
    display: DisplayWindow,
    save_result: bool,
) -> None:
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open video or camera: {capture_source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = DEFAULT_VIDEO_FPS

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_count += 1
            lines, scores, elapsed_ms = model(frame)
            result, count = annotate_image(
                frame,
                lines,
                scores,
                threshold,
                max_lines,
                elapsed_ms=elapsed_ms,
            )

            if save_result:
                if writer is None:
                    writer = open_video_writer(output_path, fps, result)
                writer.write(result)
            if frame_count % 30 == 0:
                print(
                    f"Processed {frame_count} frames: {elapsed_ms:.2f} ms, "
                    f"{count} lines"
                )
            if display.show(result, 1):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError(f"no frames were read from: {capture_source}")
    print(f"Processed {frame_count} frames.")
    if save_result:
        print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    output_directory = resolve_output_directory(args.output_dir)
    source = resolve_input_source(args.input)
    model = LineaeOnnxModel(
        args.model,
        args.variant,
        args.execution_provider,
    )
    display = DisplayWindow(enabled=not args.disable_display)
    save_result = not args.disable_save

    try:
        if source.kind == "image":
            assert source.path is not None
            process_still_image(
                model=model,
                input_path=source.path,
                output_path=result_image_path(source.path, output_directory),
                threshold=args.score_threshold,
                max_lines=args.max_lines,
                display=display,
                save_result=save_result,
                disable_wait_key=args.disable_wait_key,
            )
        elif source.kind == "image_directory":
            assert source.path is not None
            process_image_directory(
                model=model,
                input_directory=source.path,
                output_directory=output_directory,
                threshold=args.score_threshold,
                max_lines=args.max_lines,
                display=display,
                save_result=save_result,
                disable_wait_key=args.disable_wait_key,
            )
        elif source.kind == "video":
            assert source.path is not None
            process_video_source(
                model=model,
                capture_source=str(source.path),
                output_path=output_directory / f"{source.path.stem}_result.mp4",
                threshold=args.score_threshold,
                max_lines=args.max_lines,
                display=display,
                save_result=save_result,
            )
        else:
            assert source.camera_index is not None
            process_video_source(
                model=model,
                capture_source=source.camera_index,
                output_path=(
                    output_directory / f"camera_{source.camera_index}_result.mp4"
                ),
                threshold=args.score_threshold,
                max_lines=args.max_lines,
                display=display,
                save_result=save_result,
            )
    finally:
        display.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from error
