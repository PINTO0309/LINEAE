"""OpenCV-only image preprocessing shared by training and deployment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGE_PREPROCESS_SCHEMA = "opencv_rgb_inter_linear_v2"


def validate_image_preprocess_schema(value: str) -> None:
    if value != IMAGE_PREPROCESS_SCHEMA:
        raise ValueError(
            "unsupported image preprocessing schema "
            f"{value!r}; expected {IMAGE_PREPROCESS_SCHEMA!r}"
        )


def checkpoint_image_preprocess_schema(checkpoint: Mapping) -> str | None:
    """Return the schema carried by a detector checkpoint or teacher artifact."""
    schema = checkpoint.get("image_preprocess_schema")
    if schema is not None:
        return schema
    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        return config.get("image_preprocess_schema")
    return None


def validate_checkpoint_image_preprocess(checkpoint: Mapping) -> None:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("detector checkpoint must be a mapping with preprocessing metadata")
    schema = checkpoint_image_preprocess_schema(checkpoint)
    if schema is None:
        raise ValueError(
            "checkpoint predates the OpenCV image preprocessing schema; "
            "start from backbone initialization weights and train a new detector"
        )
    validate_image_preprocess_schema(schema)


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Decode one image as contiguous RGB/HWC/uint8 without EXIF rotation."""
    path = Path(path)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _validate_rgb_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have RGB/HWC shape, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"image must use uint8 pixels, got {image.dtype}")


def _validate_size_hw(size_hw: Sequence[int]) -> tuple[int, int]:
    if not isinstance(size_hw, (list, tuple)) or len(size_hw) != 2:
        raise ValueError("size_hw must contain [height, width]")
    height, width = size_hw
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (height, width)):
        raise ValueError("size_hw must contain two positive integers")
    return height, width


def resize_rgb_image(image: np.ndarray, size_hw: Sequence[int]) -> np.ndarray:
    """Resize RGB/HWC/uint8 pixels with standard OpenCV INTER_LINEAR."""
    _validate_rgb_image(image)
    height, width = _validate_size_hw(size_hw)
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized)


def rgb_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert RGB/HWC/uint8 to RGB/CHW/float32 in the [0, 1] interval."""
    _validate_rgb_image(image)
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(chw).to(dtype=torch.float32).div_(255.0)


def rgb_image_to_normalized_tensor(
    image: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    tensor = rgb_image_to_tensor(image)
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must each contain three RGB values")
    if any(float(value) <= 0.0 for value in std):
        raise ValueError("std values must be positive")
    mean_tensor = tensor.new_tensor(mean).view(3, 1, 1)
    std_tensor = tensor.new_tensor(std).view(3, 1, 1)
    return tensor.sub_(mean_tensor).div_(std_tensor)


def preprocess_image_file(
    path: str | Path,
    size_hw: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    image = resize_rgb_image(read_rgb_image(path), size_hw)
    return rgb_image_to_normalized_tensor(image, mean, std)


__all__ = [
    "IMAGE_PREPROCESS_SCHEMA",
    "checkpoint_image_preprocess_schema",
    "preprocess_image_file",
    "read_rgb_image",
    "resize_rgb_image",
    "rgb_image_to_normalized_tensor",
    "rgb_image_to_tensor",
    "validate_checkpoint_image_preprocess",
    "validate_image_preprocess_schema",
]
