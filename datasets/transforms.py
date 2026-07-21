# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Geometry-aware OpenCV transforms for images and line annotations."""

from __future__ import annotations

import numbers
import random
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional

from util.image_preprocess import resize_rgb_image, rgb_image_to_tensor


def _image_size(image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    return width, height


def _resize_tensor_bilinear(tensor: torch.Tensor, size_hw) -> torch.Tensor:
    batched = tensor.unsqueeze(0)
    resized = torch_functional.interpolate(
        batched,
        size=tuple(size_hw),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.squeeze(0)


def _clip_segment_to_rectangle(line, width, height):
    """Liang-Barsky clipping, including horizontal and vertical segments."""
    x0, y0, x1, y1 = (float(value) for value in line)
    dx, dy = x1 - x0, y1 - y0
    lower, upper = 0.0, 1.0
    for direction, distance in (
        (-dx, x0),
        (dx, width - 1 - x0),
        (-dy, y0),
        (dy, height - 1 - y0),
    ):
        if abs(direction) < 1e-12:
            if distance < 0:
                return None
            continue
        ratio = distance / direction
        if direction < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return line.new_tensor([
        x0 + lower * dx,
        y0 + lower * dy,
        x1 - (1.0 - upper) * dx,
        y1 - (1.0 - upper) * dy,
    ])


def crop(image, target, region):
    i, j, height, width = region
    cropped_image = np.ascontiguousarray(image[i : i + height, j : j + width])

    target = target.copy()
    target["size"] = torch.tensor([height, width])
    fields = ["labels", "area", "iscrowd"]

    if "lmap" in target:
        cropped_lmaps = []
        for lmap, downsampling in zip(target["lmap"], [8, 16, 32], strict=True):
            top = i // downsampling
            left = j // downsampling
            bottom = top + height // downsampling
            right = left + width // downsampling
            cropped_lmaps.append(lmap[..., top:bottom, left:right])
        target["lmap"] = cropped_lmaps

    kept_indices = torch.arange(len(target[fields[0]]), dtype=torch.long)
    if "lines" in target:
        lines = target["lines"]
        cropped_lines = lines - torch.as_tensor([j, i, j, i])
        clipped_lines = []
        kept = []
        for index, line in enumerate(cropped_lines):
            clipped = _clip_segment_to_rectangle(line, width, height)
            if clipped is not None and (clipped[:2] - clipped[2:]).norm() > 10:
                clipped_lines.append(clipped)
                kept.append(index)
        target["lines"] = (
            torch.stack(clipped_lines)
            if clipped_lines
            else cropped_lines.new_empty((0, 4))
        )
        kept_indices = torch.as_tensor(
            kept,
            dtype=torch.long,
            device=target[fields[0]].device,
        )

    for field in fields:
        target[field] = target[field][kept_indices]
    return cropped_image, target


def hflip(image, target):
    flipped_image = np.ascontiguousarray(cv2.flip(image, 1))
    width, _ = _image_size(image)

    target = target.copy()
    if "lines" in target:
        lines = target["lines"]
        target["lines"] = (
            lines[:, [2, 3, 0, 1]] * torch.as_tensor([-1, 1, -1, 1])
            + torch.as_tensor([width, 0, width, 0])
        )
    if "lmap" in target:
        target["lmap"] = [torch.flip(lmap, dims=(-1,)) for lmap in target["lmap"]]
    return flipped_image, target


def vflip(image, target):
    flipped_image = np.ascontiguousarray(cv2.flip(image, 0))
    _, height = _image_size(image)

    target = target.copy()
    if "lines" in target:
        lines = target["lines"] * torch.as_tensor([1, -1, 1, -1])
        lines = lines + torch.as_tensor([0, height, 0, height])
        vertical = lines[:, 0] == lines[:, 2]
        lines[vertical] = torch.index_select(
            lines[vertical], 1, torch.tensor([2, 3, 0, 1])
        )
        target["lines"] = lines
    if "lmap" in target:
        target["lmap"] = [torch.flip(lmap, dims=(-2,)) for lmap in target["lmap"]]
    return flipped_image, target


def resize(image, target, size, max_size=None):
    """Resize an RGB image and its target; tuple sizes use (width, height)."""

    def get_size_with_aspect_ratio(image_size, requested, maximum=None):
        width, height = image_size
        if maximum is not None:
            min_original = float(min((width, height)))
            max_original = float(max((width, height)))
            if max_original / min_original * requested > maximum:
                requested = int(round(maximum * min_original / max_original))
        if (width <= height and width == requested) or (
            height <= width and height == requested
        ):
            return height, width
        if width < height:
            output_width = requested
            output_height = int(requested * height / width)
        else:
            output_height = requested
            output_width = int(requested * width / height)
        return output_height, output_width

    def get_size(image_size, requested, maximum=None):
        if isinstance(requested, (list, tuple)):
            return tuple(requested[::-1])
        return get_size_with_aspect_ratio(image_size, requested, maximum)

    original_width, original_height = _image_size(image)
    size_hw = get_size((original_width, original_height), size, max_size)
    rescaled_image = resize_rgb_image(image, size_hw)
    if target is None:
        return rescaled_image, None

    output_height, output_width = rescaled_image.shape[:2]
    ratio_width = output_width / original_width
    ratio_height = output_height / original_height
    target = target.copy()
    if "lines" in target:
        target["lines"] = target["lines"] * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
    if "lmap" in target:
        resized_lmaps = []
        for lmap, downsampling in zip(target["lmap"], [8, 16, 32], strict=True):
            resized_lmaps.append(
                _resize_tensor_bilinear(
                    lmap,
                    (output_height // downsampling, output_width // downsampling),
                )
            )
        target["lmap"] = resized_lmaps
    target["size"] = torch.tensor([output_height, output_width])
    return rescaled_image, target


def pad(image, target, padding):
    pad_width, pad_height = padding
    padded_image = cv2.copyMakeBorder(
        image,
        0,
        pad_height,
        0,
        pad_width,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    padded_image = np.ascontiguousarray(padded_image)
    if target is None:
        return padded_image, None
    target = target.copy()
    height, width = padded_image.shape[:2]
    target["size"] = torch.tensor([height, width])
    if "masks" in target:
        target["masks"] = torch_functional.pad(
            target["masks"], (0, pad_width, 0, pad_height)
        )
    return padded_image, target


def rotation(image, target, rotation_type):
    width, height = _image_size(image)
    if rotation_type == 0:
        rotated_image = image
    elif rotation_type == 1:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_type == 2:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError(f"unsupported rotation type: {rotation_type}")
    rotated_image = np.ascontiguousarray(rotated_image)

    target = target.copy()
    if rotation_type == 1 and "lines" in target:
        lines = target["lines"]
        rotated_lines = lines[..., [1, 0, 3, 2]].clone()
        rotated_lines[..., [1, 3]] = width - 1 - rotated_lines[..., [1, 3]]
        target["lines"] = rotated_lines
    elif rotation_type == 2 and "lines" in target:
        lines = target["lines"]
        rotated_lines = lines[..., [1, 0, 3, 2]].clone()
        rotated_lines[..., [0, 2]] = height - 1 - rotated_lines[..., [0, 2]]
        target["lines"] = rotated_lines

    if "lmap" in target and rotation_type:
        turns = 1 if rotation_type == 1 else -1
        target["lmap"] = [torch.rot90(lmap, turns, (-2, -1)) for lmap in target["lmap"]]
    output_height, output_width = rotated_image.shape[:2]
    target["size"] = torch.tensor([output_height, output_width])
    return rotated_image, target


def _random_crop_params(image, output_size):
    height, width = image.shape[:2]
    target_height, target_width = output_size
    if height < target_height or width < target_width:
        raise ValueError(
            f"crop size {(target_height, target_width)} exceeds image {(height, width)}"
        )
    top = int(torch.randint(0, height - target_height + 1, size=(1,)).item())
    left = int(torch.randint(0, width - target_width + 1, size=(1,)).item())
    return top, left, target_height, target_width


class Rotation:
    def __init__(self):
        self.rotation_type = [0, 0, 0, 1, 2]

    def __call__(self, img, target):
        return rotation(img, target, int(np.random.choice(self.rotation_type)))


class ResizeDebug:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return crop(img, target, _random_crop_params(img, self.size))


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: np.ndarray, target: dict):
        width, height = _image_size(img)
        crop_width = random.randint(self.min_size, min(width, self.max_size))
        crop_height = random.randint(self.min_size, min(height, self.max_size))
        region = _random_crop_params(img, [crop_height, crop_width])
        return crop(img, target, region)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = _image_size(img)
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)
        return img, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        if not isinstance(sizes, (list, tuple)):
            raise TypeError("sizes must be a list or tuple")
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        return resize(img, target, random.choice(self.sizes), self.max_size)


class RandomPad:
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect:
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    def __call__(self, img, target):
        return rgb_image_to_tensor(img), target


class RandomErasing:
    def __init__(
        self,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=0,
        inplace=False,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError("RandomErasing probability must be in [0, 1]")
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, img, target):
        if not torch.is_tensor(img) or img.ndim != 3:
            raise TypeError("RandomErasing expects a CHW tensor")
        if torch.rand(1).item() >= self.p:
            return img, target
        channels, height, width = img.shape
        area = height * width
        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(*self.scale).item()
            aspect = torch.exp(torch.empty(1).uniform_(*log_ratio)).item()
            erase_height = int(round(math.sqrt(erase_area * aspect)))
            erase_width = int(round(math.sqrt(erase_area / aspect)))
            if erase_height < height and erase_width < width:
                top = int(torch.randint(0, height - erase_height + 1, (1,)).item())
                left = int(torch.randint(0, width - erase_width + 1, (1,)).item())
                output = img if self.inplace else img.clone()
                if self.value == "random":
                    fill = torch.empty(
                        (channels, erase_height, erase_width),
                        dtype=img.dtype,
                        device=img.device,
                    ).normal_()
                else:
                    fill = img.new_tensor(self.value)
                    if fill.ndim == 1:
                        fill = fill[:, None, None]
                output[:, top : top + erase_height, left : left + erase_width] = fill
                return output, target
        return img, target


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    return _clip_uint8(image.astype(np.float32) * factor)


def _adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    gray_mean = float(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean())
    adjusted = image.astype(np.float32) * factor + gray_mean * (1.0 - factor)
    return _clip_uint8(adjusted)


def _adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    adjusted = image.astype(np.float32) * factor + gray_rgb * (1.0 - factor)
    return _clip_uint8(adjusted)


def _adjust_hue(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16)
    hue = (hue + int(round(factor * 180.0))) % 180
    hsv = hsv.copy()
    hsv[..., 0] = hue.astype(np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))


def _sample_factor(interval) -> float:
    return torch.empty(1).uniform_(interval[0], interval[1]).item()


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue,
            "hue",
            center=0,
            bound=(-0.5, 0.5),
            clip_first_on_zero=False,
        )

    @staticmethod
    def _check_input(
        value,
        name,
        center=1,
        bound=(0, float("inf")),
        clip_first_on_zero=True,
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values must be between {bound}")
        else:
            raise TypeError(f"{name} must be a number or a two-element sequence")
        return None if value[0] == value[1] == center else value

    def __call__(self, img, target):
        for fn_id in torch.randperm(4).tolist():
            if fn_id == 0 and self.brightness is not None:
                img = _adjust_brightness(img, _sample_factor(self.brightness))
            elif fn_id == 1 and self.contrast is not None:
                img = _adjust_contrast(img, _sample_factor(self.contrast))
            elif fn_id == 2 and self.saturation is not None:
                img = _adjust_saturation(img, _sample_factor(self.saturation))
            elif fn_id == 3 and self.hue is not None:
                img = _adjust_hue(img, _sample_factor(self.hue))
        return img, target


class RandomPhotometricDistort:
    """OpenCV implementation of the Gazelle/SSD photometric distortion."""

    def __init__(
        self,
        brightness=(0.875, 1.125),
        contrast=(0.5, 1.5),
        saturation=(0.5, 1.5),
        hue=(-0.05, 0.05),
        p=0.5,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError("photometric distortion probability must be in [0, 1]")
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def _maybe_factor(self, interval):
        return _sample_factor(interval) if torch.rand(1).item() < self.p else None

    def __call__(self, img, target):
        brightness = self._maybe_factor(self.brightness)
        contrast = self._maybe_factor(self.contrast)
        saturation = self._maybe_factor(self.saturation)
        hue = self._maybe_factor(self.hue)
        contrast_before = torch.rand(()).item() < 0.5
        channel_permutation = (
            torch.randperm(3).tolist() if torch.rand(1).item() < self.p else None
        )

        if brightness is not None:
            img = _adjust_brightness(img, brightness)
        if contrast is not None and contrast_before:
            img = _adjust_contrast(img, contrast)
        if saturation is not None:
            img = _adjust_saturation(img, saturation)
        if hue is not None:
            img = _adjust_hue(img, hue)
        if contrast is not None and not contrast_before:
            img = _adjust_contrast(img, contrast)
        if channel_permutation is not None:
            img = np.ascontiguousarray(img[..., channel_permutation])
        return img, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        mean = image.new_tensor(self.mean).view(3, 1, 1)
        std = image.new_tensor(self.std).view(3, 1, 1)
        image = (image - mean) / std
        if target is None:
            return image, None
        target = target.copy()
        height, width = image.shape[-2:]
        if "lines" in target:
            lines = target["lines"] / torch.tensor(
                [width, height, width, height],
                dtype=torch.float32,
                device=target["lines"].device,
            )
            lines = lines.clamp(0.0, 1.0)
            swap = torch.logical_or(
                lines[..., 0] > lines[..., 2],
                torch.logical_or(
                    lines[..., 0] == lines[..., 2], lines[..., 1] < lines[..., 3]
                ),
            )
            lines[swap] = lines[swap][:, [2, 3, 0, 1]]
            target["lines"] = lines
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        lines.extend(f"    {transform}" for transform in self.transforms)
        lines.append(")")
        return "\n".join(lines)
