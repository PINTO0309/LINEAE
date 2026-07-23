"""Deterministic best-epoch validation rendering and retention."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch


_BEST_EPOCH_PATTERN = re.compile(r"best_epoch_(\d+)")
_HEADER_HEIGHT = 32
# OpenCV drawing colors use BGR while the documented display colors remain
# GT RGB(64, 255, 96) and prediction RGB(255, 64, 64).
_GT_COLOR = (96, 255, 64)
_PREDICTION_COLOR = (64, 64, 255)


def validate_validation_render_options(args) -> None:
    count = int(getattr(args, "validation_render_count", 0))
    keep_best = int(getattr(args, "validation_render_keep_best", 10))
    threshold = float(getattr(args, "validation_render_score_threshold", 0.3))
    max_predictions = int(getattr(args, "validation_render_max_predictions", 100))
    if count < 0:
        raise ValueError("validation_render_count must be non-negative")
    if count == 0:
        return
    if keep_best <= 0:
        raise ValueError("validation_render_keep_best must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("validation_render_score_threshold must be in [0, 1]")
    if max_predictions <= 0:
        raise ValueError("validation_render_max_predictions must be positive")
    num_select = int(getattr(args, "num_select", max_predictions))
    if max_predictions > num_select:
        raise ValueError("validation_render_max_predictions cannot exceed num_select")


def should_render_best_validation(
    *, is_best: bool, epoch_complete: bool, count: int
) -> bool:
    return bool(is_best and epoch_complete and int(count) > 0)


def filter_render_predictions(
    lines: torch.Tensor,
    scores: torch.Tensor,
    *,
    score_threshold: float,
    max_predictions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return score-sorted predictions that satisfy the render contract."""
    order = torch.argsort(scores, descending=True)
    order = order[scores[order] >= score_threshold][:max_predictions]
    return lines[order], scores[order]


def prune_best_validation_renders(root: Path, keep_best: int) -> None:
    """Keep only the newest recognized best-epoch directories."""
    if keep_best <= 0:
        raise ValueError("keep_best must be positive")
    if not root.is_dir():
        return
    recognized = []
    for path in root.iterdir():
        match = _BEST_EPOCH_PATTERN.fullmatch(path.name)
        if match is not None and path.is_dir() and not path.is_symlink():
            recognized.append((int(match.group(1)), path))
    recognized.sort(key=lambda item: item[0])
    for _, path in recognized[:-keep_best]:
        shutil.rmtree(path)


def _tensor_to_image(
    tensor: torch.Tensor,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> np.ndarray:
    mean = tensor.new_tensor(image_mean).view(-1, 1, 1)
    std = tensor.new_tensor(image_std).view(-1, 1, 1)
    pixels = (tensor.detach().cpu() * std + mean).clamp(0.0, 1.0)
    pixels = pixels.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
    return np.ascontiguousarray(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))


def _line_coordinates(line: Sequence[float], width: int, height: int):
    x0, y0, x1, y1 = (float(value) for value in line)
    return (
        int(round(max(0.0, min(x0, width - 1.0)))),
        int(round(max(0.0, min(y0, height - 1.0)))),
        int(round(max(0.0, min(x1, width - 1.0)))),
        int(round(max(0.0, min(y1, height - 1.0)))),
    )


def _render_comparison(
    image: torch.Tensor,
    target: dict,
    prediction: dict,
    *,
    image_mean: Sequence[float],
    image_std: Sequence[float],
    score_threshold: float,
    max_predictions: int,
) -> np.ndarray:
    base = _tensor_to_image(image, image_mean, image_std)
    height, width = base.shape[:2]
    canvas = np.full((height + _HEADER_HEIGHT, width * 2, 3), 255, dtype=np.uint8)
    canvas[_HEADER_HEIGHT:, :width] = base
    canvas[_HEADER_HEIGHT:, width:] = base
    line_width = max(1, round(min(width, height) / 160))

    scale = target["lines"].new_tensor([width, height, width, height])
    ground_truth = (target["lines"] * scale).detach().cpu()
    for line in ground_truth:
        coordinates = _line_coordinates(line.tolist(), width, height)
        cv2.line(
            canvas,
            (coordinates[0], coordinates[1] + _HEADER_HEIGHT),
            (coordinates[2], coordinates[3] + _HEADER_HEIGHT),
            _GT_COLOR,
            thickness=line_width,
            lineType=cv2.LINE_8,
        )

    predicted_lines, predicted_scores = filter_render_predictions(
        prediction["lines"].detach().cpu(),
        prediction["scores"].detach().cpu(),
        score_threshold=score_threshold,
        max_predictions=max_predictions,
    )
    for line in predicted_lines:
        coordinates = _line_coordinates(line.tolist(), width, height)
        cv2.line(
            canvas,
            (coordinates[0] + width, coordinates[1] + _HEADER_HEIGHT),
            (coordinates[2] + width, coordinates[3] + _HEADER_HEIGHT),
            _PREDICTION_COLOR,
            thickness=line_width,
            lineType=cv2.LINE_8,
        )

    cv2.putText(
        canvas,
        f"GT (green): {len(ground_truth)}",
        (6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 96, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Prediction (red): {len(predicted_lines)} at score >= {score_threshold:.2f}",
        (width + 6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 128),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (width, 0),
        (width, height + _HEADER_HEIGHT - 1),
        (192, 192, 192),
        thickness=1,
        lineType=cv2.LINE_8,
    )
    return canvas


def _replace_render_directory(
    temporary_dir: Path,
    final_dir: Path,
    *,
    replace_existing: bool,
) -> None:
    if not final_dir.exists():
        temporary_dir.replace(final_dir)
        return
    if not replace_existing:
        raise FileExistsError(f"render output directory already exists: {final_dir}")
    if not final_dir.is_dir() or final_dir.is_symlink():
        raise FileExistsError(
            "refusing to replace a render output path that is not a real "
            f"directory: {final_dir}"
        )

    backup_dir = final_dir.with_name(f".{final_dir.name}.backup-{uuid.uuid4().hex}")
    final_dir.replace(backup_dir)
    try:
        temporary_dir.replace(final_dir)
    except Exception:
        backup_dir.replace(final_dir)
        raise
    shutil.rmtree(backup_dir)


@torch.no_grad()
def _save_validation_renders(
    *,
    model: torch.nn.Module,
    postprocessor,
    dataset,
    device: torch.device,
    final_dir: Path,
    image_mean: Sequence[float],
    image_std: Sequence[float],
    count: int,
    score_threshold: float,
    max_predictions: int,
    batch_size: int,
    amp: bool,
    replace_existing: bool,
) -> Path | None:
    """Render a fixed validation prefix atomically into one directory."""
    count = int(count)
    if count == 0:
        return None
    if count < 0:
        raise ValueError("count must be non-negative")
    if len(dataset) < count:
        raise ValueError(
            f"validation dataset has {len(dataset)} samples, fewer than requested {count}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if int(max_predictions) <= 0:
        raise ValueError("max_predictions must be positive")

    final_dir = Path(final_dir)
    if final_dir.exists():
        if not replace_existing:
            raise FileExistsError(
                f"render output directory already exists: {final_dir}"
            )
        if not final_dir.is_dir() or final_dir.is_symlink():
            raise FileExistsError(
                "refusing to replace a render output path that is not a real "
                f"directory: {final_dir}"
            )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = final_dir.with_name(f".{final_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary_dir.mkdir()

    was_training = model.training
    saved_count = 0
    try:
        model.eval()
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            items = [dataset[index] for index in range(start, stop)]
            images = torch.stack([image for image, _ in items])
            targets = [target for _, target in items]
            device_images = images.to(device)
            device_targets = [
                {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            target_sizes = torch.stack([target["size"] for target in device_targets])
            with torch.amp.autocast(str(device), enabled=bool(amp)):
                outputs = model(device_images, device_targets)
            predictions = postprocessor(outputs, target_sizes)

            for image, target, prediction in zip(
                images, targets, predictions, strict=True
            ):
                rendered = _render_comparison(
                    image,
                    target,
                    prediction,
                    image_mean=image_mean,
                    image_std=image_std,
                    score_threshold=score_threshold,
                    max_predictions=max_predictions,
                )
                image_id = int(target["image_id"].flatten()[0].item())
                destination = temporary_dir / f"{saved_count:02d}_image_{image_id}.png"
                if not cv2.imwrite(str(destination), rendered):
                    raise RuntimeError(
                        f"OpenCV could not write validation render: {destination}"
                    )
                saved_count += 1
        if saved_count != count:
            raise RuntimeError(
                f"rendered {saved_count} validation images, expected {count}"
            )
        _replace_render_directory(
            temporary_dir,
            final_dir,
            replace_existing=replace_existing,
        )
        return final_dir
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    finally:
        model.train(was_training)


def save_validation_renders(
    *,
    model: torch.nn.Module,
    postprocessor,
    dataset,
    device: torch.device,
    output_dir: Path,
    image_mean: Sequence[float],
    image_std: Sequence[float],
    count: int,
    score_threshold: float,
    max_predictions: int,
    batch_size: int,
    amp: bool,
) -> Path | None:
    """Render an evaluated checkpoint without replacing an existing directory."""
    return _save_validation_renders(
        model=model,
        postprocessor=postprocessor,
        dataset=dataset,
        device=device,
        final_dir=Path(output_dir),
        image_mean=image_mean,
        image_std=image_std,
        count=count,
        score_threshold=score_threshold,
        max_predictions=max_predictions,
        batch_size=batch_size,
        amp=amp,
        replace_existing=False,
    )


def save_best_validation_renders(
    *,
    model: torch.nn.Module,
    postprocessor,
    dataset,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
    count: int,
    keep_best: int,
    score_threshold: float,
    max_predictions: int,
    batch_size: int,
    amp: bool,
) -> Path | None:
    """Render a completed best epoch atomically, then apply retention."""
    render_root = Path(output_dir) / "validation_renders"
    final_dir = render_root / f"best_epoch_{int(epoch):04d}"
    rendered_dir = _save_validation_renders(
        model=model,
        postprocessor=postprocessor,
        dataset=dataset,
        device=device,
        final_dir=final_dir,
        image_mean=image_mean,
        image_std=image_std,
        count=count,
        score_threshold=score_threshold,
        max_predictions=max_predictions,
        batch_size=batch_size,
        amp=amp,
        replace_existing=True,
    )
    if rendered_dir is not None:
        prune_best_validation_renders(render_root, keep_best)
    return rendered_dir


__all__ = [
    "filter_render_predictions",
    "prune_best_validation_renders",
    "save_best_validation_renders",
    "save_validation_renders",
    "should_render_best_validation",
    "validate_validation_render_options",
]
