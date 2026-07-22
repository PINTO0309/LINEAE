#!/usr/bin/env python

"""Render COCO line annotations over their original images with OpenCV."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_FILE = (
    DATA_ROOT / "wireframe_processed" / "annotations" / "lines_val2017.json"
)
DEFAULT_IMAGE_DIR = DATA_ROOT / "wireframe_processed" / "val2017"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "test_render"

# OpenCV uses BGR. This matches the green GT overlay used by validation renders.
ANNOTATION_COLOR = (96, 255, 64)


@dataclass(frozen=True)
class RenderSummary:
    annotation_file: Path
    output_dir: Path
    image_count: int
    line_count: int


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"annotation file is not valid JSON: {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("annotation JSON root must be an object")
    for field in ("images", "annotations"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"annotation JSON field {field!r} must be a list")
    return payload


def _positive_integer(value: Any, field: str, *, image_id: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"image {image_id} field {field!r} must be a positive integer"
        )
    return value


def _validate_file_name(value: Any, *, image_id: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"image {image_id} field 'file_name' must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"image {image_id} has unsafe file_name: {value!r}")
    return value


def _parse_image_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    seen_ids: set[int] = set()
    seen_output_names: set[str] = set()
    for index, raw_record in enumerate(payload["images"]):
        if not isinstance(raw_record, dict):
            raise ValueError(f"images[{index}] must be an object")
        image_id = raw_record.get("id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError(f"images[{index}].id must be an integer")
        if image_id in seen_ids:
            raise ValueError(f"duplicate image id: {image_id}")
        file_name = _validate_file_name(raw_record.get("file_name"), image_id=image_id)
        output_name = f"{Path(file_name).stem}.png"
        if output_name in seen_output_names:
            raise ValueError(f"duplicate output image name: {output_name}")
        records.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": _positive_integer(
                    raw_record.get("width"), "width", image_id=image_id
                ),
                "height": _positive_integer(
                    raw_record.get("height"), "height", image_id=image_id
                ),
                "output_name": output_name,
            }
        )
        seen_ids.add(image_id)
        seen_output_names.add(output_name)
    return records


def _parse_lines(
    payload: dict[str, Any], image_ids: set[int]
) -> dict[int, list[tuple[float, float, float, float]]]:
    grouped: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for index, annotation in enumerate(payload["annotations"]):
        if not isinstance(annotation, dict):
            raise ValueError(f"annotations[{index}] must be an object")
        image_id = annotation.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError(f"annotations[{index}].image_id must be an integer")
        if image_id not in image_ids:
            raise ValueError(
                f"annotations[{index}] references unknown image id {image_id}"
            )
        line = annotation.get("line")
        if not isinstance(line, list) or len(line) != 4:
            raise ValueError(
                f"annotations[{index}].line must contain [x, y, dx, dy]"
            )
        values = []
        for coordinate in line:
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(coordinate)
            ):
                raise ValueError(
                    f"annotations[{index}].line must contain four finite numbers"
                )
            values.append(float(coordinate))
        grouped[image_id].append(tuple(values))
    return grouped


def _clip_coordinate(value: float, maximum: int) -> int:
    return int(round(max(0.0, min(value, float(maximum - 1)))))


def _draw_lines(
    image,
    lines: list[tuple[float, float, float, float]],
) -> None:
    height, width = image.shape[:2]
    thickness = max(1, round(min(width, height) / 160))
    for x, y, dx, dy in lines:
        cv2.line(
            image,
            (_clip_coordinate(x, width), _clip_coordinate(y, height)),
            (
                _clip_coordinate(x + dx, width),
                _clip_coordinate(y + dy, height),
            ),
            ANNOTATION_COLOR,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )


def _replace_output(temporary_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        temporary_dir.replace(output_dir)
        return
    if not overwrite:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; use --overwrite to replace it"
        )
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to replace an output path that is not a real directory: {output_dir}"
        )

    backup_dir = output_dir.with_name(
        f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    )
    output_dir.replace(backup_dir)
    try:
        temporary_dir.replace(output_dir)
    except Exception:
        backup_dir.replace(output_dir)
        raise
    shutil.rmtree(backup_dir)


def _validate_output_target(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    protected = {
        Path(resolved.anchor),
        Path.cwd().resolve(),
        Path.home().resolve(),
        DATA_ROOT.resolve(),
        DATA_ROOT.parent.resolve(),
        Path(tempfile.gettempdir()).resolve(),
    }
    if resolved in protected:
        raise ValueError(f"refusing to use a broad protected output directory: {resolved}")


def render_annotations(
    *,
    annotation_file: str | Path = DEFAULT_ANNOTATION_FILE,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int = 0,
    overwrite: bool = False,
) -> RenderSummary:
    """Render an annotation set atomically and return its output counts."""
    annotation_file = Path(annotation_file)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    _validate_output_target(output_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; use --overwrite to replace it"
        )
    if output_dir.exists() and (not output_dir.is_dir() or output_dir.is_symlink()):
        raise FileExistsError(
            f"refusing to replace an output path that is not a real directory: {output_dir}"
        )

    payload = _load_json(annotation_file)
    records = _parse_image_records(payload)
    grouped_lines = _parse_lines(payload, {record["id"] for record in records})
    selected_records = records if limit == 0 else records[:limit]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    line_count = 0
    try:
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        for record in selected_records:
            image_path = image_dir / record["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"source image {record['id']} does not exist: {image_path}"
                )
            image = cv2.imread(str(image_path), flags)
            if image is None:
                raise RuntimeError(
                    f"OpenCV could not decode image {record['id']}: {image_path}"
                )
            actual_height, actual_width = image.shape[:2]
            expected_size = (record["height"], record["width"])
            actual_size = (actual_height, actual_width)
            if actual_size != expected_size:
                raise ValueError(
                    f"image {record['id']} size mismatch for {image_path}: "
                    f"annotation={expected_size}, decoded={actual_size}"
                )
            lines = grouped_lines.get(record["id"], [])
            _draw_lines(image, lines)
            destination = temporary_dir / record["output_name"]
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(f"OpenCV could not write annotation render: {destination}")
            line_count += len(lines)

        _replace_output(temporary_dir, output_dir, overwrite)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    return RenderSummary(
        annotation_file=annotation_file.resolve(),
        output_dir=output_dir.resolve(),
        image_count=len(selected_records),
        line_count=line_count,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help=f"COCO line annotation JSON (default: {DEFAULT_ANNOTATION_FILE})",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"directory containing source images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=_nonnegative_integer,
        default=0,
        help="render only the first N image records; 0 renders all (default: 0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory after rendering succeeds",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = render_annotations(
        annotation_file=args.annotation_file,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Rendered images: {summary.image_count}")
    print(f"Rendered lines: {summary.line_count}")
    print(f"Annotation: {summary.annotation_file}")
    print(f"Output: {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
