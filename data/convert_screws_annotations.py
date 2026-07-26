#!/usr/bin/env python

"""Convert Screws COCO line annotations from XYXY endpoints to XY+delta."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    DATA_ROOT / "screws_processed" / "annotations" / "lines_train2017.json"
)
DEFAULT_OUTPUT = DEFAULT_INPUT
LINE_FORMAT = "xy_dxdy_v1"
SOURCE_LINE_FORMAT = "xyxy_v1"


@dataclass(frozen=True)
class ConversionSummary:
    input_path: Path
    output_path: Path
    source_sha256: str
    image_count: int
    line_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"input annotation does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"input annotation is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("annotation JSON root must be an object")
    for field in ("images", "annotations", "categories"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"annotation JSON field {field!r} must be a list")
    return payload


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_images(
    payload: dict[str, Any], image_dir: Path
) -> dict[int, dict[str, Any]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Screws image directory does not exist: {image_dir}")
    records: dict[int, dict[str, Any]] = {}
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    for index, record in enumerate(payload["images"]):
        if not isinstance(record, dict):
            raise ValueError(f"images[{index}] must be an object")
        image_id = record.get("id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError(f"images[{index}].id must be an integer")
        if image_id in records:
            raise ValueError(f"duplicate image id: {image_id}")
        file_name = record.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"image {image_id} file_name must be a non-empty string")
        relative_path = Path(file_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"image {image_id} has unsafe file_name: {file_name!r}")
        width = record.get("width")
        height = record.get("height")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError(f"image {image_id} width must be a positive integer")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError(f"image {image_id} height must be a positive integer")
        image_path = image_dir / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Screws annotation references missing image: {image_path}"
            )
        image = cv2.imread(str(image_path), flags)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode Screws image: {image_path}")
        actual_height, actual_width = image.shape[:2]
        if (actual_height, actual_width) != (height, width):
            raise ValueError(
                f"image {image_id} size mismatch for {image_path}: "
                f"annotation={(height, width)}, decoded={(actual_height, actual_width)}"
            )
        records[image_id] = record
    return records


def _convert_lines(
    payload: dict[str, Any], images: dict[int, dict[str, Any]]
) -> None:
    seen_ids: set[int] = set()
    for index, annotation in enumerate(payload["annotations"]):
        if not isinstance(annotation, dict):
            raise ValueError(f"annotations[{index}] must be an object")
        annotation_id = annotation.get("id")
        if isinstance(annotation_id, bool) or not isinstance(annotation_id, int):
            raise ValueError(f"annotations[{index}].id must be an integer")
        if annotation_id in seen_ids:
            raise ValueError(f"duplicate annotation id: {annotation_id}")
        seen_ids.add(annotation_id)
        image_id = annotation.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError(f"annotations[{index}].image_id must be an integer")
        if image_id not in images:
            raise ValueError(
                f"annotations[{index}] references unknown image id {image_id}"
            )
        line = annotation.get("line")
        if not isinstance(line, list) or len(line) != 4:
            raise ValueError(
                f"annotations[{index}].line must contain [x1, y1, x2, y2]"
            )
        x1, y1, x2, y2 = (
            _finite_number(value, f"annotations[{index}].line[{coordinate}]")
            for coordinate, value in enumerate(line)
        )
        image = images[image_id]
        width = image["width"]
        height = image["height"]
        if not (0.0 <= x1 <= width and 0.0 <= x2 <= width):
            raise ValueError(
                f"annotations[{index}] has an x endpoint outside image {image_id}"
            )
        if not (0.0 <= y1 <= height and 0.0 <= y2 <= height):
            raise ValueError(
                f"annotations[{index}] has a y endpoint outside image {image_id}"
            )
        annotation["line"] = [x1, y1, x2 - x1, y2 - y1]


def _write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.tmp-",
        dir=output_path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def convert_screws_annotations(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    image_dir: str | Path | None = None,
    overwrite: bool = False,
) -> ConversionSummary:
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if input_path == output_path and not overwrite:
        raise ValueError("in-place conversion requires --overwrite")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output annotation already exists: {output_path}; use --overwrite to replace it"
        )
    if output_path.exists() and (not output_path.is_file() or output_path.is_symlink()):
        raise FileExistsError(
            f"refusing to replace an output path that is not a regular file: {output_path}"
        )
    resolved_image_dir = (
        Path(image_dir).expanduser().resolve()
        if image_dir is not None
        else input_path.parent.parent / "train2017"
    )
    payload = _load_payload(input_path)
    if payload.get("line_format") == LINE_FORMAT:
        raise ValueError(f"annotation is already converted to {LINE_FORMAT}: {input_path}")
    if payload.get("line_format") is not None:
        raise ValueError(
            f"unsupported source line_format={payload['line_format']!r}: {input_path}"
        )
    source_sha256 = sha256_file(input_path)
    images = _validate_images(payload, resolved_image_dir)
    _convert_lines(payload, images)
    payload["line_format"] = LINE_FORMAT
    payload["source_line_format"] = SOURCE_LINE_FORMAT
    payload["source_annotation_sha256"] = source_sha256
    _write_json_atomic(payload, output_path)
    return ConversionSummary(
        input_path=input_path,
        output_path=output_path,
        source_sha256=source_sha256,
        image_count=len(payload["images"]),
        line_count=len(payload["annotations"]),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Screws train image directory; inferred beside annotations when omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing converted annotation after validation succeeds",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = convert_screws_annotations(
        args.input,
        args.output,
        image_dir=args.image_dir,
        overwrite=args.overwrite,
    )
    print(f"Converted images: {summary.image_count}")
    print(f"Converted lines: {summary.line_count}")
    print(f"Source SHA-256: {summary.source_sha256}")
    print(f"Output: {summary.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
