import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from data.convert_screws_annotations import (
    LINE_FORMAT,
    SOURCE_LINE_FORMAT,
    convert_screws_annotations,
    sha256_file,
)


def _write_fixture(
    root: Path,
    *,
    line: list[float] | None = None,
    image_id: int = 7,
) -> tuple[Path, Path, dict]:
    image_dir = root / "train2017"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "sample.png"), image)
    payload = {
        "images": [
            {
                "file_name": "sample.png",
                "height": 12,
                "width": 16,
                "id": 7,
            }
        ],
        "annotations": [
            {
                "id": 11,
                "image_id": image_id,
                "category_id": 0,
                "line": line if line is not None else [3.0, 9.0, 14.0, 2.0],
                "area": 1,
            }
        ],
        "categories": [{"supercategory": "line", "id": "0", "name": "line"}],
    }
    input_path = annotation_dir / "lines_train2017.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    return input_path, image_dir, payload


def test_converter_preserves_source_and_writes_dxdy_metadata(tmp_path):
    input_path, image_dir, source = _write_fixture(tmp_path)
    source_bytes = input_path.read_bytes()
    output_path = input_path.with_name("converted.json")

    summary = convert_screws_annotations(
        input_path,
        output_path,
        image_dir=image_dir,
    )

    converted = json.loads(output_path.read_text(encoding="utf-8"))
    assert input_path.read_bytes() == source_bytes
    assert converted["images"] == source["images"]
    assert converted["categories"] == source["categories"]
    assert converted["annotations"][0] == {
        **source["annotations"][0],
        "line": [3.0, 9.0, 11.0, -7.0],
    }
    assert converted["line_format"] == LINE_FORMAT
    assert converted["source_line_format"] == SOURCE_LINE_FORMAT
    assert converted["source_annotation_sha256"] == sha256_file(input_path)
    assert summary.image_count == 1
    assert summary.line_count == 1
    assert output_path.stat().st_mode & 0o777 == 0o644


def test_converter_is_deterministic_and_requires_explicit_overwrite(tmp_path):
    input_path, image_dir, _ = _write_fixture(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    convert_screws_annotations(input_path, first, image_dir=image_dir)
    convert_screws_annotations(input_path, second, image_dir=image_dir)
    assert first.read_bytes() == second.read_bytes()

    original_output = first.read_bytes()
    with pytest.raises(FileExistsError, match="--overwrite"):
        convert_screws_annotations(input_path, first, image_dir=image_dir)
    assert first.read_bytes() == original_output

    convert_screws_annotations(
        input_path,
        first,
        image_dir=image_dir,
        overwrite=True,
    )
    assert first.read_bytes() == original_output


def test_converter_supports_explicit_atomic_in_place_replacement(tmp_path):
    input_path, image_dir, _ = _write_fixture(tmp_path)
    source_hash = sha256_file(input_path)

    with pytest.raises(ValueError, match="in-place.*--overwrite"):
        convert_screws_annotations(input_path, input_path, image_dir=image_dir)

    summary = convert_screws_annotations(
        input_path,
        input_path,
        image_dir=image_dir,
        overwrite=True,
    )
    converted = json.loads(input_path.read_text(encoding="utf-8"))
    assert converted["line_format"] == LINE_FORMAT
    assert converted["source_annotation_sha256"] == source_hash
    assert summary.input_path == summary.output_path == input_path.resolve()

    with pytest.raises(ValueError, match="already converted"):
        convert_screws_annotations(
            input_path,
            input_path,
            image_dir=image_dir,
            overwrite=True,
        )


@pytest.mark.parametrize(
    ("line", "image_id", "error"),
    [
        ([1.0, 2.0, float("inf"), 4.0], 7, "finite number"),
        ([1.0, 2.0, 17.0, 4.0], 7, "x endpoint outside"),
        ([1.0, 2.0, 4.0, 13.0], 7, "y endpoint outside"),
        ([1.0, 2.0, 4.0, 5.0], 99, "unknown image id"),
    ],
)
def test_converter_rejects_invalid_annotations_without_output(
    tmp_path,
    line,
    image_id,
    error,
):
    input_path, image_dir, _ = _write_fixture(
        tmp_path,
        line=line,
        image_id=image_id,
    )
    output_path = tmp_path / "converted.json"

    with pytest.raises(ValueError, match=error):
        convert_screws_annotations(input_path, output_path, image_dir=image_dir)

    assert not output_path.exists()
