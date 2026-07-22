import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from data.annotation_rendering import (
    ANNOTATION_COLOR,
    main,
    render_annotations,
)


def _write_image(path: Path, *, width: int = 12, height: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_annotations(path: Path, images: list[dict], annotations: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 0, "name": "line"}],
            }
        ),
        encoding="utf-8",
    )


def _image_record(
    image_id: int,
    file_name: str,
    *,
    width: int = 12,
    height: int = 12,
) -> dict:
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
    }


def test_render_uses_delta_endpoints_clips_and_preserves_empty_images(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "first.jpg")
    _write_image(image_dir / "empty.png")
    annotation_file = tmp_path / "lines.json"
    _write_annotations(
        annotation_file,
        [_image_record(10, "first.jpg"), _image_record(11, "empty.png")],
        [
            {"id": 1, "image_id": 10, "category_id": 0, "line": [2, 3, 5, 4]},
            {"id": 2, "image_id": 10, "category_id": 0, "line": [-4, 5, 30, 0]},
        ],
    )
    output_dir = tmp_path / "renders"

    summary = render_annotations(
        annotation_file=annotation_file,
        image_dir=image_dir,
        output_dir=output_dir,
    )

    assert summary.image_count == 2
    assert summary.line_count == 2
    assert [path.name for path in sorted(output_dir.iterdir())] == [
        "empty.png",
        "first.png",
    ]
    rendered = cv2.imread(str(output_dir / "first.png"), cv2.IMREAD_COLOR)
    assert tuple(rendered[3, 2]) == ANNOTATION_COLOR
    assert tuple(rendered[7, 7]) == ANNOTATION_COLOR
    assert tuple(rendered[5, 0]) == ANNOTATION_COLOR
    assert tuple(rendered[5, 11]) == ANNOTATION_COLOR
    empty = cv2.imread(str(output_dir / "empty.png"), cv2.IMREAD_COLOR)
    assert not np.any(empty)


def test_limit_uses_json_image_order_and_output_stems(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "z.jpg")
    _write_image(image_dir / "a.jpg")
    annotation_file = tmp_path / "lines.json"
    _write_annotations(
        annotation_file,
        [_image_record(9, "z.jpg"), _image_record(1, "a.jpg")],
        [
            {"image_id": 9, "line": [1, 1, 1, 1]},
            {"image_id": 1, "line": [1, 1, 1, 1]},
        ],
    )

    summary = render_annotations(
        annotation_file=annotation_file,
        image_dir=image_dir,
        output_dir=tmp_path / "renders",
        limit=1,
    )

    assert summary.image_count == 1
    assert summary.line_count == 1
    assert [path.name for path in summary.output_dir.iterdir()] == ["z.png"]


def test_existing_output_requires_overwrite_and_replacement_removes_stale_files(
    tmp_path,
):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "image.png")
    annotation_file = tmp_path / "lines.json"
    _write_annotations(
        annotation_file,
        [_image_record(1, "image.png")],
        [{"image_id": 1, "line": [1, 1, 2, 2]}],
    )
    output_dir = tmp_path / "renders"
    render_annotations(
        annotation_file=annotation_file,
        image_dir=image_dir,
        output_dir=output_dir,
    )
    stale = output_dir / "stale.txt"
    stale.write_text("preserve until explicit replacement", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        render_annotations(
            annotation_file=annotation_file,
            image_dir=image_dir,
            output_dir=output_dir,
        )
    assert stale.is_file()

    render_annotations(
        annotation_file=annotation_file,
        image_dir=image_dir,
        output_dir=output_dir,
        overwrite=True,
    )
    assert not stale.exists()
    assert (output_dir / "image.png").is_file()


def test_render_failure_preserves_existing_output_and_removes_temporary_files(
    tmp_path,
):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "valid.png")
    annotation_file = tmp_path / "lines.json"
    _write_annotations(
        annotation_file,
        [
            _image_record(1, "valid.png"),
            _image_record(2, "missing.png"),
        ],
        [],
    )
    output_dir = tmp_path / "renders"
    output_dir.mkdir()
    marker = output_dir / "marker.txt"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError, match="could not decode.*missing.png"):
        render_annotations(
            annotation_file=annotation_file,
            image_dir=image_dir,
            output_dir=output_dir,
            overwrite=True,
        )

    assert marker.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".renders.tmp-*"))


@pytest.mark.parametrize(
    "images,annotations,error",
    [
        (
            [_image_record(1, "a.png"), _image_record(1, "b.png")],
            [],
            "duplicate image id",
        ),
        (
            [_image_record(1, "a.png")],
            [{"image_id": 2, "line": [0, 0, 1, 1]}],
            "unknown image id",
        ),
        (
            [_image_record(1, "a.png")],
            [{"image_id": 1, "line": [0, 0, 1]}],
            "must contain.*dx",
        ),
        (
            [_image_record(1, "a.png")],
            [{"image_id": 1, "line": [0, 0, float("inf"), 1]}],
            "four finite numbers",
        ),
    ],
)
def test_invalid_annotation_records_fail_before_creating_output(
    tmp_path,
    images,
    annotations,
    error,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    annotation_file = tmp_path / "lines.json"
    _write_annotations(annotation_file, images, annotations)
    output_dir = tmp_path / "renders"

    with pytest.raises(ValueError, match=error):
        render_annotations(
            annotation_file=annotation_file,
            image_dir=image_dir,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".renders.tmp-*"))


def test_malformed_json_and_missing_lists_are_rejected(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    annotation_file = tmp_path / "lines.json"
    annotation_file.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        render_annotations(
            annotation_file=annotation_file,
            image_dir=image_dir,
            output_dir=tmp_path / "renders",
        )

    annotation_file.write_text(json.dumps({"images": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="'annotations'.*list"):
        render_annotations(
            annotation_file=annotation_file,
            image_dir=image_dir,
            output_dir=tmp_path / "renders",
        )


def test_decode_failure_and_size_mismatch_leave_no_output(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    broken = image_dir / "broken.png"
    broken.write_text("not an image", encoding="utf-8")
    annotation_file = tmp_path / "lines.json"
    _write_annotations(
        annotation_file,
        [_image_record(1, "broken.png")],
        [],
    )
    with pytest.raises(RuntimeError, match="could not decode"):
        render_annotations(
            annotation_file=annotation_file,
            image_dir=image_dir,
            output_dir=tmp_path / "broken-render",
        )
    assert not (tmp_path / "broken-render").exists()

    _write_image(image_dir / "actual.png", width=10, height=8)
    _write_annotations(
        annotation_file,
        [_image_record(2, "actual.png", width=12, height=8)],
        [],
    )
    with pytest.raises(ValueError, match="size mismatch"):
        render_annotations(
            annotation_file=annotation_file,
            image_dir=image_dir,
            output_dir=tmp_path / "mismatch-render",
        )
    assert not (tmp_path / "mismatch-render").exists()


def test_cli_reports_counts(tmp_path, capsys):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "image.png")
    annotation_file = tmp_path / "lines.json"
    _write_annotations(
        annotation_file,
        [_image_record(7, "image.png")],
        [{"image_id": 7, "line": [1, 1, 2, 2]}],
    )
    output_dir = tmp_path / "renders"

    assert main(
        [
            "--annotation-file",
            str(annotation_file),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "Rendered images: 1" in output
    assert "Rendered lines: 1" in output
    assert f"Annotation: {annotation_file.resolve()}" in output
    assert f"Output: {output_dir.resolve()}" in output


def test_real_wireframe_validation_sample_renders(tmp_path):
    summary = render_annotations(
        annotation_file="data/wireframe_processed/annotations/lines_val2017.json",
        image_dir="data/wireframe_processed/val2017",
        output_dir=tmp_path / "wireframe-render",
        limit=1,
    )

    assert summary.image_count == 1
    assert summary.line_count > 0
    images = list(summary.output_dir.glob("*.png"))
    assert len(images) == 1
    assert cv2.imread(str(images[0]), cv2.IMREAD_COLOR) is not None


def test_broad_output_directories_are_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="broad protected output directory"):
        render_annotations(
            annotation_file=tmp_path / "unused.json",
            image_dir=tmp_path,
            output_dir=tmp_path,
            overwrite=True,
        )


def test_readme_and_gitignore_document_annotation_rendering():
    readme = Path("README.md").read_text(encoding="utf-8")
    section = readme.split("### Annotation visualization", 1)[1].split(
        "## A–X supervised workflow", 1
    )[0]
    assert "python data/annotation_rendering.py" in section
    assert "--limit 10 --overwrite" in section
    assert "data/york_processed/annotations/lines_val2017.json" in section
    assert "data/wireframe_processed/annotations/lines_train2017.json" in section
    assert "data/test_render/" in Path(".gitignore").read_text(encoding="utf-8")
