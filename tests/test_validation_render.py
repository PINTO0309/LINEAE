from types import SimpleNamespace
from pathlib import Path

import cv2
import pytest
import torch
from torch import nn

from main import select_epoch_evaluation_model
from models.lineae.lineae import PostProcess
from util.validation_render import (
    filter_render_predictions,
    prune_best_validation_renders,
    save_best_validation_renders,
    should_render_best_validation,
    validate_validation_render_options,
)


class _RenderDataset:
    def __init__(self, length=12):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.zeros(3, 32, 32), {
            "image_id": torch.tensor([index]),
            "size": torch.tensor([32, 32]),
            "orig_size": torch.tensor([32, 32]),
            "lines": torch.tensor([[0.0, 0.25, 1.0, 0.25]]),
            "labels": torch.tensor([0]),
        }


class _RenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        lines = torch.tensor([[0.0, 0.75, 1.0, 0.75]]).repeat(120, 1)
        logits = torch.full((120, 2), -10.0)
        logits[:, 0] = torch.linspace(8.0, -8.0, 120)
        self.register_buffer("lines", lines)
        self.register_buffer("logits", logits)
        self.seen_image_ids = []

    def forward(self, images, targets):
        self.seen_image_ids.extend(
            int(target["image_id"].flatten()[0].item()) for target in targets
        )
        batch_size = images.shape[0]
        return {
            "pred_lines": self.lines.unsqueeze(0).expand(batch_size, -1, -1),
            "pred_logits": self.logits.unsqueeze(0).expand(batch_size, -1, -1),
        }


def test_best_validation_render_writes_fixed_ten_side_by_side_images(tmp_path):
    model = _RenderModel()
    output = tmp_path / "run"

    rendered_dir = save_best_validation_renders(
        model=model,
        postprocessor=PostProcess(num_select=100),
        dataset=_RenderDataset(),
        device=torch.device("cpu"),
        output_dir=output,
        epoch=7,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        count=10,
        keep_best=10,
        score_threshold=0.3,
        max_predictions=5,
        batch_size=4,
        amp=False,
    )

    assert rendered_dir == output / "validation_renders" / "best_epoch_0007"
    images = sorted(rendered_dir.glob("*.png"))
    assert len(images) == 10
    assert images[0].name == "00_image_0.png"
    assert images[-1].name == "09_image_9.png"
    assert model.seen_image_ids == list(range(10))
    assert model.training

    rendered = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    assert rendered.shape == (64, 64, 3)
    assert tuple(rendered[40, 16]) == (96, 255, 64)
    assert tuple(rendered[56, 48]) == (64, 64, 255)


def test_prediction_filter_applies_threshold_sorting_and_maximum():
    lines = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    scores = torch.tensor([0.2, 0.9, 0.7, 0.95, 0.4])

    selected_lines, selected_scores = filter_render_predictions(
        lines,
        scores,
        score_threshold=0.5,
        max_predictions=2,
    )

    torch.testing.assert_close(selected_scores, torch.tensor([0.95, 0.9]))
    torch.testing.assert_close(selected_lines, lines[[3, 1]])


def test_best_render_retention_keeps_latest_ten_and_unrelated_entries(tmp_path):
    root = tmp_path / "validation_renders"
    root.mkdir()
    for epoch in range(12):
        (root / f"best_epoch_{epoch:04d}").mkdir()
    unrelated = root / "manual_comparison"
    unrelated.mkdir()
    note = root / "notes.txt"
    note.write_text("keep", encoding="utf-8")

    prune_best_validation_renders(root, keep_best=10)

    retained = sorted(path.name for path in root.glob("best_epoch_*"))
    assert retained == [f"best_epoch_{epoch:04d}" for epoch in range(2, 12)]
    assert unrelated.is_dir()
    assert note.read_text(encoding="utf-8") == "keep"


def test_best_render_conditions_exclude_nonbest_partial_and_disabled_epochs():
    assert should_render_best_validation(is_best=True, epoch_complete=True, count=10)
    assert not should_render_best_validation(
        is_best=False, epoch_complete=True, count=10
    )
    assert not should_render_best_validation(
        is_best=True, epoch_complete=False, count=10
    )
    assert not should_render_best_validation(
        is_best=True, epoch_complete=True, count=0
    )


def test_render_options_validate_defaults_and_disable_with_zero_count():
    valid = SimpleNamespace(
        validation_render_count=10,
        validation_render_keep_best=10,
        validation_render_score_threshold=0.3,
        validation_render_max_predictions=100,
        num_select=300,
    )
    validate_validation_render_options(valid)

    valid.validation_render_score_threshold = 1.1
    with pytest.raises(ValueError, match="score_threshold"):
        validate_validation_render_options(valid)
    valid.validation_render_count = 0
    validate_validation_render_options(valid)


def test_epoch_evaluation_model_selection_is_shared_by_metrics_and_rendering():
    model = object()
    ema_model = object()
    ema = SimpleNamespace(module=ema_model, num_updates=1)
    args = SimpleNamespace(eval_ema=True, ema_epoch=2)

    selected, active = select_epoch_evaluation_model(model, ema, args, epoch=1)
    assert selected is model
    assert not active

    selected, active = select_epoch_evaluation_model(model, ema, args, epoch=2)
    assert selected is ema_model
    assert active

    args.eval_ema = False
    selected, active = select_epoch_evaluation_model(model, ema, args, epoch=3)
    assert selected is model
    assert not active


def test_readme_documents_best_render_defaults_and_retention():
    readme = Path("README.md").read_text(encoding="utf-8")
    for setting in (
        "validation_render_count=10",
        "validation_render_keep_best=10",
        "validation_render_score_threshold=0.3",
        "validation_render_max_predictions=100",
    ):
        assert f"`{setting}`" in readme
    assert "best_epoch_XXXX" in readme
    assert "newest 10 best-update epochs" in readme
