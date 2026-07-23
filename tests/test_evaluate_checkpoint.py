from types import SimpleNamespace

import cv2
import pytest
import torch
from torch import nn

from models.lineae.lineae import PostProcess
from tools import evaluate_checkpoint


class _Dataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return torch.zeros(3, 32, 32), {
            "image_id": torch.tensor([index]),
            "size": torch.tensor([32, 32]),
            "orig_size": torch.tensor([32, 32]),
            "lines": torch.tensor([[0.0, 0.25, 1.0, 0.25]]),
            "labels": torch.tensor([0]),
        }


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "lines",
            torch.tensor(
                [
                    [0.0, 0.75, 1.0, 0.75],
                    [0.0, 0.50, 1.0, 0.50],
                    [0.0, 0.10, 1.0, 0.10],
                    [0.0, 0.90, 1.0, 0.90],
                ]
            ),
        )
        logits = torch.full((4, 2), -10.0)
        logits[:, 0] = torch.tensor([8.0, 1.0, -1.0, -8.0])
        self.register_buffer("logits", logits)

    def forward(self, images, targets):
        batch_size = images.shape[0]
        return {
            "pred_lines": self.lines.unsqueeze(0).expand(batch_size, -1, -1),
            "pred_logits": self.logits.unsqueeze(0).expand(batch_size, -1, -1),
        }


def test_evaluate_checkpoint_optionally_renders_each_dataset(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint_best.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    dataset_root = tmp_path / "wireframe"
    annotation = dataset_root / "annotations" / "lines_val2017.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text("{}", encoding="utf-8")
    output = tmp_path / "evaluation.json"
    render_root = tmp_path / "renders"

    model = _Model()
    checkpoint = {
        "model": model.state_dict(),
        "inference_model": "model",
    }
    config = SimpleNamespace(
        image_preprocess_schema="opencv_rgb_inter_linear_v2",
        num_select=4,
        num_queries=4,
        pretrained=True,
        amp=False,
        batch_size_val=1,
        modelname="LINEAE",
        criterionname="LINEACRITERION",
        eval_spatial_size=(32, 32),
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    )
    monkeypatch.setattr(
        evaluate_checkpoint.SLConfig,
        "fromfile",
        staticmethod(lambda _: config),
    )
    monkeypatch.setattr(
        evaluate_checkpoint,
        "validate_image_preprocess_schema",
        lambda _: None,
    )
    monkeypatch.setattr(
        evaluate_checkpoint,
        "validate_checkpoint_image_preprocess",
        lambda _: None,
    )
    monkeypatch.setattr(
        evaluate_checkpoint.torch,
        "load",
        lambda *args, **kwargs: checkpoint,
    )
    monkeypatch.setattr(
        evaluate_checkpoint,
        "create",
        lambda args, name: (
            (model, PostProcess(num_select=4)) if name == "modelname" else nn.Identity()
        ),
    )
    monkeypatch.setattr(evaluate_checkpoint, "build_dataset", lambda *args: _Dataset())
    monkeypatch.setattr(
        evaluate_checkpoint,
        "test",
        lambda *args, **kwargs: {"official_sap10": 12.5},
    )

    args = SimpleNamespace(
        config="config.py",
        checkpoint=str(checkpoint_path),
        dataset=[("wireframe", dataset_root)],
        device="cpu",
        amp=False,
        batch_size=1,
        num_workers=0,
        num_select=None,
        render_count=2,
        render_score_threshold=0.5,
        render_max_predictions=2,
        render_output_dir=render_root,
        output=output,
    )

    report = evaluate_checkpoint.evaluate(args)

    render_dir = render_root / "wireframe"
    images = sorted(render_dir.glob("*.png"))
    assert len(images) == 2
    assert cv2.imread(str(images[0]), cv2.IMREAD_COLOR).shape == (64, 64, 3)
    assert report["rendering"] == {
        "enabled": True,
        "count": 2,
        "score_threshold": 0.5,
        "max_predictions": 2,
        "candidate_limit": 4,
        "output_root": str(render_root.resolve()),
    }
    assert report["datasets"]["wireframe"]["render_dir"] == str(render_dir.resolve())
    assert output.is_file()

    with pytest.raises(FileExistsError, match="render output directory already exists"):
        evaluate_checkpoint.evaluate(args)

    render_only_root = tmp_path / "render-only"
    config.num_select = 2
    args.render_only = True
    args.render_count = 1
    args.render_max_predictions = 4
    args.render_output_dir = render_only_root
    args.output = None

    def render_only_create(config, name):
        if name == "criterionname":
            pytest.fail("render-only must not construct the criterion")
        return model, PostProcess(num_select=2)

    monkeypatch.setattr(evaluate_checkpoint, "create", render_only_create)
    monkeypatch.setattr(
        evaluate_checkpoint,
        "test",
        lambda *args, **kwargs: pytest.fail(
            "render-only must not run full-dataset evaluation"
        ),
    )

    render_report = evaluate_checkpoint.evaluate(args)

    assert render_report["format"] == "lineae_render_v1"
    assert render_report["render_only"] is True
    assert "checkpoint_sha256" not in render_report
    assert "official_sap10" not in render_report["datasets"]["wireframe"]
    assert render_report["rendering"]["candidate_limit"] == 4
    assert len(list((render_only_root / "wireframe").glob("*.png"))) == 1


def test_render_dataset_name_rejects_path_traversal():
    with pytest.raises(ValueError, match="single safe path components"):
        evaluate_checkpoint._render_directory_name("../wireframe")
