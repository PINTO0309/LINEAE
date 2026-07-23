import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from main import (
    create,
    get_args_parser,
    metric_improved,
    validate_checkpoint_cli_args,
)
from models.lineae.backbones.base import unwrap_state_dict
from util.experiment import sha256_file, write_experiment_records
from util.get_param_dicts import build_adamw_optimizer, get_optim_params
from util.model_ema import ModelEMA
from util.slconfig import SLConfig
from util.training_schedule import trainable_depth_for_epoch
from util.training_state import (
    INITIALIZATION_CRITICAL_FIELDS,
    REQUIRED_RESUME_FIELDS,
    atomic_torch_save,
    build_training_checkpoint,
    collect_distributed_rng_states,
    initialize_model_from_checkpoint,
    initialize_backbone_from_checkpoint,
    restore_checkpoint_rng_state,
    restore_training_checkpoint,
    validate_initialization_checkpoint,
    validate_backbone_initialization_checkpoint,
    validate_resume_checkpoint,
)
from warmup import LinearWarmup
from util.training_schedule import build_lr_scheduler


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.core = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.head = nn.Linear(4, 1)

    def forward(self, value):
        return self.head(self.backbone.core(value))


class FeatureTinyModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.distill_feature_projections = nn.ModuleList(
            nn.Conv2d(2, 2, kernel_size=1) for _ in range(3)
        )


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/linea/linea_hgnetv2_n.py",
        "configs/lineae/lineae_n.py",
        "configs/lineae/distill/lineae_n.py",
        "configs/lineae/lineae_t.py",
        "configs/lineae/distill/lineae_t.py",
        "configs/lineae/lineae_xl.py",
        "configs/lineae/lineae_2xl.py",
        "configs/lineae/lineae_3xl.py",
        "configs/lineae/probes/lineae_s.py",
    ],
)
def test_periodic_checkpoint_snapshots_are_disabled_by_default(config_path):
    config = SLConfig.fromfile(config_path)
    assert config.save_checkpoint_interval == 0


def _step(model, optimizer, scheduler, value):
    optimizer.zero_grad()
    loss = model(value).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()


def _args():
    return SimpleNamespace(
        modelname="LINEAE",
        backbone="test",
        num_classes=2,
        hidden_dim=4,
        num_queries=2,
        image_preprocess_schema="opencv_rgb_inter_linear_v2",
    )


def _resume_checkpoint_stub(config=None, **updates):
    saved_config = {"image_preprocess_schema": "opencv_rgb_inter_linear_v2"}
    if config is not None:
        saved_config.update(config)
    checkpoint = {
        "format_version": 2,
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "warmup_scheduler": None,
        "scaler": None,
        "epoch": 0,
        "epoch_complete": True,
        "global_step": 1,
        "sampler_epoch": 0,
        "best_metric_name": None,
        "best_metric": None,
        "best_epoch": None,
        "inference_model": "model",
        "config": saved_config,
        "git": {"revision": "test", "dirty": False},
        "rng_state": {},
    }
    checkpoint.update(updates)
    return checkpoint


def test_init_checkpoint_cli_is_distinct_from_resume_and_eval():
    parser = get_args_parser()
    defaults = parser.parse_args(["-c", "config.py"])
    assert defaults.ensemble is False
    assert defaults.ensemble_york_path == "data/york_processed"
    ensemble_args = parser.parse_args(
        ["-c", "config.py", "--ensemble", "--ensemble-york-path", "/data/york"]
    )
    assert ensemble_args.ensemble is True
    assert ensemble_args.ensemble_york_path == "/data/york"
    validate_checkpoint_cli_args(ensemble_args)
    ensemble_init_args = parser.parse_args(
        [
            "-c",
            "config.py",
            "--ensemble",
            "--init-checkpoint",
            "checkpoint_best.pth",
        ]
    )
    validate_checkpoint_cli_args(ensemble_init_args)
    ensemble_backbone_init_args = parser.parse_args(
        [
            "-c",
            "config.py",
            "--ensemble",
            "--init-backbone-checkpoint",
            "checkpoint_best.pth",
        ]
    )
    validate_checkpoint_cli_args(ensemble_backbone_init_args)
    ensemble_args.eval = True
    with pytest.raises(ValueError, match="training-only"):
        validate_checkpoint_cli_args(ensemble_args)

    args = parser.parse_args(
        ["-c", "config.py", "--init-checkpoint", "checkpoint_best.pth"]
    )
    assert args.init_checkpoint == "checkpoint_best.pth"
    validate_checkpoint_cli_args(args)

    args.resume = "checkpoint.pth"
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_checkpoint_cli_args(args)
    args.resume = ""
    args.eval = True
    with pytest.raises(ValueError, match="fresh training"):
        validate_checkpoint_cli_args(args)
    args.eval = False
    args.start_epoch = 1
    with pytest.raises(ValueError, match="start_epoch=0"):
        validate_checkpoint_cli_args(args)

    backbone_args = parser.parse_args(
        ["-c", "config.py", "--init-backbone-checkpoint", "backbone_best.pth"]
    )
    assert backbone_args.init_backbone_checkpoint == "backbone_best.pth"
    validate_checkpoint_cli_args(backbone_args)
    backbone_args.init_checkpoint = "checkpoint_best.pth"
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_checkpoint_cli_args(backbone_args)
    backbone_eval_args = parser.parse_args(
        [
            "-c",
            "config.py",
            "--init-backbone-checkpoint",
            "backbone_best.pth",
            "--eval",
        ]
    )
    with pytest.raises(ValueError, match="fresh training"):
        validate_checkpoint_cli_args(backbone_eval_args)
    backbone_epoch_args = parser.parse_args(
        [
            "-c",
            "config.py",
            "--init-backbone-checkpoint",
            "backbone_best.pth",
            "--start_epoch",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="start_epoch=0"):
        validate_checkpoint_cli_args(backbone_epoch_args)


def test_init_checkpoint_provenance_is_written_to_experiment_records(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: None)
    initialization = tmp_path / "checkpoint_best.pth"
    initialization.write_bytes(b"full-model initialization")
    initialization_sha256 = sha256_file(initialization)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    args = SimpleNamespace(
        coco_path=str(tmp_path / "dataset"),
        config_file="configs/lineae/lineae_n.py",
        backbone_weights=None,
        init_checkpoint=str(initialization),
        init_checkpoint_sha256=initialization_sha256,
        init_checkpoint_load_report={
            "loaded_tensor_count": 10,
            "new_state_keys": ["distill_feature_projections.0.weight"],
            "ignored_source_keys": [],
            "strict_shared_state": True,
        },
        init_backbone_checkpoint="",
        init_backbone_checkpoint_sha256=None,
        init_backbone_checkpoint_load_report=None,
        resume="",
        distill_weight=0.0,
        seed=43,
        world_size=1,
        batch_size_train=2,
        gradient_accumulation_steps=4,
        amp=False,
        device="cpu",
    )

    output_dir = tmp_path / "output"
    write_experiment_records(
        args=args,
        model=model,
        optimizer=optimizer,
        output_dir=output_dir,
        repo_root=tmp_path,
    )

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    record = manifest["checkpoints"]["model_initialization"]
    assert record["path"] == str(initialization.resolve())
    assert record["sha256"] == initialization_sha256
    assert record["exists"] is True
    assert record["load_report"] == args.init_checkpoint_load_report
    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    assert resolved["init_checkpoint"] == str(initialization)
    assert resolved["init_checkpoint_sha256"] == initialization_sha256


def test_ensemble_dataset_provenance_is_written_to_records_and_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: None)
    primary_annotation = tmp_path / "primary_train.json"
    york_train_annotation = tmp_path / "york_train.json"
    york_val_annotation = tmp_path / "york_val.json"
    primary_annotation.write_bytes(b"primary")
    york_train_annotation.write_bytes(b"york-train")
    york_val_annotation.write_bytes(b"york-val")
    york_hashes = {
        "train": sha256_file(york_train_annotation),
        "val": sha256_file(york_val_annotation),
    }
    sources = [
        {
            "name": "primary_train",
            "split": "train",
            "root": str(tmp_path),
            "image_dir": str(tmp_path),
            "annotation_file": str(primary_annotation),
            "samples": 5000,
        },
        {
            "name": "york_train",
            "split": "train",
            "root": str(tmp_path),
            "image_dir": str(tmp_path),
            "annotation_file": str(york_train_annotation),
            "annotation_sha256": york_hashes["train"],
            "samples": 0,
        },
        {
            "name": "york_val",
            "split": "val",
            "root": str(tmp_path),
            "image_dir": str(tmp_path),
            "annotation_file": str(york_val_annotation),
            "annotation_sha256": york_hashes["val"],
            "samples": 102,
        },
    ]
    args = SimpleNamespace(
        coco_path=str(tmp_path / "wireframe"),
        config_file="configs/lineae/lineae_n.py",
        backbone_weights=None,
        init_checkpoint="",
        init_checkpoint_sha256=None,
        init_checkpoint_load_report=None,
        init_backbone_checkpoint="",
        init_backbone_checkpoint_sha256=None,
        init_backbone_checkpoint_load_report=None,
        resume="",
        distill_weight=0.0,
        seed=42,
        world_size=1,
        batch_size_train=8,
        gradient_accumulation_steps=1,
        amp=False,
        device="cpu",
        ensemble=True,
        ensemble_york_path=str(tmp_path),
        ensemble_annotation_sha256=york_hashes,
        ensemble_split_samples={"train": 0, "val": 102},
        ensemble_training_sample_count=102,
        training_dataset_sample_count=5102,
        training_dataset_sources=sources,
    )
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output_dir = tmp_path / "output"

    write_experiment_records(
        args=args,
        model=model,
        optimizer=optimizer,
        output_dir=output_dir,
        repo_root=tmp_path,
    )

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["dataset"]["ensemble"] is True
    assert manifest["dataset"]["training_samples"] == 5102
    assert [source["samples"] for source in manifest["dataset"]["training_sources"]] == [
        5000,
        0,
        102,
    ]
    assert manifest["dataset"]["training_sources"][2]["annotation"]["sha256"] == (
        york_hashes["val"]
    )
    assert manifest["training"]["drop_last"] is False
    checkpoint = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        warmup_scheduler=None,
        scaler=None,
        epoch=0,
        global_step=1,
        args=args,
        repo_root=tmp_path,
    )
    assert checkpoint["config"]["ensemble"] is True
    assert checkpoint["config"]["ensemble_annotation_sha256"] == york_hashes
    assert checkpoint["config"]["ensemble_split_samples"] == {
        "train": 0,
        "val": 102,
    }
    assert checkpoint["config"]["training_dataset_sample_count"] == 5102


def test_backbone_checkpoint_initialization_strictly_loads_only_dino_core():
    source = TinyModel()
    with torch.no_grad():
        for parameter in source.backbone.core.parameters():
            parameter.fill_(2.0)
        for parameter in source.head.parameters():
            parameter.fill_(3.0)
    args = SimpleNamespace(
        modelname="LINEAE",
        variant="S",
        backbone="dinov3_test",
        image_preprocess_schema="opencv_rgb_inter_linear_v2",
    )
    checkpoint = _resume_checkpoint_stub(
        config=vars(args),
        model=source.state_dict(),
        epoch=7,
    )
    target = TinyModel()
    head_before = {
        name: value.clone() for name, value in target.head.state_dict().items()
    }

    report = initialize_backbone_from_checkpoint(
        checkpoint,
        model=target,
        args=args,
    )

    for name, value in target.backbone.core.state_dict().items():
        assert torch.equal(value, source.backbone.core.state_dict()[name])
    for name, value in target.head.state_dict().items():
        assert torch.equal(value, head_before[name])
    assert report["source_epoch"] == 7
    assert report["inference_model"] == "model"
    assert report["loaded_tensor_count"] == len(source.backbone.core.state_dict())
    assert report["ignored_source_tensor_count"] == len(source.head.state_dict())
    assert report["strict_core_state"] is True


def test_backbone_checkpoint_initialization_selects_ema_core():
    source = TinyModel()
    ema_state = {
        name: value.detach().clone().add_(4.0)
        for name, value in source.state_dict().items()
    }
    args = SimpleNamespace(
        modelname="LINEAE",
        variant="S",
        backbone="dinov3_test",
        image_preprocess_schema="opencv_rgb_inter_linear_v2",
    )
    checkpoint = _resume_checkpoint_stub(
        config=vars(args),
        model=source.state_dict(),
        inference_model="ema_model",
        ema_model={"model": ema_state},
        epoch=9,
    )
    target = TinyModel()
    head_before = {
        name: value.clone() for name, value in target.head.state_dict().items()
    }

    report = initialize_backbone_from_checkpoint(
        checkpoint,
        model=target,
        args=args,
    )

    selected_core = {
        name.removeprefix("backbone.core."): value
        for name, value in ema_state.items()
        if name.startswith("backbone.core.")
    }
    torch.testing.assert_close(target.backbone.core.state_dict(), selected_core)
    torch.testing.assert_close(target.head.state_dict(), head_before)
    assert report["source_epoch"] == 9
    assert report["inference_model"] == "ema_model"


def test_backbone_init_provenance_is_written_to_experiment_records(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: None)
    initialization = tmp_path / "old_wide_head_best.pth"
    initialization.write_bytes(b"same-variant DINO core")
    initialization_sha256 = sha256_file(initialization)
    load_report = {
        "source_epoch": 49,
        "inference_model": "model",
        "loaded_tensor_count": 368,
        "ignored_source_tensor_count": 100,
        "strict_core_state": True,
    }
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    args = SimpleNamespace(
        coco_path=str(tmp_path / "dataset"),
        config_file="configs/lineae/lineae_2xl.py",
        backbone_weights=None,
        init_checkpoint="",
        init_checkpoint_sha256=None,
        init_checkpoint_load_report=None,
        init_backbone_checkpoint=str(initialization),
        init_backbone_checkpoint_sha256=initialization_sha256,
        init_backbone_checkpoint_load_report=load_report,
        resume="",
        distill_weight=0.0,
        seed=42,
        world_size=1,
        batch_size_train=4,
        gradient_accumulation_steps=2,
        amp=False,
        device="cpu",
    )

    output_dir = tmp_path / "output"
    write_experiment_records(
        args=args,
        model=model,
        optimizer=optimizer,
        output_dir=output_dir,
        repo_root=tmp_path,
    )

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    record = manifest["checkpoints"]["backbone_model_initialization"]
    assert record["path"] == str(initialization.resolve())
    assert record["sha256"] == initialization_sha256
    assert record["load_report"] == load_report
    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    assert resolved["init_backbone_checkpoint"] == str(initialization)
    assert resolved["init_backbone_checkpoint_sha256"] == initialization_sha256
    assert resolved["init_backbone_checkpoint_load_report"] == load_report
    checkpoint = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        warmup_scheduler=None,
        scaler=None,
        epoch=0,
        global_step=1,
        args=args,
        repo_root=tmp_path,
    )
    assert checkpoint["config"]["init_backbone_checkpoint"] == str(
        initialization
    )
    assert (
        checkpoint["config"]["init_backbone_checkpoint_sha256"]
        == initialization_sha256
    )
    assert checkpoint["config"]["init_backbone_checkpoint_load_report"] == (
        load_report
    )


def test_backbone_checkpoint_initialization_rejects_unsafe_sources_before_mutation():
    source = TinyModel()
    args = SimpleNamespace(
        modelname="LINEAE",
        variant="S",
        backbone="dinov3_test",
        image_preprocess_schema="opencv_rgb_inter_linear_v2",
    )
    checkpoint = _resume_checkpoint_stub(config=vars(args), model=source.state_dict())
    assert validate_backbone_initialization_checkpoint(checkpoint, args)

    target = TinyModel()
    before = {name: value.clone() for name, value in target.state_dict().items()}
    checkpoint["config"]["variant"] = "M"
    with pytest.raises(ValueError, match="variant"):
        initialize_backbone_from_checkpoint(checkpoint, model=target, args=args)
    torch.testing.assert_close(target.state_dict(), before)

    checkpoint["config"]["variant"] = "S"
    checkpoint["epoch_complete"] = False
    with pytest.raises(ValueError, match="completed epoch"):
        initialize_backbone_from_checkpoint(checkpoint, model=target, args=args)
    checkpoint["epoch_complete"] = True
    checkpoint["format_version"] = 1
    with pytest.raises(ValueError, match="format version"):
        initialize_backbone_from_checkpoint(checkpoint, model=target, args=args)
    checkpoint["format_version"] = 2
    checkpoint["config"]["image_preprocess_schema"] = "legacy_pillow_v1"
    with pytest.raises(ValueError, match="image preprocessing schema"):
        initialize_backbone_from_checkpoint(checkpoint, model=target, args=args)
    checkpoint["config"]["image_preprocess_schema"] = (
        "opencv_rgb_inter_linear_v2"
    )
    checkpoint["model"] = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith("backbone.core.")
    }
    with pytest.raises(ValueError, match="contains no backbone.core tensors"):
        initialize_backbone_from_checkpoint(checkpoint, model=target, args=args)
    wrong_state = dict(source.state_dict())
    wrong_state["backbone.core.0.weight"] = torch.zeros(3, 3)
    checkpoint["model"] = wrong_state
    with pytest.raises(ValueError, match="shape_or_dtype"):
        initialize_backbone_from_checkpoint(checkpoint, model=target, args=args)
    torch.testing.assert_close(target.state_dict(), before)


def test_init_checkpoint_strict_loads_only_selected_model_weights():
    torch.manual_seed(3)
    source = TinyModel()
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=3e-4)
    _step(
        source,
        source_optimizer,
        torch.optim.lr_scheduler.LambdaLR(source_optimizer, lambda _: 1.0),
        torch.randn(2, 4),
    )
    args = _args()
    args.epochs = 9
    args.progressive_unfreeze = True
    checkpoint = _resume_checkpoint_stub(
        config={**vars(args), "epochs": 72, "progressive_unfreeze": False},
        model=source.state_dict(),
        epoch=7,
        global_step=123,
    )

    torch.manual_seed(17)
    initialized = TinyModel()
    fresh_optimizer = torch.optim.AdamW(initialized.parameters(), lr=1e-5)
    assert not fresh_optimizer.state
    report = initialize_model_from_checkpoint(
        checkpoint,
        model=initialized,
        args=args,
    )

    torch.testing.assert_close(initialized.state_dict(), source.state_dict())
    assert not fresh_optimizer.state
    assert report == {
        "source_epoch": 7,
        "inference_model": "model",
        "tensor_count": len(source.state_dict()),
        "loaded_tensor_count": len(source.state_dict()),
        "new_state_keys": [],
        "ignored_source_keys": [],
        "strict_shared_state": True,
    }
    assert checkpoint["global_step"] == 123


def test_old_accuracy_head_checkpoint_is_rejected_for_full_model_initialization():
    args = SimpleNamespace(
        accuracy_head_schema="wide_multilevel_residual_v1",
    )
    checkpoint = _resume_checkpoint_stub(model=TinyModel().state_dict())

    with pytest.raises(ValueError, match="accuracy_head_schema"):
        validate_initialization_checkpoint(checkpoint, args)


def test_init_checkpoint_selects_ema_and_rejects_mismatch_before_mutation():
    args = _args()
    source = TinyModel()
    ema_state = {
        name: value.detach().clone().add_(2.0)
        for name, value in source.state_dict().items()
    }
    checkpoint = _resume_checkpoint_stub(
        config=vars(args),
        model=source.state_dict(),
        inference_model="ema_model",
        ema_model={"model": ema_state},
    )
    initialized = TinyModel()
    report = initialize_model_from_checkpoint(
        checkpoint,
        model=initialized,
        args=args,
    )
    torch.testing.assert_close(initialized.state_dict(), ema_state)
    assert report["inference_model"] == "ema_model"

    before = {name: value.clone() for name, value in initialized.state_dict().items()}
    incompatible_args = _args()
    incompatible_args.num_queries = 3
    with pytest.raises(ValueError, match="num_queries"):
        initialize_model_from_checkpoint(
            checkpoint,
            model=initialized,
            args=incompatible_args,
        )
    torch.testing.assert_close(initialized.state_dict(), before)

    wrong_state = dict(ema_state)
    wrong_state["head.weight"] = torch.zeros(2, 4)
    checkpoint["ema_model"] = {"model": wrong_state}
    with pytest.raises(ValueError, match="shape_or_dtype"):
        initialize_model_from_checkpoint(
            checkpoint,
            model=initialized,
            args=args,
        )
    torch.testing.assert_close(initialized.state_dict(), before)


def test_init_checkpoint_only_allows_fresh_feature_kd_projection_keys():
    source = TinyModel()
    args = _args()
    args.distill_feature_weight = 1.0
    checkpoint = _resume_checkpoint_stub(
        config={**vars(args), "distill_feature_weight": 0.0},
        model=source.state_dict(),
    )
    initialized = FeatureTinyModel()
    initial_projection_state = {
        name: value.clone()
        for name, value in initialized.state_dict().items()
        if name.startswith("distill_feature_projections.")
    }

    report = initialize_model_from_checkpoint(
        checkpoint,
        model=initialized,
        args=args,
    )

    initialized_state = initialized.state_dict()
    for name, value in source.state_dict().items():
        assert torch.equal(initialized_state[name], value)
    for name, value in initial_projection_state.items():
        assert torch.equal(initialized_state[name], value)
    assert report["new_state_keys"] == sorted(initial_projection_state)
    assert report["loaded_tensor_count"] == len(source.state_dict())
    assert report["ignored_source_keys"] == []
    assert report["strict_shared_state"] is True

    unsafe_source = dict(source.state_dict())
    del unsafe_source["head.bias"]
    checkpoint["model"] = unsafe_source
    untouched = FeatureTinyModel()
    before = {name: value.clone() for name, value in untouched.state_dict().items()}
    with pytest.raises(ValueError, match="head.bias"):
        initialize_model_from_checkpoint(
            checkpoint,
            model=untouched,
            args=args,
        )
    torch.testing.assert_close(untouched.state_dict(), before)


def test_init_checkpoint_ignores_only_source_feature_projection_keys_when_disabled():
    source = FeatureTinyModel()
    args = _args()
    args.distill_feature_weight = 0.0
    checkpoint = _resume_checkpoint_stub(
        config={**vars(args), "distill_feature_weight": 1.0},
        model=source.state_dict(),
    )
    initialized = TinyModel()

    report = initialize_model_from_checkpoint(
        checkpoint,
        model=initialized,
        args=args,
    )

    for name, value in initialized.state_dict().items():
        assert torch.equal(value, source.state_dict()[name])
    assert report["new_state_keys"] == []
    assert report["ignored_source_keys"] == sorted(
        name
        for name in source.state_dict()
        if name.startswith("distill_feature_projections.")
    )


def test_init_checkpoint_requires_completed_current_format_and_model_semantics():
    args = _args()
    checkpoint = _resume_checkpoint_stub(config=vars(args), model=TinyModel().state_dict())
    assert validate_initialization_checkpoint(checkpoint, args)
    assert "epochs" not in INITIALIZATION_CRITICAL_FIELDS
    assert "progressive_unfreeze" not in INITIALIZATION_CRITICAL_FIELDS
    assert "distill_weight" not in INITIALIZATION_CRITICAL_FIELDS

    checkpoint["epoch_complete"] = False
    with pytest.raises(ValueError, match="completed epoch"):
        validate_initialization_checkpoint(checkpoint, args)
    checkpoint["epoch_complete"] = True
    checkpoint["format_version"] = 1
    with pytest.raises(ValueError, match="format version"):
        validate_initialization_checkpoint(checkpoint, args)


def test_resume_matches_uninterrupted_next_step(tmp_path):
    torch.manual_seed(7)
    np.random.seed(7)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    _step(model, optimizer, scheduler, torch.randn(3, 4))

    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=None,
        scaler=scaler,
        epoch=0,
        global_step=1,
        args=_args(),
        repo_root=tmp_path,
        best_metric_name="sap10",
        best_metric=42.5,
        best_epoch=0,
    )
    assert payload["epoch_complete"] is True
    checkpoint_path = tmp_path / "checkpoint.pth"
    atomic_torch_save(payload, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(checkpoint, _args())
    assert checkpoint["best_metric_name"] == "sap10"
    assert checkpoint["best_metric"] == 42.5
    assert checkpoint["best_epoch"] == 0

    next_value = torch.randn(3, 4)
    _step(model, optimizer, scheduler, next_value)

    resumed = TinyModel()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=3)
    resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
    restore_training_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        warmup_scheduler=None,
        scaler=resumed_scaler,
        device=torch.device("cpu"),
    )
    resumed_next_value = torch.randn(3, 4)
    assert torch.equal(next_value, resumed_next_value)
    _step(resumed, resumed_optimizer, resumed_scheduler, resumed_next_value)

    for expected, actual in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(expected, actual)
    assert scheduler.state_dict() == resumed_scheduler.state_dict()


def test_optimizer_cosine_warmup_is_contained_in_total_step_horizon():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    args = SimpleNamespace(
        lr_scheduler="cosine",
        scheduler_step_unit="optimizer",
        optimizer_steps_per_epoch=5,
        epochs=2,
        min_lr=0.0,
        use_warmup=True,
        warmup_iters=3,
    )
    scheduler = build_lr_scheduler(args, optimizer)
    warmup = LinearWarmup(scheduler, warmup_duration=args.warmup_iters)
    used_lrs = []

    for _ in range(10):
        used_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        warmup.step()
        if warmup.finished():
            scheduler.step()

    assert scheduler.T_max == 8
    assert scheduler.last_epoch == 8
    assert used_lrs[:3] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    assert args.lr_scheduler_total_units_resolved == 10
    assert args.lr_scheduler_warmup_units_resolved == 3
    assert args.lr_scheduler_post_warmup_units_resolved == 8


def test_optimizer_cosine_without_warmup_keeps_original_horizon():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    args = SimpleNamespace(
        lr_scheduler="cosine",
        scheduler_step_unit="optimizer",
        optimizer_steps_per_epoch=5,
        epochs=2,
        min_lr=0.0,
        use_warmup=False,
        warmup_iters=3,
    )

    scheduler = build_lr_scheduler(args, optimizer)

    assert scheduler.T_max == 10
    assert args.lr_scheduler_total_units_resolved == 10
    assert args.lr_scheduler_warmup_units_resolved == 0
    assert args.lr_scheduler_post_warmup_units_resolved == 10


def test_optimizer_warmup_must_fit_inside_total_horizon():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    args = SimpleNamespace(
        lr_scheduler="cosine",
        scheduler_step_unit="optimizer",
        optimizer_steps_per_epoch=5,
        epochs=2,
        min_lr=0.0,
        use_warmup=True,
        warmup_iters=10,
    )

    with pytest.raises(ValueError, match="smaller than the total optimizer-step horizon"):
        build_lr_scheduler(args, optimizer)


def _complete_optional_state_checkpoint(tmp_path):
    torch.manual_seed(17)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    warmup = LinearWarmup(scheduler, warmup_duration=3)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    ema = ModelEMA(model, decay=0.5, device=torch.device("cpu"))

    optimizer.zero_grad()
    loss = model(torch.randn(3, 4)).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    warmup.step()
    ema.update(model)

    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=warmup,
        scaler=scaler,
        epoch=0,
        global_step=1,
        args=SimpleNamespace(**vars(_args()), use_ema=True),
        repo_root=tmp_path,
        ema_model=ema,
        inference_model="ema_model",
    )
    return payload, model, optimizer, scheduler, warmup, scaler, ema


def _fresh_complete_optional_state():
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    warmup = LinearWarmup(scheduler, warmup_duration=3)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    ema = ModelEMA(model, decay=0.5, device=torch.device("cpu"))
    return model, optimizer, scheduler, warmup, scaler, ema


def test_resume_restores_every_optional_training_component(tmp_path):
    payload, model, optimizer, scheduler, warmup, scaler, ema = (
        _complete_optional_state_checkpoint(tmp_path)
    )
    expected_random = torch.randn(4)
    resumed = _fresh_complete_optional_state()
    resumed_model, resumed_optimizer, resumed_scheduler, resumed_warmup, resumed_scaler, resumed_ema = resumed

    restore_training_checkpoint(
        payload,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        warmup_scheduler=resumed_warmup,
        scaler=resumed_scaler,
        device=torch.device("cpu"),
        ema_model=resumed_ema,
    )

    torch.testing.assert_close(model.state_dict(), resumed_model.state_dict())
    torch.testing.assert_close(optimizer.state_dict(), resumed_optimizer.state_dict())
    assert scheduler.state_dict() == resumed_scheduler.state_dict()
    assert warmup.state_dict() == resumed_warmup.state_dict()
    assert scaler.state_dict() == resumed_scaler.state_dict()
    torch.testing.assert_close(ema.module.state_dict(), resumed_ema.module.state_dict())
    assert ema.num_updates == resumed_ema.num_updates
    assert torch.equal(expected_random, torch.randn(4))


def test_resume_rejects_missing_or_unexpected_component_state_before_loading(tmp_path):
    payload, *_ = _complete_optional_state_checkpoint(tmp_path)
    for field in ("scheduler", "warmup_scheduler", "scaler", "ema_model"):
        bad = dict(payload)
        bad[field] = None
        resumed = _fresh_complete_optional_state()
        resumed_model, resumed_optimizer, resumed_scheduler, resumed_warmup, resumed_scaler, resumed_ema = resumed
        model_before = {
            name: value.clone() for name, value in resumed_model.state_dict().items()
        }
        with pytest.raises(ValueError, match=field):
            restore_training_checkpoint(
                bad,
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                warmup_scheduler=resumed_warmup,
                scaler=resumed_scaler,
                device=torch.device("cpu"),
                ema_model=resumed_ema,
            )
        torch.testing.assert_close(model_before, resumed_model.state_dict())

    resumed_model = TinyModel()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="scheduler"):
        restore_training_checkpoint(
            payload,
            model=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=None,
            warmup_scheduler=None,
            scaler=None,
            device=torch.device("cpu"),
            ema_model=None,
        )


def test_partial_epoch_checkpoint_is_explicitly_non_resumable():
    checkpoint = _resume_checkpoint_stub(epoch_complete=False)

    with pytest.raises(ValueError, match="partial-epoch checkpoint"):
        validate_resume_checkpoint(checkpoint, SimpleNamespace())


@pytest.mark.parametrize("field", sorted(REQUIRED_RESUME_FIELDS))
def test_resume_schema_rejects_every_missing_full_state_field(field):
    checkpoint = _resume_checkpoint_stub()
    del checkpoint[field]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_resume_checkpoint(checkpoint, SimpleNamespace())


def test_resume_schema_rejects_sampler_or_ema_inconsistency():
    checkpoint = _resume_checkpoint_stub(sampler_epoch=1)
    with pytest.raises(ValueError, match="sampler_epoch"):
        validate_resume_checkpoint(checkpoint, SimpleNamespace())

    checkpoint = _resume_checkpoint_stub(inference_model="ema_model")
    with pytest.raises(ValueError, match="no EMA state"):
        validate_resume_checkpoint(checkpoint, SimpleNamespace())


def test_optimizer_topology_includes_frozen_backbone_for_progressive_unfreeze():
    model = TinyModel()
    model.backbone.core[0].requires_grad_(False)
    config = [
        {"params": "^backbone\\.core\\..*$", "lr": 1e-5},
        {"params": "^head\\..*$", "lr": 1e-3},
    ]
    groups = get_optim_params(config, model, include_frozen_backbone=True)
    grouped_ids = {id(parameter) for group in groups for parameter in group["params"]}
    assert grouped_ids == {id(parameter) for parameter in model.parameters()}
    assert config[0]["params"] == "^backbone\\.core\\..*$"


def test_optimizer_fallback_includes_unmatched_frozen_backbone_parameters():
    model = TinyModel()
    model.backbone.core[0].requires_grad_(False)
    groups = get_optim_params(
        [{"params": "^head\\..*$", "lr": 1e-3}],
        model,
        include_frozen_backbone=True,
    )

    grouped_ids = {id(parameter) for group in groups for parameter in group["params"]}
    assert grouped_ids == {id(parameter) for parameter in model.parameters()}


def test_s_optimizer_contains_every_future_trainable_parameter_once():
    config = SLConfig.fromfile("configs/lineae/lineae_s.py")
    config.pretrained = False
    model, _ = create(config, "modelname")
    model.backbone.set_trainable_depth(-1)
    optimizer = build_adamw_optimizer(config, model, torch.device("cpu"))

    eligible_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad or name.startswith("backbone.")
    }
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == eligible_ids


def _optimizer_args():
    return SimpleNamespace(
        model_parameters=[
            {"params": "^backbone\\.core\\..*$", "lr": 1e-5},
            {"params": "^head\\..*$", "lr": 1e-3},
        ],
        progressive_unfreeze=True,
        optimizer_fused=True,
        lr=1e-3,
        betas=[0.9, 0.999],
        weight_decay=1e-4,
    )


def test_adamw_builder_falls_back_from_fused_on_cpu():
    model = TinyModel()
    optimizer = build_adamw_optimizer(_optimizer_args(), model, torch.device("cpu"))

    assert optimizer.lineae_fused is False
    assert optimizer.defaults["fused"] is False
    grouped = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert grouped == {id(parameter) for parameter in model.parameters()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW requires CUDA")
def test_adamw_builder_uses_fused_kernel_on_cuda():
    model = TinyModel().cuda()
    optimizer = build_adamw_optimizer(_optimizer_args(), model, torch.device("cuda"))

    assert optimizer.lineae_fused is True
    assert optimizer.defaults["fused"] is True
    loss = model(torch.randn(2, 4, device="cuda")).square().mean()
    loss.backward()
    optimizer.step()


def _late_unfreeze_ddp_worker(rank, world_size, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(29)
        model = TinyModel()

        # Match main.py: DDP sees all future-trainable parameters before the
        # requested initial freeze is restored.
        model.backbone.core.requires_grad_(True)
        ddp_model = DistributedDataParallel(model, find_unused_parameters=True)
        ddp_model.module.backbone.core[0].requires_grad_(False)
        optimizer = build_adamw_optimizer(
            _optimizer_args(), ddp_model.module, torch.device("cpu")
        )
        frozen_parameter = ddp_model.module.backbone.core[0].weight
        grouped_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        assert id(frozen_parameter) in grouped_ids

        optimizer.zero_grad(set_to_none=True)
        frozen_before = frozen_parameter.detach().clone()
        first_input = torch.full((3, 4), float(rank + 1))
        ddp_model(first_input).square().mean().backward()
        optimizer.step()
        assert torch.equal(frozen_parameter, frozen_before)

        ddp_model.module.backbone.core[0].requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        unfrozen_before = frozen_parameter.detach().clone()
        second_input = torch.full((3, 4), float(2 * rank + 1))
        ddp_model(second_input).square().mean().backward()
        optimizer.step()
        assert not torch.equal(frozen_parameter, unfrozen_before)

        gathered = [torch.empty_like(frozen_parameter) for _ in range(world_size)]
        dist.all_gather(gathered, frozen_parameter.detach())
        for replica in gathered[1:]:
            torch.testing.assert_close(replica, gathered[0], rtol=0, atol=0)

        random.seed(101 + rank)
        np.random.seed(211 + rank)
        torch.manual_seed(307 + rank)
        rng_states = collect_distributed_rng_states()
        assert len(rng_states) == world_size
        assert not torch.equal(rng_states[0]["torch"], rng_states[1]["torch"])
        expected = (random.random(), float(np.random.rand()), torch.rand(3))

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        restore_checkpoint_rng_state(
            {
                "rng_state": rng_states[0],
                "rng_state_by_rank": rng_states,
            }
        )
        actual = (random.random(), float(np.random.rand()), torch.rand(3))
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert torch.equal(actual[2], expected[2])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="late-unfreeze DDP regression requires the Gloo backend",
)
def test_ddp_late_unfreeze_updates_and_synchronizes_registered_parameter(tmp_path):
    init_method = (tmp_path / "late-unfreeze-init").resolve().as_uri()
    mp.spawn(
        _late_unfreeze_ddp_worker,
        args=(2, init_method),
        nprocs=2,
        join=True,
    )


def test_progressive_unfreeze_schedule_boundaries():
    values = [
        trainable_depth_for_epoch(
            epoch=epoch,
            total_blocks=12,
            initial_depth=2,
            initial_freeze_epochs=5,
            unfreeze_interval=2,
            progressive=True,
        )
        for epoch in range(10)
    ]
    assert values == [2, 2, 2, 2, 2, 3, 3, 4, 4, 5]


def test_initial_freeze_boundary_adds_first_progressive_block():
    saved_epoch = 4
    resumed_epoch = saved_epoch + 1
    depth = trainable_depth_for_epoch(
        epoch=resumed_epoch,
        total_blocks=12,
        initial_depth=2,
        initial_freeze_epochs=5,
        unfreeze_interval=2,
        progressive=True,
    )
    assert depth == 3


def test_nonprogressive_negative_depth_keeps_entire_backbone_frozen():
    assert trainable_depth_for_epoch(
        epoch=0,
        total_blocks=12,
        initial_depth=-1,
        initial_freeze_epochs=0,
        unfreeze_interval=0,
        progressive=False,
    ) == -1


def test_validation_metric_selection_supports_max_min_and_rejects_nonfinite():
    assert metric_improved(1.0, None, "max")
    assert metric_improved(2.0, 1.0, "max")
    assert not metric_improved(1.0, 2.0, "max")
    assert metric_improved(1.0, 2.0, "min")
    with pytest.raises(FloatingPointError):
        metric_improved(float("nan"), 1.0, "max")


def test_resume_validation_covers_training_semantics_and_normalizes_sequences():
    args = SimpleNamespace(
        modelname="LINEAE",
        seed=42,
        eval_spatial_size=(640, 640),
        data_aug_scales=[(640, 640)],
        use_photometric_distort=False,
        optimizer_fused=True,
        distill_allow_unqualified_teacher=False,
        distill_teacher_resize=True,
        distill_matching_mode="gt_anchored",
        distill_teacher_gt_max_distance=10.0,
        distill_confidence_power=1.0,
        distill_temperature_steps=-1,
        distill_temperature_steps_resolved=99,
        distill_temperature_end=1.0,
        ema_decay=0.9997,
        sap_evaluation_protocol="official_all_queries_and_deployment_topk",
    )
    checkpoint = _resume_checkpoint_stub(
        config={
            **vars(args),
            "eval_spatial_size": [640, 640],
            "data_aug_scales": [[640, 640]],
        },
    )
    validate_resume_checkpoint(checkpoint, args)
    args.optimizer_fused = False
    with pytest.raises(ValueError, match="optimizer_fused"):
        validate_resume_checkpoint(checkpoint, args)
    args.optimizer_fused = True
    args.distill_teacher_resize = False
    with pytest.raises(ValueError, match="distill_teacher_resize"):
        validate_resume_checkpoint(checkpoint, args)
    args.distill_teacher_resize = True
    args.distill_matching_mode = "independent"
    with pytest.raises(ValueError, match="distill_matching_mode"):
        validate_resume_checkpoint(checkpoint, args)
    args.distill_matching_mode = "gt_anchored"
    args.distill_temperature_steps_resolved = 100
    with pytest.raises(ValueError, match="distill_temperature_steps_resolved"):
        validate_resume_checkpoint(checkpoint, args)
    args.distill_temperature_steps_resolved = 99
    args.distill_temperature_end = 2.0
    with pytest.raises(ValueError, match="distill_temperature_end"):
        validate_resume_checkpoint(checkpoint, args)


def test_resume_rejects_checkpoint_before_corrected_dn_noise_schema():
    args = SimpleNamespace(dn_line_noise_schema="endpoint_offset_v2")
    checkpoint = _resume_checkpoint_stub(config={})

    with pytest.raises(ValueError, match="denoising-line noise schema"):
        validate_resume_checkpoint(checkpoint, args)


def test_eval_can_load_checkpoint_before_corrected_dn_noise_schema():
    args = SimpleNamespace(dn_line_noise_schema="endpoint_offset_v2", eval=True)
    validate_resume_checkpoint(_resume_checkpoint_stub(config={}), args)


def test_resume_rejects_checkpoint_before_endpoint_loss_tie_schema():
    args = SimpleNamespace(endpoint_loss_schema="undirected_direct_tie_v2")
    checkpoint = _resume_checkpoint_stub(config={})

    with pytest.raises(ValueError, match="endpoint-loss tie handling"):
        validate_resume_checkpoint(checkpoint, args)


def test_resume_rejects_derivative_checkpoint_before_synthetic_p5_schema():
    args = SimpleNamespace(synthetic_p5_schema="avgpool_pointwise_v2")
    checkpoint = _resume_checkpoint_stub(config={})

    with pytest.raises(ValueError, match="synthetic-P5 architecture"):
        validate_resume_checkpoint(checkpoint, args)


def test_resume_rejects_checkpoint_before_wide_accuracy_head_schema():
    args = SimpleNamespace(
        eval=False,
        accuracy_head_schema="wide_multilevel_residual_v1",
    )
    checkpoint = _resume_checkpoint_stub(config={})

    with pytest.raises(ValueError, match="wide multilevel accuracy-head"):
        validate_resume_checkpoint(checkpoint, args)


def test_resume_rejects_checkpoint_before_ensemble_dataset_contract():
    args = SimpleNamespace(eval=False, ensemble=True)
    checkpoint = _resume_checkpoint_stub(config={})

    with pytest.raises(ValueError, match="Wireframe plus YorkUrban"):
        validate_resume_checkpoint(checkpoint, args)


def test_native_p5_variant_does_not_require_synthetic_p5_schema():
    args = SimpleNamespace(synthetic_p5_schema=None)
    validate_resume_checkpoint(_resume_checkpoint_stub(config={}), args)


@pytest.mark.parametrize(
    ("field", "saved", "changed"),
    [
        ("feat_channels_decoder", [256, 256, 256], [128, 128, 128]),
        ("dino_intermediate_fusion_schema", "weighted_v1", "residual_final_v1"),
        ("accuracy_head_schema", "legacy_xl_head_v1", "wide_multilevel_residual_v1"),
        ("encoder_use_indices", [2], [1, 2]),
        ("encoder_num_layers", 1, 2),
        ("ensemble", True, False),
        ("ensemble_york_path", "data/york_processed", "/datasets/york"),
        (
            "ensemble_annotation_sha256",
            {"train": "a" * 64, "val": "b" * 64},
            {"train": "a" * 64, "val": "c" * 64},
        ),
        (
            "ensemble_split_samples",
            {"train": 0, "val": 102},
            {"train": 0, "val": 101},
        ),
        ("training_dataset_sample_count", 5102, 5101),
        ("eval_idx", 5, 2),
        ("pe_temperatureH", 20, 10),
        ("freeze_norm", False, True),
        ("use_lab", True, False),
        ("set_cost_lines", 5.0, 2.0),
        ("endpoint_invariant_lines", True, False),
        ("use_lmap", False, True),
        ("num_workers", 8, 4),
        ("train_multiscale_scales", [256, 320, 320], [224, 320, 320]),
    ],
)
def test_resume_rejects_changed_architecture_loss_and_data_semantics(
    field, saved, changed
):
    args = SimpleNamespace(**{field: changed})
    checkpoint = _resume_checkpoint_stub(config={field: saved})

    with pytest.raises(ValueError, match=field):
        validate_resume_checkpoint(checkpoint, args)


def test_all_dino_training_recipes_reach_full_unfreeze_before_training_ends():
    paths = (
        "configs/lineae/lineae_s.py",
        "configs/lineae/probes/lineae_s.py",
        "configs/lineae/lineae_m.py",
        "configs/lineae/lineae_l.py",
        "configs/lineae/lineae_x.py",
        "configs/lineae/lineae_xl.py",
        "configs/lineae/distill/lineae_s.py",
        "configs/lineae/distill/lineae_m.py",
        "configs/lineae/distill/lineae_l.py",
        "configs/lineae/distill/lineae_x.py",
    )
    for path in paths:
        config = SLConfig.fromfile(path)
        schedule = [
            trainable_depth_for_epoch(
                epoch=epoch,
                total_blocks=12,
                initial_depth=config.backbone_trainable_layers,
                initial_freeze_epochs=config.initial_freeze_epochs,
                unfreeze_interval=config.unfreeze_interval,
                progressive=config.progressive_unfreeze,
            )
            for epoch in range(config.epochs)
        ]

        assert schedule[:5] == [2] * 5
        assert schedule[5:7] == [3] * 2
        assert config.epochs > 23
        assert schedule[23:] == [12] * (config.epochs - 23)


@pytest.mark.parametrize(
    "path,total_blocks",
    [
        ("configs/lineae/lineae_2xl.py", 24),
        ("configs/lineae/lineae_3xl.py", 32),
    ],
)
def test_large_accuracy_variants_are_fully_unfrozen_from_epoch_zero(
    path,
    total_blocks,
):
    config = SLConfig.fromfile(path)
    schedule = [
        trainable_depth_for_epoch(
            epoch=epoch,
            total_blocks=total_blocks,
            initial_depth=config.backbone_trainable_layers,
            initial_freeze_epochs=config.initial_freeze_epochs,
            unfreeze_interval=config.unfreeze_interval,
            progressive=config.progressive_unfreeze,
        )
        for epoch in range(config.epochs)
    ]

    assert schedule == [total_blocks] * config.epochs


def test_ema_updates_resume_and_selects_inference_weights(tmp_path):
    torch.manual_seed(11)
    model = TinyModel()
    ema = ModelEMA(model, decay=0.5, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    first = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
    ema.update(model)

    for name, value in ema.module.state_dict().items():
        expected = (first[name] + model.state_dict()[name]) * 0.5
        assert torch.equal(value, expected)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        warmup_scheduler=None,
        scaler=None,
        epoch=2,
        global_step=3,
        args=_args(),
        repo_root=tmp_path,
        ema_model=ema,
        inference_model="ema_model",
    )
    selected = unwrap_state_dict(payload)
    for name, value in ema.module.state_dict().items():
        assert torch.equal(selected[name], value)

    restored_model = TinyModel()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_ema = ModelEMA(restored_model, decay=0.5, device=torch.device("cpu"))
    restore_training_checkpoint(
        payload,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=None,
        warmup_scheduler=None,
        scaler=None,
        device=torch.device("cpu"),
        ema_model=restored_ema,
    )
    assert restored_ema.num_updates == 2
    for expected, actual in zip(ema.module.parameters(), restored_ema.module.parameters()):
        assert torch.equal(expected, actual)


def test_real_xl_ema_checkpoint_selects_reloadable_canonical_teacher_weights(tmp_path):
    def small_config(path):
        config = SLConfig.fromfile(path)
        config.pretrained = False
        config.eval_spatial_size = (64, 64)
        config.enforce_variant_input = False
        config.num_queries = 20
        config.num_select = 10
        config.dn_number = 4
        return config

    torch.manual_seed(73)
    config = small_config("configs/lineae/ablations/lineae_xl_ema.py")
    model, _ = create(config, "modelname")
    criterion = create(config, "criterionname")
    model.train()
    criterion.train()
    tracked_name, tracked = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "decoder.class_embed" in name
        and name.endswith(".weight")
        and parameter.requires_grad
    ][-1]
    optimizer = torch.optim.SGD([tracked], lr=1e-3)
    ema = ModelEMA(model, decay=0.5, device=torch.device("cpu"))
    targets = [{
        "labels": torch.tensor([0]),
        "lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]]),
    }]

    first_source = None
    for step in range(2):
        images = torch.randn(1, 3, 64, 64)
        model.zero_grad(set_to_none=True)
        loss = sum(criterion(model(images, targets), targets).values())
        loss.backward()
        optimizer.step()
        ema.update(model)
        assert torch.isfinite(loss)
        if step == 0:
            first_source = tracked.detach().clone()
            first_ema = dict(ema.module.named_parameters())[tracked_name]
            assert torch.equal(first_ema, first_source)

    assert first_source is not None
    ema_tracked = dict(ema.module.named_parameters())[tracked_name]
    torch.testing.assert_close(
        ema_tracked,
        first_source * 0.5 + tracked.detach() * 0.5,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(ema_tracked, tracked.detach())
    assert ema.num_updates == 2

    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        warmup_scheduler=None,
        scaler=None,
        epoch=0,
        global_step=2,
        args=SimpleNamespace(**config._cfg_dict.to_dict()),
        repo_root=tmp_path,
        ema_model=ema,
        inference_model="ema_model",
    )
    selected = unwrap_state_dict(payload)
    assert payload["inference_model"] == "ema_model"
    assert selected.keys() == ema.module.state_dict().keys()
    assert all(
        torch.equal(value, ema.module.state_dict()[name])
        for name, value in selected.items()
    )

    source_reload, _ = create(
        small_config("configs/lineae/ablations/lineae_xl_ema.py"),
        "modelname",
    )
    source_reload.load_state_dict(selected, strict=True)
    source_reload.eval()
    probe = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        source_outputs = source_reload(probe)
        ema_outputs = ema.module(probe)
    del source_reload

    canonical_reload, _ = create(
        small_config("configs/lineae/lineae_xl.py"),
        "modelname",
    )
    canonical_reload.load_state_dict(selected, strict=True)
    canonical_reload.eval()
    with torch.inference_mode():
        canonical_outputs = canonical_reload(probe)

    for key in ("pred_logits", "pred_lines"):
        torch.testing.assert_close(
            ema_outputs[key],
            source_outputs[key],
            rtol=1e-6,
            atol=1e-6,
        )
        assert torch.equal(source_outputs[key], canonical_outputs[key])
