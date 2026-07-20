import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from main import create, metric_improved
from models.lineae.backbones.base import unwrap_state_dict
from util.get_param_dicts import build_adamw_optimizer, get_optim_params
from util.model_ema import ModelEMA
from util.slconfig import SLConfig
from util.training_schedule import trainable_depth_for_epoch
from util.training_state import (
    REQUIRED_RESUME_FIELDS,
    atomic_torch_save,
    build_training_checkpoint,
    collect_distributed_rng_states,
    restore_checkpoint_rng_state,
    restore_training_checkpoint,
    validate_resume_checkpoint,
)
from warmup import LinearWarmup


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.core = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.head = nn.Linear(4, 1)

    def forward(self, value):
        return self.head(self.backbone.core(value))


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
    )


def _resume_checkpoint_stub(config=None, **updates):
    checkpoint = {
        "format_version": 1,
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
        "config": {} if config is None else config,
        "git": {"revision": "test", "dirty": False},
        "rng_state": {},
    }
    checkpoint.update(updates)
    return checkpoint


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
    assert values == [-1, -1, -1, -1, -1, 2, 2, 3, 3, 4]


def test_resume_epoch_crosses_from_frozen_warmup_to_initial_depth():
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
    assert depth == 2


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
        distill_temperature_steps=-1,
        distill_temperature_steps_resolved=99,
        distill_temperature_end=1.0,
        ema_decay=0.9997,
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
    args.distill_temperature_steps_resolved = 100
    with pytest.raises(ValueError, match="distill_temperature_steps_resolved"):
        validate_resume_checkpoint(checkpoint, args)
    args.distill_temperature_steps_resolved = 99
    args.distill_temperature_end = 2.0
    with pytest.raises(ValueError, match="distill_temperature_end"):
        validate_resume_checkpoint(checkpoint, args)


@pytest.mark.parametrize(
    ("field", "saved", "changed"),
    [
        ("feat_channels_decoder", [256, 256, 256], [128, 128, 128]),
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
        "configs/lineae/baselines/lineae_s.py",
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

        assert schedule[:5] == [-1] * 5
        assert schedule[5] == 2
        assert config.epochs > 25
        assert schedule[25:] == [12] * (config.epochs - 25)


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
