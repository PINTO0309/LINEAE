
from pathlib import Path

import pytest
import torch

from main import build_frozen_teacher, validate_teacher_artifact
from main import create
from datasets.collate import generate_scales
from models.lineae.hgnetv2 import LearnableAffineBlock
from models.lineae.tuning import TUNING_CANDIDATES
from models.lineae.variants import VARIANTS, validate_variant_config
from util.slconfig import SLConfig
from util.experiment import config_fingerprint, sha256_file
from util.profiler import stats as complexity_stats


DEPLOY_PARAMETER_COUNTS = {
    "A": (296_392, 1_601_102, 1_897_494),
    "F": (653_000, 1_946_613, 2_599_613),
    "P": (1_012_370, 1_985_013, 2_997_383),
    "N": (1_850_396, 2_063_349, 3_913_745),
    "T": (2_197_004, 6_236_101, 8_433_105),
    "S": (6_003_648, 5_924_805, 11_928_453),
    "M": (10_593_536, 6_677_228, 17_270_764),
    "L": (22_979_200, 6_677_228, 29_656_428),
    "X": (30_075_520, 8_083_770, 38_159_290),
    "XL": (88_423_936, 8_083_770, 96_507_706),
    "2XL": (306_825_984, 8_083_770, 314_909_754),
    "3XL": (845_222_912, 8_083_770, 853_306_682),
}

DEPLOY_GFLOPS = {
    "A": 2.5,
    "F": 4.7,
    "P": 10.8,
    "N": 11.7,
    "T": 29.4,
    "S": 39.2,
    "M": 55.5,
    "L": 94.5,
    "X": 121.2,
    "XL": 306.3,
    "2XL": 1005.6,
    "3XL": 2731.4,
}

LARGE_VARIANTS = ("2XL", "3XL")
RUNTIME_TEST_VARIANTS = tuple(
    variant for variant in VARIANTS if variant not in LARGE_VARIANTS
)

LAB_MODULE_COUNTS = {"A": 20, "F": 20, "P": 25, "N": 30, "T": 30}

NO_KD_EPOCHS = {
    "A": 72,
    "F": 66,
    "P": 60,
    "N": 72,
    "T": 60,
    "S": 45,
    "M": 45,
    "L": 40,
    "X": 35,
    "XL": 36,
    "2XL": 60,
    "3XL": 72,
}

DISTILL_EPOCHS = {
    "A": 60,
    "F": 55,
    "P": 50,
    "N": 50,
    "T": 45,
    "S": 40,
    "M": 40,
    "L": 30,
    "X": 30,
}


def _readme_table_rows() -> set[tuple[str, ...]]:
    return {
        tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        for line in Path("README.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    }


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_variant_registry_matches_config(variant):
    config = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    assert validate_variant_config(config) == VARIANTS[variant]
    assert not [key for key in config._cfg_dict if key.startswith("_")]
    assert config.pretty_text


def test_variant_registry_prevents_silent_preprocessing_and_shape_drift():
    config = SLConfig.fromfile("configs/lineae/lineae_s.py")
    config.image_mean = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="image_mean"):
        validate_variant_config(config)
    config = SLConfig.fromfile("configs/lineae/lineae_s.py")
    config.eval_spatial_size = (512, 512)
    with pytest.raises(ValueError, match="eval_spatial_size"):
        validate_variant_config(config)
    config.enforce_variant_input = False
    assert validate_variant_config(config) == VARIANTS["S"]


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_readme_deploy_parameter_and_lab_inventory_matches_models(variant):
    config = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    config.pretrained = False
    if variant in LARGE_VARIANTS:
        with torch.device("meta"):
            model, _ = create(config, "modelname")
            model.deploy()
        backbone = sum(parameter.numel() for parameter in model.backbone.parameters())
        total = sum(parameter.numel() for parameter in model.parameters())
        after_backbone = total - backbone
    else:
        model, _ = create(config, "modelname")
        model.deploy()
        backbone = sum(parameter.numel() for parameter in model.backbone.parameters())
        total = sum(parameter.numel() for parameter in model.parameters())
        after_backbone = total - backbone
    assert (backbone, after_backbone, total) == DEPLOY_PARAMETER_COUNTS[variant]

    lab_modules = sum(
        isinstance(module, LearnableAffineBlock)
        for module in model.backbone.modules()
    )
    assert lab_modules == LAB_MODULE_COUNTS.get(variant, 0)

    expected_row = (
        variant,
        f"{backbone / 1_000_000:.1f}",
        f"{after_backbone / 1_000_000:.1f}",
        f"{total / 1_000_000:.1f}",
        f"{DEPLOY_GFLOPS[variant]:.1f}",
    )
    assert expected_row in _readme_table_rows()


@pytest.mark.parametrize(
    "variant,expected_flops,expected_macs",
    [
        ("2XL", "1.0056 TFLOPS", "502.431 GMACs"),
        ("3XL", "2.7314 TFLOPS", "1.3652 TMACs"),
    ],
)
def test_large_variant_complexity_uses_meta_graph(
    variant,
    expected_flops,
    expected_macs,
):
    config = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    config.pretrained = False
    with torch.device("meta"):
        model, _ = create(config, "modelname")

    complexity = complexity_stats(model, config)

    assert complexity["flops"] == expected_flops
    assert complexity["macs"] == expected_macs
    assert complexity["params"] == DEPLOY_PARAMETER_COUNTS[variant][2]


def test_afpn_deploy_parameter_counts_are_strictly_monotonic():
    totals = [DEPLOY_PARAMETER_COUNTS[variant][2] for variant in ("A", "F", "P", "N")]
    backbones = [
        DEPLOY_PARAMETER_COUNTS[variant][0] for variant in ("A", "F", "P", "N")
    ]
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)
    assert backbones == sorted(backbones)
    assert len(set(backbones)) == len(backbones)


def test_afpn_query_and_decoder_scale_matches_latency_contract():
    configs = [
        SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
        for variant in ("A", "F", "P", "N")
    ]
    assert [config.num_queries for config in configs] == [600, 800, 1100, 1200]
    assert [config.dec_layers for config in configs] == [2, 3, 3, 3]
    assert [config.eval_idx for config in configs] == [1, 2, 2, 2]


def test_t_capacity_is_between_n_and_s_and_below_linea_s_limit():
    n_total = DEPLOY_PARAMETER_COUNTS["N"][2]
    t_total = DEPLOY_PARAMETER_COUNTS["T"][2]
    s_total = DEPLOY_PARAMETER_COUNTS["S"][2]

    assert n_total < t_total < s_total
    assert t_total <= 8_600_000


def test_t_config_uses_hgnetv2_b1_and_linea_s_detector_dimensions():
    config = SLConfig.fromfile("configs/lineae/lineae_t.py")

    assert config.backbone == "hgnetv2_t"
    assert config.backbone_weights == "ckpts/PPHGNetV2_B1_stage1.pth"
    assert config.eval_spatial_size == (640, 640)
    assert config.in_channels_encoder == [256, 512, 1024]
    assert config.hidden_dim == 256
    assert config.dim_feedforward == 512
    assert config.nheads == 8
    assert config.dec_layers == 3
    assert config.eval_idx == 2
    assert config.num_queries == 1100
    assert config.num_select == 300
    assert config.progressive_unfreeze is False
    assert config.backbone_trainable_layers == 0


@pytest.mark.parametrize("variant", RUNTIME_TEST_VARIANTS)
def test_every_variant_runs_the_shared_detector_and_selection_contract(variant):
    config = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    model, postprocessor = create(config, "modelname")
    model.eval()

    with torch.inference_mode():
        outputs = model(torch.randn(1, 3, 64, 64))
        selected = postprocessor(outputs, torch.ones(1, 2))

    assert outputs["pred_logits"].shape == (1, 20, 2)
    assert outputs["pred_lines"].shape == (1, 20, 4)
    assert torch.isfinite(outputs["pred_logits"]).all()
    assert torch.isfinite(outputs["pred_lines"]).all()
    assert selected[0]["scores"].shape == (10,)
    assert selected[0]["lines"].shape == (10, 4)


@pytest.mark.parametrize("variant", RUNTIME_TEST_VARIANTS)
def test_every_variant_runs_a_finite_denoising_training_update(variant):
    torch.manual_seed(47)
    config = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    config.dn_number = 4
    model, _ = create(config, "modelname")
    criterion = create(config, "criterionname")
    model.train()
    criterion.train()

    tracked = next(
        parameter
        for name, parameter in model.named_parameters()
        if "decoder.class_embed" in name and parameter.requires_grad
    )
    optimizer = torch.optim.SGD([tracked], lr=1e-3)
    before = tracked.detach().clone()
    images = torch.randn(1, 3, 64, 64)
    targets = [{
        "labels": torch.tensor([0]),
        "lines": torch.tensor([[0.1, 0.2, 0.8, 0.9]]),
    }]

    outputs = model(images, targets)
    total = sum(criterion(outputs, targets).values())
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    assert torch.isfinite(total)
    assert tracked.grad is not None
    assert torch.isfinite(tracked.grad).all()
    assert torch.count_nonzero(tracked.grad)
    assert not torch.equal(before, tracked.detach())


@pytest.mark.parametrize("variant", RUNTIME_TEST_VARIANTS)
def test_every_variant_preserves_detector_and_topk_outputs_after_fused_deploy(variant):
    torch.manual_seed(89)
    config = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    model, postprocessor = create(config, "modelname")
    model.eval()
    images = torch.randn(1, 3, 64, 64)
    target_sizes = torch.ones(1, 2)

    with torch.inference_mode():
        reference = model(images)
        reference_selected = postprocessor(reference, target_sizes)[0]
        model.deploy()
        postprocessor.deploy()
        deployed = model(images)
        deployed_lines, deployed_scores = postprocessor(deployed, target_sizes)

    for key in ("pred_logits", "pred_lines"):
        torch.testing.assert_close(
            deployed[key],
            reference[key],
            rtol=1e-4,
            atol=1e-5,
        )
    assert deployed_lines.shape == (1, 10, 4)
    assert deployed_scores.shape == (1, 10)
    torch.testing.assert_close(
        deployed_lines[0],
        reference_selected["lines"],
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        deployed_scores[0],
        reference_selected["scores"],
        rtol=1e-4,
        atol=1e-5,
    )
    assert torch.isfinite(deployed_lines).all()
    assert torch.isfinite(deployed_scores).all()


@pytest.mark.parametrize("variant", ["A", "F", "P", "N", "T", "S", "M", "L", "X"])
def test_distillation_configs_are_enabled_and_require_qualified_teacher(variant, tmp_path):
    config = SLConfig.fromfile(f"configs/lineae/distill/lineae_{variant.lower()}.py")
    config.distill_teacher_checkpoint = str(tmp_path / "missing_teacher.pth")
    assert config.distill_weight > 0
    assert config.distill_teacher_resize is True
    assert config.distill_temperature_start == 1.0
    assert config.distill_temperature_end == 4.0
    assert config.distill_temperature_steps == -1
    with pytest.raises(FileNotFoundError, match="teacher checkpoint is missing"):
        build_frozen_teacher(config, "cpu")


def test_no_kd_does_not_require_or_construct_teacher(tmp_path):
    config = SLConfig.fromfile("configs/lineae/lineae_s.py")
    config.distill_teacher_config = str(tmp_path / "missing.py")
    config.distill_teacher_checkpoint = str(tmp_path / "missing.pth")
    teacher, criterion = build_frozen_teacher(config, "cpu")
    assert teacher is None
    assert criterion is None


def test_full_s_recipe_is_distillation_matched_but_probe_stays_one_batch():
    probe = SLConfig.fromfile("configs/lineae/probes/lineae_s.py")
    baseline = SLConfig.fromfile("configs/lineae/lineae_s.py")
    distillation = SLConfig.fromfile("configs/lineae/distill/lineae_s.py")

    assert probe.gradient_accumulation_steps == 1
    assert probe.recipe_reference_effective_batch_size == 1
    assert baseline.distill_weight == 0.0
    assert probe.training_profile == "p0_smoke"
    assert probe.multi_scale_train is False
    assert baseline.training_profile == "single_gpu_96gb"
    assert baseline.multi_scale_train is True
    assert distillation.multi_scale_train is True
    assert baseline.pin_memory is True
    assert baseline.prefetch_factor == 2
    assert baseline.multiprocessing_sharing_strategy == "file_system"
    assert baseline.batch_size_train == 8
    assert baseline.recipe_reference_effective_batch_size == 8
    assert baseline.gradient_accumulation_steps == 1
    assert baseline.batch_size_train * baseline.gradient_accumulation_steps == 8
    assert baseline.scheduler_step_unit == "optimizer"
    assert baseline.gradient_accumulation_steps == distillation.gradient_accumulation_steps
    assert baseline.scheduler_step_unit == distillation.scheduler_step_unit
    assert probe.epochs == 36
    assert baseline.epochs == NO_KD_EPOCHS["S"]
    assert distillation.epochs == DISTILL_EPOCHS["S"]


@pytest.mark.parametrize("variant", ["A", "F", "P", "N", "T", "S", "M", "L", "X", "XL"])
def test_full_lineae_recipes_restore_linea_multiscale_training(variant):
    config = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    assert config.training_profile == "single_gpu_96gb"
    assert config.recipe_reference_effective_batch_size == 8
    assert config.multi_scale_train is True
    assert config.optimizer_fused is True


@pytest.mark.parametrize(
    "variant,batch_size,accumulation,epochs,initial_depth",
    [("2XL", 4, 2, 60, 4), ("3XL", 2, 4, 72, 6)],
)
def test_accuracy_tier_large_recipes_keep_effective_batch_and_xl_head(
    variant,
    batch_size,
    accumulation,
    epochs,
    initial_depth,
):
    config = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    assert config.training_profile == "single_gpu_96gb_accuracy"
    assert config.eval_spatial_size == (640, 640)
    assert config.multi_scale_train is True
    assert config.use_checkpoint is True
    assert config.use_ema is False
    assert config.dino_intermediate_layers == []
    assert config.distill_weight == 0.0
    assert config.distill_feature_weight == 0.0
    assert config.batch_size_train == batch_size
    assert config.gradient_accumulation_steps == accumulation
    assert batch_size * accumulation == 8
    assert config.recipe_reference_effective_batch_size == 8
    assert config.scheduler_step_unit == "optimizer"
    assert config.lr == 2e-4
    assert config.lr_scheduler == "cosine"
    assert config.min_lr == 1e-7
    assert config.use_warmup is False
    assert config.model_parameters[0]["lr"] == 1e-5
    assert config.model_parameters[1]["lr"] == 1e-5
    assert config.epochs == epochs
    assert config.backbone_trainable_layers == initial_depth
    assert config.initial_freeze_epochs == 5
    assert config.unfreeze_interval == 2
    assert config.in_channels_encoder == [256, 256, 256]
    assert config.hidden_dim == 256
    assert config.dec_layers == 6
    assert config.eval_idx == 5
    assert config.num_queries == 1100
    assert config.num_select == 300


@pytest.mark.parametrize("variant", ["A", "F", "P", "N", "T", "S", "M", "L", "X"])
def test_no_kd_and_kd_training_step_semantics_match(variant):
    baseline = SLConfig.fromfile(f"configs/lineae/lineae_{variant.lower()}.py")
    distillation = SLConfig.fromfile(f"configs/lineae/distill/lineae_{variant.lower()}.py")

    assert baseline.batch_size_train == distillation.batch_size_train
    assert baseline.gradient_accumulation_steps == distillation.gradient_accumulation_steps
    assert baseline.scheduler_step_unit == distillation.scheduler_step_unit
    assert baseline.multi_scale_train == distillation.multi_scale_train
    assert baseline.pin_memory == distillation.pin_memory
    assert baseline.prefetch_factor == distillation.prefetch_factor
    assert (
        baseline.multiprocessing_sharing_strategy
        == distillation.multiprocessing_sharing_strategy
    )


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_capacity_aware_epoch_budgets_match_configs_and_readme(variant):
    lower = variant.lower()
    no_kd_path = f"configs/lineae/lineae_{lower}.py"
    no_kd_epochs = SLConfig.fromfile(no_kd_path).epochs
    assert no_kd_epochs == NO_KD_EPOCHS[variant]

    no_kd_steps = f"{no_kd_epochs * 625:,}"
    if variant in {"XL", "2XL", "3XL"}:
        expected_row = (
            variant,
            f"`{no_kd_path}`",
            str(no_kd_epochs),
            no_kd_steps,
            "not applicable (supervised teacher)",
            "—",
            "—",
        )
    else:
        distill_path = f"configs/lineae/distill/lineae_{lower}.py"
        distill_epochs = SLConfig.fromfile(distill_path).epochs
        assert distill_epochs == DISTILL_EPOCHS[variant]
        expected_row = (
            variant,
            f"`{no_kd_path}`",
            str(no_kd_epochs),
            no_kd_steps,
            f"`{distill_path}`",
            str(distill_epochs),
            f"{distill_epochs * 625:,}",
        )
    assert expected_row in _readme_table_rows()


def test_readme_documents_complete_dino_unfreeze_schedule():
    rows = _readme_table_rows()
    expected = {
        ("0–4", "10–11", "2/12", "blocks 10 and 11"),
        ("5–6", "9–11", "3/12", "block 9"),
        ("7–8", "8–11", "4/12", "block 8"),
        ("9–10", "7–11", "5/12", "block 7"),
        ("11–12", "6–11", "6/12", "block 6"),
        ("13–14", "5–11", "7/12", "block 5"),
        ("15–16", "4–11", "8/12", "block 4"),
        ("17–18", "3–11", "9/12", "block 3"),
        ("19–20", "2–11", "10/12", "block 2"),
        ("21–22", "1–11", "11/12", "block 1"),
        (
            "23–final",
            "0–11",
            "12/12",
            "block 0 and every remaining DINO-core parameter",
        ),
    }
    assert expected <= rows


@pytest.mark.parametrize(
    "label,path",
    [
        ("S P0 probe", "configs/lineae/probes/lineae_s.py"),
        ("S no-KD", "configs/lineae/lineae_s.py"),
        ("S direct-XL KD", "configs/lineae/distill/lineae_s.py"),
        ("M no-KD", "configs/lineae/lineae_m.py"),
        ("M direct-XL KD", "configs/lineae/distill/lineae_m.py"),
        ("L no-KD", "configs/lineae/lineae_l.py"),
        ("L direct-XL KD", "configs/lineae/distill/lineae_l.py"),
        ("X no-KD", "configs/lineae/lineae_x.py"),
        ("X direct-XL KD", "configs/lineae/distill/lineae_x.py"),
        ("XL no-KD teacher", "configs/lineae/lineae_xl.py"),
    ],
)
def test_readme_documents_dino_recipe_fully_unfrozen_horizon(label, path):
    epochs = SLConfig.fromfile(path).epochs
    expected_row = (
        label,
        f"`{path}`",
        str(epochs),
        f"23–{epochs - 1}",
        str(epochs - 23),
    )
    assert expected_row in _readme_table_rows()


@pytest.mark.parametrize(
    "label,path,fully_unfrozen_epoch",
    [
        ("2XL no-KD teacher candidate", "configs/lineae/lineae_2xl.py", 43),
        ("3XL no-KD teacher candidate", "configs/lineae/lineae_3xl.py", 55),
    ],
)
def test_readme_documents_large_dino_fully_unfrozen_horizon(
    label,
    path,
    fully_unfrozen_epoch,
):
    epochs = SLConfig.fromfile(path).epochs
    expected_row = (
        label,
        f"`{path}`",
        str(epochs),
        f"{fully_unfrozen_epoch}–{epochs - 1}",
        str(epochs - fully_unfrozen_epoch),
    )
    assert expected_row in _readme_table_rows()


@pytest.mark.parametrize("variant", ["L", "M", "S", "N", "P", "F", "A"])
def test_x_teacher_cascade_configs_match_direct_xl_controls(variant):
    lower = variant.lower()
    direct = SLConfig.fromfile(f"configs/lineae/distill/lineae_{lower}.py")
    cascade = SLConfig.fromfile(f"configs/lineae/cascade/lineae_{lower}.py")

    assert validate_variant_config(cascade) == VARIANTS[variant]
    assert cascade.distill_weight == direct.distill_weight
    assert cascade.batch_size_train == direct.batch_size_train
    assert cascade.gradient_accumulation_steps == direct.gradient_accumulation_steps
    assert cascade.scheduler_step_unit == direct.scheduler_step_unit
    assert cascade.distill_teacher_config == "configs/lineae/lineae_x.py"
    assert cascade.distill_teacher_checkpoint == "ckpts/lineae_x_teacher.pth"


@pytest.mark.parametrize("key,candidate", TUNING_CANDIDATES.items())
def test_p4_tuning_configs_are_matched_feasible_direct_xl_screens(key, candidate):
    variant, profile = key
    lower = variant.lower()
    config = SLConfig.fromfile(
        f"configs/lineae/tuning/lineae_{lower}_{profile}.py"
    )
    direct = SLConfig.fromfile(f"configs/lineae/distill/lineae_{lower}.py")

    assert validate_variant_config(config) == VARIANTS[variant]
    assert config.training_profile == "single_gpu_96gb_tuning"
    assert tuple(config.eval_spatial_size) == (candidate.input_size,) * 2
    assert config.data_aug_scales == [tuple(config.eval_spatial_size)]
    assert config.num_queries == candidate.num_queries
    assert config.num_select == candidate.num_select
    assert config.dec_layers == candidate.decoder_layers
    assert config.eval_idx == candidate.decoder_layers - 1
    assert config.distill_weight == direct.distill_weight
    assert config.distill_teacher_config == direct.distill_teacher_config
    assert config.distill_teacher_checkpoint == direct.distill_teacher_checkpoint
    assert config.batch_size_train == direct.batch_size_train
    assert config.gradient_accumulation_steps == direct.gradient_accumulation_steps
    assert config.scheduler_step_unit == direct.scheduler_step_unit
    assert config.multi_scale_train == direct.multi_scale_train
    for size in generate_scales(candidate.input_size, 3):
        feature_positions = sum((size // stride) ** 2 for stride in (8, 16, 32))
        assert feature_positions >= candidate.num_queries
@pytest.mark.parametrize(
    "path,field,expected",
    [
        ("lineae_xl_ema.py", "use_ema", True),
        ("lineae_x_intermediate.py", "dino_intermediate_layers", [3, 7, 11]),
        ("lineae_xl_photometric.py", "use_photometric_distort", True),
        ("lineae_x_feature_kd.py", "distill_feature_weight", 1.0),
        ("lineae_xl_frozen.py", "backbone_trainable_layers", -1),
    ],
)
def test_independent_ablation_configs_load(path, field, expected):
    config = SLConfig.fromfile(f"configs/lineae/ablations/{path}")
    assert getattr(config, field) == expected
    assert config.training_profile == "single_gpu_96gb"


def test_qualified_teacher_artifact_binds_canonical_resolved_config():
    config_path = Path("configs/lineae/lineae_xl.py")
    baseline_config_path = Path("configs/linea/linea_hgnetv2_n.py")
    config = SLConfig.fromfile(str(config_path))
    baseline_config = SLConfig.fromfile(str(baseline_config_path))
    source_hash = "c" * 64
    metrics = {
        "sap5": 49.0,
        "sap10": 50.0,
        "sap15": 51.0,
        "deploy_sap5": 49.0,
        "deploy_sap10": 50.0,
        "deploy_sap15": 51.0,
    }
    candidate = {
        "format": "lineae_evaluation_v3",
        "image_preprocess_schema": "opencv_rgb_inter_linear_v2",
        "opencv_version": "4.13.0",
        "checkpoint_sha256": source_hash,
        "config": str(config_path.resolve()),
        "num_queries": 1100,
        "num_select": 300,
        "sap_protocol": "official_all_queries_and_deployment_topk",
        "datasets": {
            dataset: {
                **metrics,
                "annotation_sha256": "a" * 64,
                "samples": 10,
            }
            for dataset in ("wireframe", "york")
        },
    }
    baseline = {
        "format": "lineae_evaluation_v3",
        "image_preprocess_schema": "opencv_rgb_inter_linear_v2",
        "opencv_version": "4.13.0",
        "checkpoint_sha256": "b" * 64,
        "config": str(baseline_config_path.resolve()),
        "num_queries": 1100,
        "num_select": 300,
        "sap_protocol": "official_all_queries_and_deployment_topk",
        "datasets": {
            dataset: {
                "sap5": 39.0,
                "sap10": 40.0,
                "sap15": 41.0,
                "deploy_sap5": 39.0,
                "deploy_sap10": 40.0,
                "deploy_sap15": 41.0,
                "annotation_sha256": "a" * 64,
                "samples": 10,
            }
            for dataset in ("wireframe", "york")
        },
    }
    artifact = {
        "format": "lineae_teacher_v3",
        "image_preprocess_schema": "opencv_rgb_inter_linear_v2",
        "opencv_version": "4.13.0",
        "variant": "XL",
        "model": {"weight": torch.zeros(1)},
        "source_checkpoint_sha256": source_hash,
        "source_config": str(config_path.resolve()),
        "source_config_file_sha256": sha256_file(config_path),
        "source_config_sha256": config_fingerprint(config),
        "baseline_checkpoint_sha256": "b" * 64,
        "baseline_config": str(baseline_config_path.resolve()),
        "baseline_config_file_sha256": sha256_file(baseline_config_path),
        "baseline_config_sha256": config_fingerprint(baseline_config),
        "inference_config": str(config_path.resolve()),
        "inference_config_file_sha256": sha256_file(config_path),
        "inference_config_sha256": config_fingerprint(config),
        "qualification": {
            "candidate": candidate,
            "baseline": baseline,
            "minimum_ap10_gain": 0.0,
            "reload_identical": True,
            "canonical_inference_identical": True,
        },
    }
    validate_teacher_artifact(artifact, config_path)
    with pytest.raises(ValueError, match="artifact variant mismatch"):
        validate_teacher_artifact({**artifact, "variant": "X"}, config_path)
    stale = {**artifact, "inference_config_sha256": "0" * 64}
    with pytest.raises(ValueError, match="config hash mismatch"):
        validate_teacher_artifact(stale, config_path)
    with pytest.raises(ValueError, match="qualified lineae_teacher_v3"):
        validate_teacher_artifact({"model": {}}, config_path)
    forged = {
        **artifact,
        "qualification": {
            **artifact["qualification"],
            "candidate": {
                **candidate,
                "datasets": {
                    **candidate["datasets"],
                    "wireframe": {
                        **candidate["datasets"]["wireframe"],
                        "sap10": 39.0,
                    },
                },
            },
        },
    }
    with pytest.raises(ValueError, match="no longer satisfies"):
        validate_teacher_artifact(forged, config_path)
