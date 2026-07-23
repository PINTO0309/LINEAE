import json
import random
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from datasets import build_dataset, resolve_ensemble_training_sources
from datasets.collate import BatchImageCollateFunction, encoder_token_count
from datasets.coco import make_coco_transforms
from datasets.transforms import ColorJitter, Normalize, crop, hflip, resize, rotation
from main import (
    build_training_data_loader,
    configure_multiprocessing_sharing,
    create,
    data_loader_options,
    resolve_training_horizon,
)
from util.slconfig import SLConfig


def _args(**overrides):
    values = dict(
        data_aug_scales=[(64, 64)],
        data_aug_max_size=1333,
        data_aug_scales2_resize=[32, 48],
        data_aug_scales2_crop=[24, 48],
        eval_spatial_size=(64, 64),
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        image_preprocess_schema="opencv_rgb_inter_linear_v2",
        coco_path="data/wireframe_processed",
        use_lmap=False,
        batch_size_train=1,
        batch_size_val=1,
        use_photometric_distort=False,
        photometric_distort_probability=0.5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_coco_split(root, split, image_count, *, write_images=True):
    image_dir = root / f"{split}2017"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for index in range(image_count):
        file_name = f"{split}_{index}.png"
        images.append({
            "id": index,
            "file_name": file_name,
            "height": 16,
            "width": 16,
        })
        annotations.append({
            "id": index,
            "image_id": index,
            "category_id": 0,
            "line": [1.0, 2.0, 10.0, 8.0],
            "area": 1,
        })
        if write_images:
            assert cv2.imwrite(
                str(image_dir / file_name),
                np.full((16, 16, 3), 127, dtype=np.uint8),
            )
    annotation = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "line"}],
    }
    (annotation_dir / f"lines_{split}2017.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )


def test_dino_normalization_profile_is_applied_once():
    args = _args()
    transform = make_coco_transforms("val", args)
    image, _ = transform(np.zeros((64, 64, 3), dtype=np.uint8), None)
    expected = -torch.tensor(args.image_mean) / torch.tensor(args.image_std)
    assert torch.allclose(image[:, 0, 0], expected, atol=1e-6)


def test_normalize_clamps_boundary_roundoff_to_line_coordinate_contract():
    transform = Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    image = torch.zeros(3, 64, 64)
    _, target = transform(
        image,
        {"lines": torch.tensor([[-1e-12, 0.0, 64.0 + 1e-12, 64.0]])},
    )
    assert ((target["lines"] >= 0.0) & (target["lines"] <= 1.0)).all()


def test_crop_range_comes_from_config():
    args = _args(data_aug_scales2_crop=[17, 29])
    transform = make_coco_transforms("train", args)
    crop = transform.transforms[1].transforms2.transforms[1]
    assert crop.min_size == 17
    assert crop.max_size == 29


def test_optional_photometric_distortion_never_changes_line_targets():
    args = _args(use_photometric_distort=True, photometric_distort_probability=1.0)
    transform = make_coco_transforms("train", args)
    photometric = transform.transforms[2]
    target = {"lines": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
    image = np.full((64, 64, 3), (80, 120, 160), dtype=np.uint8)
    output, actual_target = photometric(image, target)
    assert output.shape == image.shape
    assert actual_target is target
    assert torch.equal(actual_target["lines"], target["lines"])


def test_real_wireframe_photometric_ablation_preserves_geometry_and_trains_xl():
    def config_with_photometric(enabled):
        config = SLConfig.fromfile("configs/lineae/ablations/lineae_xl_photometric.py")
        config.pretrained = False
        config.eval_spatial_size = (64, 64)
        config.enforce_variant_input = False
        config.num_queries = 20
        config.num_select = 10
        config.dn_number = 4
        config.data_aug_scales = [(64, 64)]
        config.data_aug_scales2_resize = [32, 48]
        config.data_aug_scales2_crop = [24, 48]
        config.batch_size_train = 1
        config.batch_size_val = 1
        config.coco_path = "data/wireframe_processed"
        config.use_photometric_distort = enabled
        config.photometric_distort_probability = 1.0
        return config

    def seeded_sample(dataset):
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        return dataset[0]

    control_config = config_with_photometric(False)
    photo_config = config_with_photometric(True)
    control_dataset = build_dataset("train", control_config)
    photo_dataset = build_dataset("train", photo_config)
    control_image, control_target = seeded_sample(control_dataset)
    photo_image, photo_target = seeded_sample(photo_dataset)
    repeated_image, repeated_target = seeded_sample(photo_dataset)

    assert not torch.equal(control_image, photo_image)
    assert torch.equal(photo_image, repeated_image)
    for key in ("lines", "labels", "area", "iscrowd", "size", "orig_size", "image_id"):
        assert torch.equal(control_target[key], photo_target[key])
        assert torch.equal(photo_target[key], repeated_target[key])
    assert photo_target["lines"].shape[0] > 0
    assert torch.isfinite(photo_image).all()
    assert torch.isfinite(photo_target["lines"]).all()
    assert ((photo_target["lines"] >= 0) & (photo_target["lines"] <= 1)).all()

    images, targets = BatchImageCollateFunction(base_size=64)([
        (photo_image, photo_target),
    ])
    model, _ = create(photo_config, "modelname")
    criterion = create(photo_config, "criterionname")
    model.train()
    criterion.train()
    tracked = [
        parameter
        for name, parameter in model.named_parameters()
        if "decoder.class_embed" in name
        and name.endswith(".weight")
        and parameter.requires_grad
    ][-1]
    optimizer = torch.optim.SGD([tracked], lr=1e-3)
    before = tracked.detach().clone()

    outputs = model(images, targets)
    total = sum(criterion(outputs, targets).values())
    model.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()

    assert outputs["pred_logits"].shape == (1, 20, 2)
    assert outputs["pred_lines"].shape == (1, 20, 4)
    assert torch.isfinite(total)
    assert tracked.grad is not None and torch.isfinite(tracked.grad).all()
    assert not torch.equal(before, tracked.detach())


def test_copied_wireframe_dataset_is_readable():
    args = _args(
        batch_size_train=8,
        multi_scale_train=False,
        num_queries=20,
        feat_strides=[8, 16, 32],
    )
    train_dataset = build_dataset("train", args)
    dataset = build_dataset("val", args)
    assert len(train_dataset) == 5000
    assert not isinstance(train_dataset, torch.utils.data.ConcatDataset)
    assert len(dataset) == 462
    image, target = dataset[0]
    assert image.shape == (3, 64, 64)
    assert target["lines"].ndim == 2 and target["lines"].shape[1] == 4
    assert torch.isfinite(image).all()
    assert torch.isfinite(target["lines"]).all()
    assert ((target["lines"] >= 0) & (target["lines"] <= 1)).all()
    sampler = torch.utils.data.SequentialSampler(train_dataset)
    loader = build_training_data_loader(
        train_dataset,
        sampler,
        args,
        {"num_workers": 0, "pin_memory": False},
    )
    assert loader.drop_last is True
    assert len(loader) == 625


def test_real_ensemble_dataset_includes_all_york_splits_and_keeps_wireframe_val():
    args = _args(
        ensemble=True,
        ensemble_york_path="data/york_processed",
        batch_size_train=8,
        multi_scale_train=False,
        num_queries=20,
        feat_strides=[8, 16, 32],
    )
    sources = resolve_ensemble_training_sources(args)
    train_dataset = build_dataset("train", args)
    val_dataset = build_dataset("val", args)

    assert [(source["name"], source["samples"]) for source in sources] == [
        ("york_train", 0),
        ("york_val", 102),
    ]
    assert len(train_dataset) == 5102
    assert [len(dataset) for dataset in train_dataset.datasets] == [5000, 102]
    assert len(val_dataset) == 462
    assert args.ensemble_split_samples == {"train": 0, "val": 102}
    assert args.ensemble_training_sample_count == 102
    assert args.training_dataset_sample_count == 5102

    sampler = torch.utils.data.RandomSampler(
        train_dataset,
        generator=torch.Generator().manual_seed(42),
    )
    loader = build_training_data_loader(
        train_dataset,
        sampler,
        args,
        {"num_workers": 0, "pin_memory": False},
    )
    assert loader.drop_last is False
    assert len(loader) == 638


def test_real_york_ensemble_sample_runs_finite_forward_and_backward():
    torch.manual_seed(109)
    random.seed(109)
    np.random.seed(109)
    config = SLConfig.fromfile("configs/lineae/lineae_n.py")
    config.pretrained = False
    config.eval_spatial_size = (64, 64)
    config.enforce_variant_input = False
    config.num_queries = 20
    config.num_select = 10
    config.dn_number = 4
    config.data_aug_scales = [(64, 64)]
    config.data_aug_scales2_resize = [32, 48]
    config.data_aug_scales2_crop = [24, 48]
    config.batch_size_train = 1
    config.batch_size_val = 1
    config.coco_path = "data/wireframe_processed"
    config.ensemble = True
    config.ensemble_york_path = "data/york_processed"
    resolve_ensemble_training_sources(config)
    dataset = build_dataset("train", config)
    image, target = dataset[5000]
    images, targets = BatchImageCollateFunction(base_size=64)([(image, target)])
    model, _ = create(config, "modelname")
    criterion = create(config, "criterionname")

    outputs = model(images, targets)
    loss = sum(criterion(outputs, targets).values())
    loss.backward()

    assert outputs["pred_logits"].shape == (1, 20, 2)
    assert outputs["pred_lines"].shape == (1, 20, 4)
    assert target["lines"].shape[0] > 0
    assert torch.isfinite(target["lines"]).all()
    assert torch.isfinite(loss)


def test_ensemble_source_preflight_rejects_invalid_inputs(tmp_path):
    args = _args(
        ensemble=True,
        ensemble_york_path=str(tmp_path / "missing"),
    )
    with pytest.raises(FileNotFoundError, match="image directory"):
        resolve_ensemble_training_sources(args)

    empty_root = tmp_path / "empty"
    _write_coco_split(empty_root, "train", 0)
    _write_coco_split(empty_root, "val", 0)
    args.ensemble_york_path = str(empty_root)
    with pytest.raises(ValueError, match="contains no images"):
        resolve_ensemble_training_sources(args)

    missing_image_root = tmp_path / "missing-image"
    _write_coco_split(missing_image_root, "train", 0)
    _write_coco_split(missing_image_root, "val", 1, write_images=False)
    args.ensemble_york_path = str(missing_image_root)
    with pytest.raises(FileNotFoundError, match="references missing images"):
        resolve_ensemble_training_sources(args)

    valid_root = tmp_path / "valid"
    _write_coco_split(valid_root, "train", 0)
    _write_coco_split(valid_root, "val", 1)
    args.ensemble_york_path = str(valid_root)
    args.use_lmap = True
    with pytest.raises(ValueError, match="use_lmap=True"):
        resolve_ensemble_training_sources(args)


def test_crop_clips_horizontal_vertical_and_rejects_outside_or_short_lines():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    target = {
        "lines": torch.tensor([
            [-5.0, 10.0, 20.0, 10.0],
            [10.0, -5.0, 10.0, 20.0],
            [-5.0, -5.0, -1.0, -1.0],
            [2.0, 2.0, 3.0, 3.0],
        ]),
        "labels": torch.arange(4),
        "area": torch.ones(4),
        "iscrowd": torch.zeros(4),
    }
    _, clipped = crop(image, target, (0, 0, 16, 16))
    assert torch.equal(clipped["labels"], torch.tensor([0, 1]))
    assert torch.allclose(clipped["lines"], torch.tensor([
        [0.0, 10.0, 15.0, 10.0],
        [10.0, 0.0, 10.0, 15.0],
    ]))
    assert torch.isfinite(clipped["lines"]).all()


def test_opencv_geometry_keeps_image_and_line_coordinates_aligned():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    target = {"lines": torch.tensor([[1.0, 1.0, 4.0, 2.0]])}

    flipped_image, flipped_target = hflip(image, target)
    assert np.array_equal(flipped_image, cv2.flip(image, 1))
    torch.testing.assert_close(
        flipped_target["lines"], torch.tensor([[2.0, 2.0, 5.0, 1.0]])
    )

    rotated_image, rotated_target = rotation(image, target, 1)
    assert np.array_equal(rotated_image, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    torch.testing.assert_close(
        rotated_target["lines"], torch.tensor([[1.0, 4.0, 2.0, 1.0]])
    )
    assert torch.equal(rotated_target["size"], torch.tensor([6, 4]))

    resized_image, resized_target = resize(image, target, (3, 2))
    assert np.array_equal(
        resized_image,
        cv2.resize(image, (3, 2), interpolation=cv2.INTER_LINEAR),
    )
    torch.testing.assert_close(
        resized_target["lines"], torch.tensor([[0.5, 0.5, 2.0, 1.0]])
    )


def test_opencv_color_jitter_is_seed_reproducible():
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    jitter = ColorJitter()
    torch.manual_seed(73)
    first, _ = jitter(image, None)
    torch.manual_seed(73)
    second, _ = jitter(image, None)
    assert np.array_equal(first, second)


def test_batch_multiscale_resizes_pixels_but_preserves_normalized_lines(monkeypatch):
    collate = BatchImageCollateFunction(base_size=64, base_size_repeat=3)
    monkeypatch.setattr(random, "choice", lambda _values: 96)
    normalized_lines = torch.tensor([[0.1, 0.2, 0.8, 0.9]])
    source = torch.arange(3 * 64 * 64, dtype=torch.float32).reshape(3, 64, 64)
    items = [
        (
            source.clone(),
            {"lines": normalized_lines.clone(), "labels": torch.tensor([0])},
        )
        for _ in range(2)
    ]

    images, targets = collate(items)

    assert images.shape == (2, 3, 96, 96)
    expected = cv2.resize(
        source.permute(1, 2, 0).numpy(),
        (96, 96),
        interpolation=cv2.INTER_LINEAR,
    )
    torch.testing.assert_close(
        images[0],
        torch.from_numpy(expected).permute(2, 0, 1),
        rtol=1e-5,
        atol=5e-6,
    )
    assert all(torch.equal(target["lines"], normalized_lines) for target in targets)


def test_a_multiscale_supports_600_queries_at_every_generated_resolution():
    collate = BatchImageCollateFunction(
        base_size=320,
        base_size_repeat=3,
        minimum_tokens=600,
        feature_strides=(8, 16, 32),
    )

    assert set(collate.scales) == {224, 256, 288, 320, 352, 384}
    assert all(encoder_token_count(scale) >= 600 for scale in collate.scales)


def test_data_loader_options_enable_pinned_prefetch_only_when_applicable():
    args = SimpleNamespace(num_workers=8, pin_memory=True, prefetch_factor=4)
    assert data_loader_options(args, torch.device("cuda")) == {
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
    }
    assert data_loader_options(args, torch.device("cpu"))["pin_memory"] is False
    args.num_workers = 0
    assert data_loader_options(args, torch.device("cuda")) == {
        "num_workers": 0,
        "pin_memory": True,
    }
    args.num_workers = 1
    args.prefetch_factor = 0
    with pytest.raises(ValueError, match="prefetch_factor must be positive"):
        data_loader_options(args, torch.device("cuda"))


def test_multiprocessing_tensor_sharing_uses_configured_strategy(monkeypatch):
    selected = []
    monkeypatch.setattr(
        torch.multiprocessing,
        "get_all_sharing_strategies",
        lambda: {"file_descriptor", "file_system"},
    )
    monkeypatch.setattr(
        torch.multiprocessing,
        "set_sharing_strategy",
        selected.append,
    )

    args = SimpleNamespace(
        num_workers=8,
        multiprocessing_sharing_strategy="file_system",
    )
    assert configure_multiprocessing_sharing(args) == "file_system"
    assert selected == ["file_system"]

    args.num_workers = 0
    assert configure_multiprocessing_sharing(args) is None
    assert selected == ["file_system"]

    args.num_workers = 1
    args.multiprocessing_sharing_strategy = "unsupported"
    with pytest.raises(ValueError, match="unsupported multiprocessing_sharing_strategy"):
        configure_multiprocessing_sharing(args)


def test_training_horizon_reports_unscaled_large_batch_recipe():
    args = SimpleNamespace(
        batch_size_train=64,
        world_size=1,
        gradient_accumulation_steps=1,
        recipe_reference_effective_batch_size=8,
        epochs=36,
    )

    assert resolve_training_horizon(args, 78) == {
        "effective_batch_size": 64,
        "reference_effective_batch_size": 8,
        "optimizer_steps_per_epoch": 78,
        "total_optimizer_steps": 2808,
        "batch_scale": 8.0,
    }

    args.recipe_reference_effective_batch_size = 0
    with pytest.raises(ValueError, match="must be positive"):
        resolve_training_horizon(args, 78)
