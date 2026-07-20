import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from engine import train_one_epoch
from tools.compare_training_profiles import compare
from util.artifact_validation import validate_training_profile_report


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.backbone = SimpleNamespace(last_feature_shapes=[(2, 1, 1, 1)])

    def forward(self, samples, targets):
        del targets
        batch_size = samples.shape[0]
        value = self.weight * samples.mean(dim=(1, 2, 3))
        return {
            "pred_logits": value[:, None, None].expand(batch_size, 2, 2),
            "pred_lines": value[:, None, None].expand(batch_size, 2, 4),
        }


class _TinyCriterion(torch.nn.Module):
    def forward(self, outputs, targets):
        del targets
        return {"loss_test": outputs["pred_logits"].square().mean()}


def test_train_one_epoch_profile_callback_is_opt_in_and_bounded():
    model = _TinyModel()
    criterion = _TinyCriterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        gradient_accumulation_steps=1,
        verify_optimizer_step=False,
        pin_memory=False,
        output_dir="",
        use_ema=False,
        scheduler_step_unit="epoch",
    )
    targets = [{"labels": torch.tensor([0]), "lines": torch.zeros(1, 4)} for _ in range(2)]
    loader = [(torch.ones(2, 3, 4, 4), targets) for _ in range(3)]
    samples = []

    _, completed, epoch_complete = train_one_epoch(
        model,
        criterion,
        loader,
        optimizer,
        torch.device("cpu"),
        epoch=0,
        args=args,
        max_steps=2,
        start_global_step=7,
        step_profile_callback=samples.append,
    )

    assert completed == 2
    assert epoch_complete is False
    assert len(samples) == 2
    assert [sample["global_step"] for sample in samples] == [7, 8]
    assert all(sample["optimizer_stepped"] for sample in samples)
    assert all(sample["input_size"] == [4, 4] for sample in samples)
    assert all(sample["peak_memory_mib"] is None for sample in samples)
    assert all(sample["online_kd_ms"] == 0 for sample in samples)
    assert all(sample["step_ms"] > 0 for sample in samples)


def _summary(values):
    return {
        "mean": sum(values) / len(values),
        "p50": sum(values) / len(values),
        "p95": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def _profile(*, distilled: bool) -> dict:
    iterations = 2
    batch_size = 2
    step_ms = 15.0 if distilled else 10.0
    teacher_ms = 3.0 if distilled else 0.0
    kd_loss_ms = 2.0 if distilled else 0.0
    phases = {
        "transfer_ms": 1.0,
        "student_supervised_ms": 3.0,
        "teacher_forward_ms": teacher_ms,
        "kd_loss_ms": kd_loss_ms,
        "online_kd_ms": teacher_ms + kd_loss_ms,
        "backward_ms": 4.0,
        "optimizer_ms": 2.0,
        "step_ms": step_ms,
        "throughput_images_per_second": batch_size * 1000.0 / step_ms,
    }
    samples = []
    for index in range(iterations):
        samples.append({
            **phases,
            "peak_memory_mib": 120.0 if distilled else 100.0,
            "batch_size": batch_size,
            "input_size": [640, 640],
            "optimizer_stepped": True,
            "global_step": 101 + index,
            "loss": 1.0,
            "kd_matches": 10 if distilled else None,
            "kd_weight": 1.0 if distilled else 0.0,
            "kd_temperature": 2.0 if distilled else None,
        })
    return {
        "format": "lineae_training_profile_v1",
        "config": f"/configs/{'kd' if distilled else 'baseline'}.py",
        "resolved_config_sha256": ("c" if distilled else "b") * 64,
        "variant": "S",
        "dataset": {
            "root": "/data",
            "annotation_sha256": "a" * 64,
            "samples": 100,
        },
        "initialization": {"path": "/ckpts/s.pt", "sha256": "d" * 64},
        "teacher": (
            {"path": "/ckpts/xl.pth", "sha256": "e" * 64}
            if distilled
            else None
        ),
        "device": "cuda:0",
        "gpu": "test GPU",
        "python": "3.10.0",
        "torch": "2.9.0",
        "cuda_runtime": "13.0",
        "cudnn": 91000,
        "amp": True,
        "seed": 42,
        "epoch": 1,
        "start_global_step": 100,
        "warmup": 1,
        "iterations": iterations,
        "batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer_fused": True,
        "trainable_depth": 2,
        "trainable_parameters": 1000,
        "distill_temperature_steps_resolved": 1000 if distilled else None,
        "phase_ms": {
            name: _summary([value] * iterations)
            for name, value in phases.items()
        },
        "peak_memory_mib": _summary([
            120.0 if distilled else 100.0
        ] * iterations),
        "online_kd_fraction": _summary([
            (teacher_ms + kd_loss_ms) / step_ms
        ] * iterations),
        "samples": samples,
    }


def test_training_profile_schema_binds_summaries_to_raw_steps():
    report = _profile(distilled=True)
    validate_training_profile_report(report, require_cuda=True)
    report["phase_ms"]["step_ms"]["mean"] += 1
    with pytest.raises(ValueError, match="does not match raw samples"):
        validate_training_profile_report(report, require_cuda=True)


def test_compare_training_profiles_reports_online_teacher_overhead(tmp_path):
    baseline = _profile(distilled=False)
    distilled = _profile(distilled=True)
    baseline_path = tmp_path / "baseline.json"
    distilled_path = tmp_path / "distilled.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    distilled_path.write_text(json.dumps(distilled), encoding="utf-8")

    report = compare(baseline_path, distilled_path)

    assert report["direct_online_kd_ms"] == 5.0
    assert report["observed_step_overhead_ms"] == 5.0
    assert report["observed_step_overhead_percent"] == 50.0
    assert report["peak_memory_delta_mib"] == 20.0
    assert math.isclose(report["throughput_ratio"], 2 / 3)

    mismatch = copy.deepcopy(distilled)
    mismatch["samples"][1]["input_size"] = [768, 768]
    mismatch["phase_ms"] = distilled["phase_ms"]
    distilled_path.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(ValueError, match="input-size sequence"):
        compare(baseline_path, distilled_path)
