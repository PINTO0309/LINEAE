import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qualify_teacher import _load_compatible_teacher_configs, qualify
from tools.qualify_best_teacher import Candidate, select_candidate
from util.experiment import sha256_file


def _report(
    checkpoint_hash,
    sap10,
    *,
    annotation_hash="a" * 64,
    samples=10,
    config="configs/lineae/lineae_xl.py",
):
    metrics = {"sap5": sap10 - 1, "sap10": sap10, "sap15": sap10 + 1}
    return {
        "format": "lineae_evaluation_v1",
        "checkpoint_sha256": checkpoint_hash,
        "config": str(Path(config).resolve()),
        "datasets": {
            "wireframe": {
                **metrics,
                "annotation_sha256": annotation_hash,
                "samples": samples,
            },
            "york": {
                **metrics,
                "annotation_sha256": annotation_hash,
                "samples": samples,
            },
        },
    }


def _baseline_report(checkpoint_hash, sap10, **kwargs):
    return _report(
        checkpoint_hash,
        sap10,
        config="configs/linea/linea_hgnetv2_n.py",
        **kwargs,
    )


def test_teacher_configs_allow_same_non_xl_variant_and_reject_mixed_variants():
    _, spec, _ = _load_compatible_teacher_configs(
        Path("configs/lineae/distill/lineae_x.py"),
        Path("configs/lineae/lineae_x.py"),
    )
    assert spec.name == "X"
    with pytest.raises(ValueError, match="different variants"):
        _load_compatible_teacher_configs(
            Path("configs/lineae/distill/lineae_x.py"),
            Path("configs/lineae/lineae_l.py"),
        )


def test_teacher_promotion_rejects_candidate_that_does_not_beat_baseline(tmp_path):
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(b"not loaded because the metric gate runs first")
    candidate_hash = sha256_file(candidate)
    candidate_metrics = tmp_path / "candidate.json"
    baseline_metrics = tmp_path / "baseline.json"
    candidate_metrics.write_text(json.dumps(_report(candidate_hash, 50.0)))
    baseline_metrics.write_text(json.dumps(_baseline_report("b" * 64, 50.0)))
    args = SimpleNamespace(
        output=tmp_path / "teacher.pth",
        force=False,
        candidate=candidate,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        minimum_ap10_gain=0.0,
        config="configs/lineae/lineae_xl.py",
        device="cpu",
    )
    with pytest.raises(ValueError, match="does not exceed baseline"):
        qualify(args)


def test_teacher_promotion_rejects_negative_minimum_gain(tmp_path):
    args = SimpleNamespace(
        output=tmp_path / "teacher.pth",
        force=False,
        minimum_ap10_gain=-0.1,
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        qualify(args)


def test_teacher_promotion_rejects_stale_metric_report(tmp_path):
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(b"candidate")
    candidate_metrics = tmp_path / "candidate.json"
    baseline_metrics = tmp_path / "baseline.json"
    candidate_metrics.write_text(json.dumps(_report("d" * 64, 60.0)))
    baseline_metrics.write_text(json.dumps(_baseline_report("b" * 64, 50.0)))
    args = SimpleNamespace(
        output=tmp_path / "teacher.pth",
        force=False,
        candidate=candidate,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        minimum_ap10_gain=0.0,
        config="configs/lineae/lineae_xl.py",
        device="cpu",
    )
    with pytest.raises(ValueError, match="different checkpoint"):
        qualify(args)


def test_teacher_promotion_rejects_mismatched_evaluation_dataset(tmp_path):
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(b"candidate")
    candidate_metrics = tmp_path / "candidate.json"
    baseline_metrics = tmp_path / "baseline.json"
    candidate_metrics.write_text(json.dumps(_report(sha256_file(candidate), 60.0)))
    baseline_metrics.write_text(
        json.dumps(_baseline_report("b" * 64, 50.0, annotation_hash="c" * 64))
    )
    args = SimpleNamespace(
        output=tmp_path / "teacher.pth",
        force=False,
        candidate=candidate,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        minimum_ap10_gain=0.0,
        config="configs/lineae/lineae_xl.py",
        device="cpu",
    )
    with pytest.raises(ValueError, match="different wireframe.annotation_sha256"):
        qualify(args)


def test_teacher_promotion_rejects_wrong_baseline_recipe(tmp_path):
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(b"candidate")
    candidate_metrics = tmp_path / "candidate.json"
    baseline_metrics = tmp_path / "baseline.json"
    candidate_metrics.write_text(json.dumps(_report(sha256_file(candidate), 60.0)))
    baseline_metrics.write_text(json.dumps(_report("b" * 64, 50.0)))
    args = SimpleNamespace(
        output=tmp_path / "teacher.pth",
        force=False,
        candidate=candidate,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        minimum_ap10_gain=0.0,
        config="configs/lineae/lineae_xl.py",
        device="cpu",
    )
    with pytest.raises(ValueError, match="baseline metrics were produced with a different"):
        qualify(args)


def test_best_xl_recipe_is_selected_from_matched_progressive_and_frozen_runs(tmp_path):
    progressive = tmp_path / "progressive.pth"
    frozen = tmp_path / "frozen.pth"
    progressive.write_bytes(b"progressive")
    frozen.write_bytes(b"frozen")
    progressive_report = tmp_path / "progressive.json"
    frozen_report = tmp_path / "frozen.json"
    baseline_report = tmp_path / "baseline.json"
    progressive_report.write_text(json.dumps(_report(
        sha256_file(progressive),
        61.0,
        config="configs/lineae/lineae_xl.py",
    )))
    frozen_report.write_text(json.dumps(_report(
        sha256_file(frozen),
        59.0,
        config="configs/lineae/ablations/lineae_xl_frozen.py",
    )))
    baseline_report.write_text(json.dumps(_baseline_report("b" * 64, 55.0)))
    selected = select_candidate([
        Candidate(Path("configs/lineae/lineae_xl.py"), progressive, progressive_report),
        Candidate(
            Path("configs/lineae/ablations/lineae_xl_frozen.py"),
            frozen,
            frozen_report,
        ),
    ], baseline_report)
    assert selected.checkpoint == progressive


def test_teacher_promotion_rejects_stale_baseline_metric_report(tmp_path):
    candidate = tmp_path / "candidate.pth"
    baseline = tmp_path / "baseline.pth"
    candidate.write_bytes(b"candidate")
    baseline.write_bytes(b"baseline")
    candidate_metrics = tmp_path / "candidate.json"
    baseline_metrics = tmp_path / "baseline.json"
    candidate_metrics.write_text(json.dumps(_report(sha256_file(candidate), 60.0)))
    baseline_metrics.write_text(json.dumps(_baseline_report("b" * 64, 50.0)))
    args = SimpleNamespace(
        output=tmp_path / "teacher.pth",
        force=False,
        candidate=candidate,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        baseline_checkpoint=baseline,
        minimum_ap10_gain=0.0,
        config="configs/lineae/lineae_xl.py",
        device="cpu",
    )
    with pytest.raises(ValueError, match="baseline metrics were produced from a different"):
        qualify(args)


def test_teacher_promotion_rejects_partial_epoch_checkpoint(tmp_path):
    candidate = tmp_path / "candidate.pth"
    baseline = tmp_path / "baseline.pth"
    torch.save({"epoch_complete": False, "model": {}}, candidate)
    baseline.write_bytes(b"baseline")
    candidate_metrics = tmp_path / "candidate.json"
    baseline_metrics = tmp_path / "baseline.json"
    candidate_metrics.write_text(
        json.dumps(_report(sha256_file(candidate), 60.0)), encoding="utf-8"
    )
    baseline_metrics.write_text(
        json.dumps(_baseline_report(sha256_file(baseline), 50.0)), encoding="utf-8"
    )
    args = SimpleNamespace(
        output=tmp_path / "teacher.pth",
        force=False,
        candidate=candidate,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        baseline_checkpoint=baseline,
        minimum_ap10_gain=0.0,
        config="configs/lineae/lineae_xl.py",
        device="cpu",
    )

    with pytest.raises(ValueError, match="partial-epoch checkpoint"):
        qualify(args)
