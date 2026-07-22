import json
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.analyze_pareto import analyze
from tools.analyze_repeated_pareto import (
    RepeatedRun,
    analyze as analyze_repeated,
    run_argument,
)
from tools.generate_model_card import generate
from tools.evaluate_onnx import compare_ap, evaluate as evaluate_onnx
from tools.plan_experiment_matrix import Task
from util.experiment import sha256_file

_PREPROCESSING = {
    "image_preprocess_schema": "opencv_rgb_inter_linear_v2",
    "opencv_version": "4.13.0",
}


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _datasets(sap10):
    return {
        dataset: {
            "annotation_sha256": "d" * 64,
            "samples": 10,
            "sap5": sap10 - 1,
            "sap10": sap10,
            "sap15": sap10 + 1,
            "deploy_sap5": sap10 - 1,
            "deploy_sap10": sap10,
            "deploy_sap15": sap10 + 1,
        }
        for dataset in ("wireframe", "york")
    }


def _evaluation(checkpoint_hash, sap10):
    return {
        **_PREPROCESSING,
        "format": "lineae_evaluation_v3",
        "config": "/test/config.py",
        "checkpoint_sha256": checkpoint_hash,
        "num_queries": 1100,
        "num_select": 300,
        "sap_protocol": "official_all_queries_and_deployment_topk",
        "datasets": _datasets(sap10),
    }


def _torch_benchmark(checkpoint_hash, *, p50, peak_memory, parameters):
    return {
        **_PREPROCESSING,
        "format": "lineae_torch_benchmark_v1",
        "config": "/test/config.py",
        "checkpoint_sha256": checkpoint_hash,
        "device": "cuda",
        "gpu": "test GPU",
        "batch_size": 1,
        "deploy_mode": True,
        "num_select": 300,
        "spatial_size": 640,
        "parameters": parameters,
        "latency_ms": {"p50": p50, "p95": p50},
        "peak_memory_mib": peak_memory,
        "amp": True,
        "warmup": 1,
        "iterations": 2,
        "samples_ms": [p50, p50],
    }


def _repeated_run(
    root,
    *,
    label,
    seed,
    sap10,
    p50,
    peak_memory,
    parameters,
    teacher_hash="e" * 64,
):
    run_dir = root / label / f"seed{seed}"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint_best.pth"
    checkpoint.write_bytes(f"{label}:{seed}".encode())
    checkpoint_hash = sha256_file(checkpoint)
    _write_json(run_dir / "run_complete.json", {
        "format": "lineae_run_completion_v1",
        "status": "full",
        "best_checkpoint_sha256": checkpoint_hash,
    })
    _write_json(run_dir / "run_manifest.json", {
        "config_file": "/test/config.py",
        "training": {"seed": seed},
        "checkpoints": {
            "teacher": {
                "exists": True,
                "sha256": teacher_hash,
            },
        },
    })
    evaluation = _write_json(
        run_dir / "evaluation.json",
        _evaluation(checkpoint_hash, sap10),
    )
    benchmark = _write_json(
        run_dir / "benchmark.json",
        _torch_benchmark(
            checkpoint_hash,
            p50=p50,
            peak_memory=peak_memory,
            parameters=parameters,
        ),
    )
    return RepeatedRun(label, seed, run_dir, evaluation, benchmark)


def test_pareto_analysis_requires_checkpoint_identity_and_finds_dominance(tmp_path):
    checkpoint_hash = "a" * 64
    evaluation_a = _write_json(
        tmp_path / "eval-a.json",
        _evaluation(checkpoint_hash, 80),
    )
    benchmark_a = _write_json(
        tmp_path / "bench-a.json",
        _torch_benchmark(checkpoint_hash, p50=2.0, peak_memory=100, parameters=10),
    )
    evaluation_b = _write_json(
        tmp_path / "eval-b.json",
        _evaluation("b" * 64, 70),
    )
    benchmark_b = _write_json(
        tmp_path / "bench-b.json",
        _torch_benchmark("b" * 64, p50=3.0, peak_memory=120, parameters=12),
    )

    report = analyze(
        [("A", evaluation_a, benchmark_a), ("B", evaluation_b, benchmark_b)]
    )

    assert report["pareto_variants"] == ["A"]
    assert report["points"][1]["dominated_by"] == ["A"]
    report_path = _write_json(tmp_path / "pareto.json", report)
    task = Task(
        name="pareto",
        stage="tuning",
        command=[],
        completion_kind="pareto",
        completion_paths=(report_path,),
        pareto_records=(
            ("A", evaluation_a, benchmark_a),
            ("B", evaluation_b, benchmark_b),
        ),
    )
    assert task.complete()
    report["pareto_variants"] = ["B"]
    _write_json(report_path, report)
    assert not task.complete()
    bad_benchmark = _write_json(
        tmp_path / "bad.json",
        _torch_benchmark("c" * 64, p50=1.0, peak_memory=1, parameters=1),
    )
    with pytest.raises(ValueError, match="checkpoint SHA-256 differ"):
        analyze([("bad", evaluation_a, bad_benchmark)])


def test_repeated_pareto_uses_three_matched_seeds_and_paired_confidence(tmp_path):
    records = []
    values = {
        "S-direct-xl": ([69.0, 70.0, 71.0], [3.9, 4.0, 4.1], 100.0, 20),
        "S-speed": ([67.0, 68.0, 69.0], [1.9, 2.0, 2.1], 80.0, 18),
        "S-accuracy": ([74.0, 75.0, 76.0], [4.9, 5.0, 5.1], 120.0, 24),
    }
    for label, (sap_values, latency_values, memory, parameters) in values.items():
        for seed, sap10, p50 in zip((42, 43, 44), sap_values, latency_values, strict=True):
            records.append(_repeated_run(
                tmp_path,
                label=label,
                seed=seed,
                sap10=sap10,
                p50=p50,
                peak_memory=memory,
                parameters=parameters,
            ))

    report = analyze_repeated(
        records,
        baseline_label="S-direct-xl",
        minimum_sap10_gain=1.0,
    )
    points = {point["label"]: point for point in report["points"]}
    assert report["format"] == "lineae_repeated_pareto_v1"
    assert report["seeds"] == [42, 43, 44]
    assert report["seed_count"] == 3
    assert points["S-direct-xl"]["sap10"]["mean"] == 70.0
    assert points["S-speed"]["paired_vs_baseline"]["latency_gain_confident"]
    assert points["S-speed"]["paired_vs_baseline"]["memory_gain_confident"]
    assert points["S-accuracy"]["paired_vs_baseline"]["accuracy_gain_confident"]
    assert set(report["pareto_mean"]) == set(values)

    manifest = records[-1].run_dir / "run_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["training"]["seed"] = 99
    _write_json(manifest, payload)
    with pytest.raises(ValueError, match="run-manifest seed mismatch"):
        analyze_repeated(records, baseline_label="S-direct-xl")


def test_repeated_pareto_parser_and_minimum_seed_gate(tmp_path):
    parsed = run_argument("S-speed:42=/run,/eval.json,/bench.json")
    assert parsed.label == "S-speed"
    assert parsed.seed == 42
    with pytest.raises(Exception, match="run must be"):
        run_argument("invalid")

    records = [
        _repeated_run(
            tmp_path,
            label="S-direct-xl",
            seed=seed,
            sap10=70.0,
            p50=4.0,
            peak_memory=100.0,
            parameters=20,
        )
        for seed in (42, 43)
    ]
    with pytest.raises(ValueError, match="at least 3 are required"):
        analyze_repeated(records, baseline_label="S-direct-xl")


def test_model_card_requires_evaluation_torch_and_tensorrt_hash_chain(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"model")
    checkpoint_hash = sha256_file(checkpoint)
    datasets = _datasets(2.0)
    evaluation = _write_json(
        tmp_path / "evaluation.json",
        {
            **_PREPROCESSING,
            "format": "lineae_evaluation_v3",
            "config": "/test/config.py",
            "checkpoint_sha256": checkpoint_hash,
            "num_queries": 1100,
            "num_select": 300,
            "sap_protocol": "official_all_queries_and_deployment_topk",
            "datasets": datasets,
        },
    )
    torch_benchmark_payload = {
        **_PREPROCESSING,
        "checkpoint_sha256": checkpoint_hash,
        "format": "lineae_torch_benchmark_v1",
        "config": "/test/config.py",
        "device": "cuda",
        "batch_size": 1,
        "deploy_mode": True,
        "num_select": 300,
        "spatial_size": 640,
        "parameters": 10,
        "complexity": {"flops": "1 GFLOPS", "macs": "0.5 GMACs"},
        "latency_ms": {"p50": 2.0, "p95": 2.0},
        "peak_memory_mib": 100.0,
        "gpu": "test GPU",
        "amp": True,
        "warmup": 10,
        "iterations": 20,
        "samples_ms": [2.0] * 20,
    }
    torch_benchmark = _write_json(
        tmp_path / "torch.json", torch_benchmark_payload
    )
    tensorrt = _write_json(
        tmp_path / "trt.json",
        {
            "format": "lineae_tensorrt_benchmark_v1",
            "checkpoint_sha256": checkpoint_hash,
            "onnx_sha256": "a" * 64,
            "engine_sha256": "b" * 64,
            "fp16": True,
            "tf32_disabled": True,
            "onnxruntime_version": "1.26.0",
            "parity": {"pred_logits": True, "pred_lines": True},
            "latency_ms": {"median": 1.0, "percentile_95": 1.5},
        },
    )
    pareto_payload = {
        "format": "lineae_pareto_v1",
        "points": [{
            "variant": "S",
            "checkpoint_sha256": checkpoint_hash,
            "evaluation_sha256": sha256_file(evaluation),
            "benchmark_sha256": sha256_file(torch_benchmark),
            "pareto": True,
        }],
        "pareto_variants": ["S"],
    }
    pareto = _write_json(tmp_path / "pareto.json", pareto_payload)
    args = SimpleNamespace(
        variant="S",
        checkpoint=checkpoint,
        evaluation=evaluation,
        torch_benchmark=torch_benchmark,
        tensorrt_benchmark=tensorrt,
        pareto_report=pareto,
        initialization=None,
        limitations=None,
    )

    card = generate(args)

    assert checkpoint_hash in card
    assert "LINEAE-S model card" in card
    pareto_payload["points"][0]["pareto"] = False
    _write_json(pareto, pareto_payload)
    with pytest.raises(ValueError, match="not a hash-matched Pareto"):
        generate(args)
    pareto_payload["points"][0]["pareto"] = True
    _write_json(pareto, pareto_payload)
    torch_benchmark_payload["checkpoint_sha256"] = "e" * 64
    _write_json(torch_benchmark, torch_benchmark_payload)
    with pytest.raises(ValueError, match="PyTorch benchmark checkpoint hash"):
        generate(args)


def test_full_dataset_onnx_ap_parity_is_hash_and_annotation_bound():
    def report(offset=0.0, annotation="a" * 64):
        return {
            "checkpoint_sha256": "c" * 64,
            "num_select": 300,
            "image_preprocess_schema": "opencv_rgb_inter_linear_v2",
            "datasets": {
                name: {
                    "annotation_sha256": annotation,
                    "samples": 10,
                    "deploy_sap5": 70.0 + offset,
                    "deploy_sap10": 65.0 + offset,
                    "deploy_sap15": 60.0 + offset,
                }
                for name in ("wireframe", "york")
            },
        }

    parity = compare_ap(report(0.01), report(), tolerance=0.05)
    assert parity["maximum_delta"] == pytest.approx(0.01)
    with pytest.raises(RuntimeError, match="final-AP parity failed"):
        compare_ap(report(0.1), report(), tolerance=0.05)
    with pytest.raises(ValueError, match="annotation_sha256 mismatch"):
        compare_ap(report(annotation="b" * 64), report(), tolerance=0.05)


def test_onnx_evaluation_rejects_an_outdated_export_schema(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not parsed because the export report is rejected first")
    export_report = _write_json(
        tmp_path / "model.export.json",
        {
            **_PREPROCESSING,
            "format": "lineae_onnx_export_v2",
            "config": "/test/config.py",
            "checkpoint_sha256": "a" * 64,
            "onnx_sha256": sha256_file(model),
            "onnxruntime_version": "1.26.0",
            "onnxsim_version": "v0.6.5",
            "onnx_simplified": True,
            "deploy_mode": True,
            "seed": 0,
            "input_shape": [1, 3, 640, 640],
            "num_select": 300,
            "output_shapes": {
                "pred_logits": [1, 300, 2],
                "pred_lines": [1, 300, 4],
            },
        },
    )
    args = SimpleNamespace(onnx=model, onnx_report=export_report)
    with pytest.raises(ValueError, match="unsupported ONNX export report format"):
        evaluate_onnx(args)


def test_export_runtime_versions_are_exactly_pinned():
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    lock = Path("uv.lock").read_text(encoding="utf-8")
    assert '"onnxruntime-gpu==1.26.0"' in project
    assert '"onnxsim==0.6.5"' in project
    assert 'name = "onnxruntime-gpu"\nversion = "1.26.0"' in lock
    assert 'name = "onnxsim"\nversion = "0.6.5"' in lock


def test_all_direct_dependencies_are_exactly_pinned_and_locked():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    locked_names = [
        package["name"].lower().replace("_", "-") for package in lock["package"]
    ]
    assert len(locked_names) == len(set(locked_names)), (
        "uv.lock contains Python-dependent package-version forks"
    )
    locked = {
        (package["name"].lower().replace("_", "-"), package["version"])
        for package in lock["package"]
    }
    groups = {
        "runtime": project["project"]["dependencies"],
        **project["project"]["optional-dependencies"],
    }
    for group, requirements in groups.items():
        for requirement in requirements:
            assert ";" not in requirement, (
                f"{group} dependency varies by environment: {requirement}"
            )
            declaration = requirement.split(";", 1)[0].strip()
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^,;\s]+)", declaration)
            assert match is not None, f"{group} dependency is not exact: {requirement}"
            name = match.group(1).lower().replace("_", "-")
            version = match.group(2)
            assert (name, version) in locked, f"missing lock entry for {requirement}"


def test_documented_uv_commands_refuse_dependency_reresolution():
    document = Path("README.md")
    for line_number, line in enumerate(
        document.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if re.search(r"\buv (?:sync|run)\b", line):
            assert "--locked" in line, f"{document}:{line_number}: {line}"
        assert "--frozen" not in line, f"{document}:{line_number}: {line}"


def test_readme_documents_every_tensorboard_scalar_family():
    readme = Path("README.md").read_text(encoding="utf-8")
    for tag in (
        "Loss/total",
        "Loss/loss_logits",
        "Loss/loss_line",
        "Loss/loss_logits_<i>",
        "Loss/loss_line_<i>",
        "Loss/loss_logits_interm",
        "Loss/loss_line_interm",
        "Loss/loss_logits_dn_<i>",
        "Loss/loss_line_dn_<i>",
        "Loss/loss_kd_logits",
        "Loss/loss_kd_line",
        "Loss/loss_kd_feature",
        "Lr/pg_<i>",
        "GradNorm/pg_<i>",
        "Distillation/weight",
        "Distillation/temperature",
        "Distillation/matches",
        "Distillation/overhead_ms",
        "Test/loss",
        "Test/loss_logits",
        "Test/loss_line",
        "Test/sap5",
        "Test/sap10",
        "Test/sap15",
        "Test/official_sap5",
        "Test/official_sap10",
        "Test/official_sap15",
        "Test/deploy_sap5",
        "Test/deploy_sap10",
        "Test/deploy_sap15",
    ):
        assert f"`{tag}`" in readme


def test_readme_documents_exact_xl_resume_command():
    readme = Path("README.md").read_text(encoding="utf-8")
    xl_workflow = readme.split("## XL teacher workflow", 1)[1].split(
        "## Distillation", 1
    )[0]
    assert "-c configs/lineae/lineae_xl.py" in xl_workflow
    assert (
        "--resume outputs/lineae_xl-seed42/checkpoint.pth"
        in xl_workflow
    )
    assert "output_dir=outputs/lineae_xl-seed42" in xl_workflow
    assert "batch_size_train=8" in xl_workflow
    assert "batch_size_val=64" in xl_workflow
    assert "epochs=36" in xl_workflow
    assert "gradient_accumulation_steps=1" in xl_workflow
    assert "use_checkpoint=False" in xl_workflow


def test_readme_documents_a_through_x_supervised_commands():
    readme = Path("README.md").read_text(encoding="utf-8")
    workflow = readme.split(
        "## A–X supervised workflow without distillation", 1
    )[1].split("## XL teacher workflow", 1)[0]

    for variant in ("A", "F", "P", "N", "M", "L", "X"):
        path = f"configs/lineae/lineae_{variant.lower()}.py"
        assert f"| {variant} | `{path}` |" in workflow
    assert "| S | `configs/lineae/baselines/lineae_s.py` |" in workflow
    assert '-c "configs/lineae/lineae_${VARIANT}.py"' in workflow
    assert "-c configs/lineae/baselines/lineae_s.py" in workflow
    assert "distill_weight=0.0" in workflow
    assert "ckpts/vitt_distill.pt" in workflow


def test_readme_documents_xl_full_unfreeze_command():
    readme = Path("README.md").read_text(encoding="utf-8")
    xl_workflow = readme.split("## XL teacher workflow", 1)[1].split(
        "## Distillation", 1
    )[0]
    assert "output_dir=outputs/lineae_xl-full-unfreeze-v2-seed42" in xl_workflow
    assert "progressive_unfreeze=False backbone_trainable_layers=0" in xl_workflow
    assert "initial_freeze_epochs=0 unfreeze_interval=0" in xl_workflow
    assert "`backbone_trainable_layers=0` means all backbone blocks" in xl_workflow
    assert "cannot resume a checkpoint created with the progressive settings" in xl_workflow
