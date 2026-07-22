import hashlib
import json
from pathlib import Path

from main import write_run_completion
from tools.plan_experiment_matrix import (
    CASCADE_STUDENT_ORDER,
    MatrixOptions,
    STUDENT_ORDER,
    TUNING_STUDENT_ORDER,
    Task,
    build_matrix,
    selected_tasks,
)

_PREPROCESSING = {
    "image_preprocess_schema": "opencv_rgb_inter_linear_v2",
    "opencv_version": "4.13.0",
}


def _options(tmp_path):
    return MatrixOptions(
        output_root=tmp_path / "matrix",
        wireframe=Path("data/wireframe_processed"),
        york=Path("data/york_processed"),
        teacher=tmp_path / "lineae_xl_teacher.pth",
        python="python",
    )


def test_matrix_has_unique_gated_xl_to_student_order_and_matched_controls(tmp_path):
    options = _options(tmp_path)
    tasks = build_matrix(options)
    names = [task.name for task in tasks]
    assert len(names) == len(set(names))
    assert names.index("qualify_lineae_xl") < names.index("lineae_x_nokd")
    assert names.index("lineae_x_kd") < names.index("lineae_l_nokd")

    by_name = {task.name: task for task in tasks}
    previous_kd = "qualify_lineae_xl"
    for variant in STUDENT_ORDER:
        lower = variant.lower()
        no_kd = by_name[f"lineae_{lower}_nokd"]
        kd = by_name[f"lineae_{lower}_kd"]
        assert no_kd.name in kd.dependencies
        assert "qualify_lineae_xl" in kd.dependencies
        if variant != "S":
            assert previous_kd in no_kd.dependencies
        else:
            assert previous_kd in kd.dependencies
        assert f"distill_teacher_checkpoint={options.teacher}" in kd.command
        option_index = kd.command.index("--options")
        assert kd.command.index("--amp") < option_index
        assert kd.command[option_index + 1].startswith("output_dir=")
        previous_kd = kd.name

    teacher_stage = selected_tasks(tasks, "teacher")
    assert len(teacher_stage) == 8
    assert all(task.stage in {"baseline", "teacher"} for task in teacher_stage)
    trt = by_name["tensorrt_lineae_x_kd"]
    assert "evaluate_onnx_lineae_x_kd" in trt.dependencies


def test_opt_in_ablation_matrix_adds_matched_accuracy_and_deployment_tasks(tmp_path):
    options = _options(tmp_path)
    options.include_ablations = True
    tasks = build_matrix(options)
    by_name = {task.name: task for task in tasks}

    assert len(tasks) == 131
    assert len(by_name) == len(tasks)
    assert len(selected_tasks(tasks, "teacher")) == 12
    qualify = by_name["qualify_lineae_xl"]
    for name, config in (
        ("lineae_xl_ema_ablation", "configs/lineae/ablations/lineae_xl_ema.py"),
        (
            "lineae_xl_photometric_ablation",
            "configs/lineae/ablations/lineae_xl_photometric.py",
        ),
    ):
        evaluation_name = f"evaluate_{name}"
        assert evaluation_name in qualify.dependencies
        assert any(
            argument.startswith(f"{config},")
            for argument in qualify.command
        )

    intermediate = by_name["lineae_x_intermediate_ablation"]
    feature = by_name["lineae_x_feature_kd_ablation"]
    assert intermediate.dependencies == ("lineae_x_nokd",)
    assert set(feature.dependencies) == {"lineae_x_kd", "qualify_lineae_xl"}
    assert f"distill_teacher_checkpoint={options.teacher}" in feature.command
    for name in (intermediate.name, feature.name):
        assert f"evaluate_{name}" in by_name
        assert f"benchmark_{name}" in by_name
        assert f"export_{name}" in by_name
        assert f"evaluate_onnx_{name}" in by_name
        assert f"tensorrt_{name}" in by_name
        assert f"evaluate_onnx_{name}" in by_name[f"tensorrt_{name}"].dependencies


def test_opt_in_x_teacher_cascade_runs_only_after_direct_xl_controls(tmp_path):
    options = _options(tmp_path)
    options.include_cascade = True
    options.cascade_teacher = tmp_path / "lineae_x_teacher.pth"
    tasks = build_matrix(options)
    by_name = {task.name: task for task in tasks}

    assert len(tasks) == 158
    assert len(by_name) == len(tasks)
    assert len(selected_tasks(tasks, "cascade")) == 35
    qualify = by_name["qualify_lineae_x_cascade_teacher"]
    assert qualify.dependencies == (
        "evaluate_lineae_x_nokd",
        "evaluate_lineae_x_kd",
    )
    assert qualify.source_checkpoint == (
        options.output_root / "training/lineae_x_nokd/checkpoint_best.pth"
    )
    assert qualify.candidate_checkpoints == (
        options.output_root / "training/lineae_x_kd/checkpoint_best.pth",
    )
    assert "configs/lineae/distill/lineae_x.py" in qualify.command
    assert "configs/lineae/lineae_x.py" in qualify.command
    assert str(options.cascade_teacher) in qualify.command

    previous = qualify.name
    for variant in CASCADE_STUDENT_ORDER[1:]:
        lower = variant.lower()
        name = f"lineae_{lower}_cascade_x"
        cascade = by_name[name]
        assert cascade.dependencies == (f"lineae_{lower}_kd", previous)
        assert f"configs/lineae/cascade/lineae_{lower}.py" in cascade.command
        assert "distill_teacher_config=configs/lineae/lineae_x.py" in cascade.command
        assert f"distill_teacher_checkpoint={options.cascade_teacher}" in cascade.command
        for prefix in ("evaluate_", "benchmark_", "export_", "evaluate_onnx_", "tensorrt_"):
            assert f"{prefix}{name}" in by_name
        previous = name

    assert "lineae_t_cascade_x" not in by_name

    indices = {task.name: index for index, task in enumerate(tasks)}
    for task in tasks:
        assert all(indices[dependency] < indices[task.name] for dependency in task.dependencies)

    options.include_ablations = True
    assert len(build_matrix(options)) == 174


def test_opt_in_tuning_matrix_compares_two_bundles_to_each_direct_xl_model(tmp_path):
    options = _options(tmp_path)
    options.include_tuning = True
    tasks = build_matrix(options)
    by_name = {task.name: task for task in tasks}

    assert len(tasks) == 171
    assert len(by_name) == len(tasks)
    assert len(selected_tasks(tasks, "tuning")) == 97
    previous = "qualify_lineae_xl"
    for variant in TUNING_STUDENT_ORDER:
        lower = variant.lower()
        direct_name = f"lineae_{lower}_kd"
        assert by_name[f"evaluate_{direct_name}"].stage == "tuning"
        assert by_name[f"benchmark_{direct_name}"].stage == "tuning"
        candidates = []
        for profile in ("speed", "accuracy"):
            name = f"lineae_{lower}_tune_{profile}"
            candidate = by_name[name]
            assert candidate.dependencies == (direct_name, previous)
            assert f"configs/lineae/tuning/{name}.py" in candidate.command
            assert f"distill_teacher_checkpoint={options.teacher}" in candidate.command
            assert by_name[f"evaluate_{name}"].stage == "tuning"
            assert by_name[f"benchmark_{name}"].stage == "tuning"
            assert f"export_{name}" not in by_name
            candidates.append(name)
            previous = f"benchmark_{name}"

        pareto = by_name[f"pareto_lineae_{lower}_tuning"]
        assert pareto.dependencies == (
            f"evaluate_{direct_name}",
            f"benchmark_{direct_name}",
            f"evaluate_{candidates[0]}",
            f"benchmark_{candidates[0]}",
            f"evaluate_{candidates[1]}",
            f"benchmark_{candidates[1]}",
        )
        assert [record[0] for record in pareto.pareto_records] == [
            f"{variant}-direct-xl",
            f"{variant}-speed",
            f"{variant}-accuracy",
        ]
        previous = pareto.name

    assert "lineae_t_tune_speed" not in by_name
    assert "lineae_t_tune_accuracy" not in by_name

    indices = {task.name: index for index, task in enumerate(tasks)}
    for task in tasks:
        assert all(indices[dependency] < indices[task.name] for dependency in task.dependencies)

    options.include_ablations = True
    options.include_cascade = True
    assert len(build_matrix(options)) == 230


def test_training_completion_marker_distinguishes_bounded_full_and_stale(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "checkpoint.pth").write_bytes(b"latest")
    best = output / "checkpoint_best.pth"
    best.write_bytes(b"best")
    task = Task(
        name="train",
        stage="baseline",
        command=["python", "main.py"],
        completion_kind="run",
        output_dir=output,
    )
    assert not task.complete()
    write_run_completion(
        output,
        status="bounded",
        final_epoch=0,
        global_step=1,
        best_metric_name="sap10",
        best_metric=1.0,
        best_epoch=0,
    )
    assert not task.complete()
    write_run_completion(
        output,
        status="full",
        final_epoch=35,
        global_step=100,
        best_metric_name="sap10",
        best_metric=10.0,
        best_epoch=30,
    )
    assert task.complete()
    best.write_bytes(b"changed")
    assert not task.complete()


def test_plan_records_command_and_reports_are_bound_to_checkpoint(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    train = Task(
        name="train",
        stage="baseline",
        command=["python", "main.py"],
        completion_kind="run",
        output_dir=output,
    )
    assert train.record()["command"] == ["python", "main.py"]

    report_path = tmp_path / "evaluation.json"
    source = tmp_path / "best.pth"
    source.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    report_path.write_text(json.dumps({
        **_PREPROCESSING,
        "format": "lineae_evaluation_v3",
        "config": str(Path("configs/lineae/lineae_s.py").resolve()),
        "checkpoint_sha256": digest,
        "num_queries": 1100,
        "num_select": 300,
        "sap_protocol": "official_all_queries_and_deployment_topk",
        "datasets": {
            dataset: {
                "annotation_sha256": "a" * 64,
                "samples": 10,
                "sap5": 1.0,
                "sap10": 2.0,
                "sap15": 3.0,
                "deploy_sap5": 1.0,
                "deploy_sap10": 2.0,
                "deploy_sap15": 3.0,
            }
            for dataset in ("wireframe", "york")
        },
    }), encoding="utf-8")
    report = Task(
        name="eval",
        stage="deployment",
        command=[],
        completion_kind="evaluation",
        completion_paths=(report_path,),
        source_checkpoint=source,
    )
    assert report.complete()
    source.write_bytes(b"new weights")
    assert not report.complete()


def test_onnx_completion_requires_pinned_simplified_export_report(tmp_path):
    source = tmp_path / "model.pth"
    source.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    export_report_path = tmp_path / "model.export.json"
    payload = {
        **_PREPROCESSING,
        "format": "lineae_onnx_export_v3",
        "config": "/test/config.py",
        "checkpoint_sha256": digest,
        "onnx_sha256": hashlib.sha256(b"onnx").hexdigest(),
        "onnx_simplified": True,
        "deploy_mode": True,
        "seed": 0,
        "onnxsim_version": "v0.6.5",
        "input_shape": [1, 3, 640, 640],
        "num_select": 300,
        "output_shapes": {
            "pred_logits": [1, 300, 2],
            "pred_lines": [1, 300, 4],
        },
    }
    export_report_path.write_text(json.dumps(payload), encoding="utf-8")
    task = Task(
        name="onnx",
        stage="deployment",
        command=[],
        completion_kind="onnx",
        completion_paths=(model, export_report_path),
        source_checkpoint=source,
    )
    assert task.complete()
    payload["onnx_simplified"] = False
    export_report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert not task.complete()
    payload["onnx_simplified"] = True
    payload["format"] = "lineae_onnx_export_v2"
    export_report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert not task.complete()


def test_onnx_ap_completion_binds_model_torch_report_and_cuda_policy(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    torch_report = tmp_path / "torch.json"
    torch_report.write_bytes(b"torch report")
    result = tmp_path / "onnx-eval.json"
    deltas = {
        dataset: {
            metric: 0.01
            for metric in ("deploy_sap5", "deploy_sap10", "deploy_sap15")
        }
        for dataset in ("wireframe", "york")
    }
    result.write_text(json.dumps({
        **_PREPROCESSING,
        "format": "lineae_onnx_evaluation_v3",
        "config": str(Path("configs/lineae/lineae_s.py").resolve()),
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "onnx_sha256": hashlib.sha256(b"onnx").hexdigest(),
        "torch_report_sha256": hashlib.sha256(b"torch report").hexdigest(),
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "provider_options": {"CUDAExecutionProvider": {"use_tf32": "0"}},
        "cpu_ep_fallback_disabled": True,
        "num_select": 300,
        "sap_protocol": "deployment_topk",
        "datasets": {
            dataset: {
                "annotation_sha256": "a" * 64,
                "samples": 10,
                "deploy_sap5": 1.0,
                "deploy_sap10": 2.0,
                "deploy_sap15": 3.0,
            }
            for dataset in ("wireframe", "york")
        },
        "ap_parity": {
            "tolerance": 0.05,
            "maximum_delta": 0.01,
            "absolute_delta": deltas,
        },
    }), encoding="utf-8")
    task = Task(
        name="onnx-ap",
        stage="deployment",
        command=[
            "python",
            "tools/evaluate_onnx.py",
            "-c",
            "configs/lineae/lineae_s.py",
            "--cuda-ort",
        ],
        completion_kind="onnx_ap",
        completion_paths=(result,),
        source_checkpoint=checkpoint,
        source_artifact=onnx,
        comparison_artifact=torch_report,
    )
    assert task.complete()
    torch_report.write_bytes(b"changed")
    assert not task.complete()
