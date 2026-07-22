"""Write the gated LINEAE full-training experiment plan.

This command is intentionally non-executing: it only writes and prints a
machine-readable plan. Training, evaluation, and export commands are launched
individually by the operator on the intended host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from util.artifact_validation import (
    validate_evaluation_report,
    validate_onnx_evaluation_report,
    validate_onnx_export_report,
    validate_tensorrt_report,
    validate_torch_benchmark_report,
)


STAGE_RANK = {
    "baseline": 0,
    "teacher": 1,
    "distillation": 2,
    "cascade": 3,
    "tuning": 4,
    "deployment": 5,
}
STUDENT_ORDER = ("X", "L", "M", "S", "T", "N", "P", "F", "A")
# T is part of the core supervised/direct-XL comparison.  Cascade and coarse
# tuning remain limited to the previously registered experiment families until
# T has separate, explicitly reviewed candidate configs.
CASCADE_STUDENT_ORDER = ("X", "L", "M", "S", "N", "P", "F", "A")
TUNING_STUDENT_ORDER = ("X", "L", "M", "S", "N", "P", "F", "A")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Task:
    name: str
    stage: str
    command: list[str]
    dependencies: tuple[str, ...] = ()
    completion_kind: str = "file"
    completion_paths: tuple[Path, ...] = ()
    output_dir: Path | None = None
    source_checkpoint: Path | None = None
    source_artifact: Path | None = None
    comparison_artifact: Path | None = None
    candidate_checkpoints: tuple[Path, ...] = ()
    pareto_records: tuple[tuple[str, Path, Path], ...] = ()

    def complete(self) -> bool:
        if self.completion_kind == "run":
            if self.output_dir is None:
                return False
            record_path = self.output_dir / "run_complete.json"
            best_path = self.output_dir / "checkpoint_best.pth"
            if not record_path.is_file() or not best_path.is_file():
                return False
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            return (
                record.get("format") == "lineae_run_completion_v1"
                and record.get("status") == "full"
                and record.get("best_checkpoint_sha256") == _sha256(best_path)
            )
        if self.completion_kind == "onnx":
            if len(self.completion_paths) != 2 or not all(
                path.is_file() for path in self.completion_paths
            ):
                return False
            if self.completion_paths[0].stat().st_size <= 0:
                return False
            try:
                report = json.loads(self.completion_paths[1].read_text(encoding="utf-8"))
                validate_onnx_export_report(
                    report,
                    expected_checkpoint=(
                        _sha256(self.source_checkpoint)
                        if self.source_checkpoint is not None
                        and self.source_checkpoint.is_file()
                        else None
                    ),
                    expected_onnx=_sha256(self.completion_paths[0]),
                    expected_config=self._expected_config(),
                    require_simplified=True,
                )
            except (OSError, json.JSONDecodeError):
                return False
            except (TypeError, ValueError):
                return False
            return self._report_matches_source(report)
        if self.completion_kind == "teacher":
            if len(self.completion_paths) != 2 or not all(
                path.is_file() for path in self.completion_paths
            ):
                return False
            if self.completion_paths[0].stat().st_size <= 0:
                return False
            try:
                report = json.loads(self.completion_paths[1].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            try:
                import torch

                from main import validate_teacher_artifact

                inference_config = Path(report.get("inference_config", ""))
                artifact = torch.load(
                    self.completion_paths[0],
                    map_location="cpu",
                    weights_only=False,
                )
                validate_teacher_artifact(artifact, inference_config)
            except (OSError, RuntimeError, TypeError, ValueError):
                return False
            sources = self.candidate_checkpoints or (
                (self.source_checkpoint,) if self.source_checkpoint is not None else ()
            )
            return (
                report.get("format") == "lineae_teacher_qualification_v3"
                and report.get("reload_identical") is True
                and report.get("canonical_inference_identical") is True
                and report.get("sha256") == _sha256(self.completion_paths[0])
                and any(
                    source.is_file()
                    and report.get("source_sha256") == _sha256(source)
                    for source in sources
                )
                and (
                    self.source_checkpoint is None
                    or (
                        self.source_checkpoint.is_file()
                        and report.get("baseline_source_sha256")
                        == _sha256(self.source_checkpoint)
                    )
                )
            )
        if self.completion_kind in {
            "evaluation",
            "benchmark",
            "onnx_ap",
            "tensorrt",
        }:
            report_path = (
                self.completion_paths[0]
                if self.completion_kind in {"evaluation", "benchmark", "onnx_ap"}
                else self.completion_paths[1]
            )
            if not all(path.is_file() for path in self.completion_paths):
                return False
            if self.completion_kind == "tensorrt" and any(
                path.stat().st_size <= 0 for path in self.completion_paths
            ):
                return False
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if not self._report_matches_source(report):
                return False
            expected_checkpoint = (
                _sha256(self.source_checkpoint)
                if self.source_checkpoint is not None and self.source_checkpoint.is_file()
                else None
            )
            expected_artifact = (
                _sha256(self.source_artifact)
                if self.source_artifact is not None and self.source_artifact.is_file()
                else None
            )
            expected_comparison = (
                _sha256(self.comparison_artifact)
                if self.comparison_artifact is not None
                and self.comparison_artifact.is_file()
                else None
            )
            expected_config = self._expected_config()
            try:
                if self.completion_kind == "evaluation":
                    validate_evaluation_report(
                        report,
                        expected_checkpoint=expected_checkpoint,
                        expected_config=expected_config,
                    )
                elif self.completion_kind == "benchmark":
                    validate_torch_benchmark_report(
                        report,
                        expected_checkpoint=expected_checkpoint,
                        expected_config=expected_config,
                        require_cuda=self._requires_cuda(),
                        require_samples=True,
                    )
                elif self.completion_kind == "onnx_ap":
                    validate_onnx_evaluation_report(
                        report,
                        expected_checkpoint=expected_checkpoint,
                        expected_onnx=expected_artifact,
                        expected_torch_report=expected_comparison,
                        expected_config=expected_config,
                        require_cuda="--cuda-ort" in self.command,
                    )
                else:
                    validate_tensorrt_report(
                        report,
                        expected_checkpoint=expected_checkpoint,
                        expected_onnx=expected_artifact,
                        expected_engine=_sha256(self.completion_paths[0]),
                        require_fp16="--fp16" in self.command,
                    )
            except (TypeError, ValueError):
                return False
            return True
        if self.completion_kind == "pareto":
            if len(self.completion_paths) != 1 or not self.completion_paths[0].is_file():
                return False
            try:
                from tools.analyze_pareto import analyze

                report = json.loads(self.completion_paths[0].read_text(encoding="utf-8"))
                return report == analyze(self.pareto_records)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                return False
        return bool(self.completion_paths) and all(path.is_file() for path in self.completion_paths)

    def _report_matches_source(self, report: dict) -> bool:
        if self.source_checkpoint is None:
            return True
        return (
            self.source_checkpoint.is_file()
            and report.get("checkpoint_sha256") == _sha256(self.source_checkpoint)
        )

    def _requires_cuda(self) -> bool:
        try:
            device = self.command[self.command.index("--device") + 1]
        except (ValueError, IndexError):
            return False
        return device.startswith("cuda")

    def _expected_config(self) -> str | None:
        for option in ("-c", "--config"):
            try:
                value = self.command[self.command.index(option) + 1]
            except (ValueError, IndexError):
                continue
            return str(Path(value).resolve())
        return None

    def record(self) -> dict:
        return {
            "name": self.name,
            "stage": self.stage,
            "dependencies": list(self.dependencies),
            "command": list(self.command),
            "completion_kind": self.completion_kind,
            "completion_paths": [str(path) for path in self.completion_paths],
            "source_checkpoint": (
                str(self.source_checkpoint) if self.source_checkpoint is not None else None
            ),
            "source_artifact": (
                str(self.source_artifact) if self.source_artifact is not None else None
            ),
            "comparison_artifact": (
                str(self.comparison_artifact)
                if self.comparison_artifact is not None else None
            ),
            "candidate_checkpoints": [str(path) for path in self.candidate_checkpoints],
            "pareto_records": [
                [label, str(evaluation), str(benchmark)]
                for label, evaluation, benchmark in self.pareto_records
            ],
            "status": "complete" if self.complete() else "pending",
        }


@dataclass
class MatrixOptions:
    output_root: Path
    wireframe: Path
    york: Path
    teacher: Path
    device: str = "cuda"
    workers: int = 8
    seed: int = 42
    amp: bool = True
    minimum_ap10_gain: float = 0.0
    include_ablations: bool = False
    include_cascade: bool = False
    include_tuning: bool = False
    cascade_teacher: Path = Path("ckpts/lineae_x_teacher.pth")
    python: str = sys.executable
    extra_train_args: list[str] = field(default_factory=list)


def _train_task(
    options: MatrixOptions,
    *,
    name: str,
    stage: str,
    config: str,
    dependencies=(),
    config_options=(),
) -> Task:
    output_dir = options.output_root / "training" / name
    command = [
        options.python,
        "main.py",
        "-c",
        config,
        "--coco_path",
        str(options.wireframe),
        "--device",
        options.device,
        "--seed",
        str(options.seed),
        "--num_workers",
        str(options.workers),
    ]
    if options.amp:
        command.append("--amp")
    command.extend([
        "--options",
        f"output_dir={output_dir}",
        *config_options,
    ])
    command.extend(options.extra_train_args)
    return Task(
        name=name,
        stage=stage,
        command=command,
        dependencies=tuple(dependencies),
        completion_kind="run",
        completion_paths=(
            output_dir / "run_complete.json",
            output_dir / "checkpoint_best.pth",
        ),
        output_dir=output_dir,
    )


def _evaluation_task(
    options: MatrixOptions,
    *,
    name: str,
    stage: str,
    config: str,
    training_task: Task,
) -> Task:
    output = options.output_root / "evaluations" / f"{name}.json"
    checkpoint = training_task.output_dir / "checkpoint_best.pth"
    command = [
        options.python,
        "tools/evaluate_checkpoint.py",
        "-c",
        config,
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        f"wireframe={options.wireframe}",
        "--dataset",
        f"york={options.york}",
        "--device",
        options.device,
        "--batch-size",
        "1",
        "--num-workers",
        str(options.workers),
        "--output",
        str(output),
    ]
    if options.amp:
        command.insert(-2, "--amp")
    return Task(
        name=f"evaluate_{name}",
        stage=stage,
        command=command,
        dependencies=(training_task.name,),
        completion_paths=(output,),
        completion_kind="evaluation",
        source_checkpoint=checkpoint,
    )


def _benchmark_task(
    options: MatrixOptions,
    *,
    name: str,
    config: str,
    training_task: Task,
    stage: str = "deployment",
) -> Task:
    output = options.output_root / "benchmarks" / f"{name}-amp.json"
    checkpoint = training_task.output_dir / "checkpoint_best.pth"
    command = [
        options.python,
        "tools/benchmark.py",
        "-c",
        config,
        "--checkpoint",
        str(checkpoint),
        "--device",
        options.device,
        "--warmup",
        "50",
        "--iterations",
        "200",
        "--include-samples",
        "--output",
        str(output),
    ]
    if options.amp:
        command.insert(-2, "--amp")
    return Task(
        name=f"benchmark_{name}",
        stage=stage,
        command=command,
        dependencies=(training_task.name,),
        completion_paths=(output,),
        completion_kind="benchmark",
        source_checkpoint=checkpoint,
    )


def _export_tasks(
    options: MatrixOptions,
    *,
    name: str,
    config: str,
    training_task: Task,
) -> tuple[Task, Task]:
    checkpoint = training_task.output_dir / "checkpoint_best.pth"
    onnx = options.output_root / "onnx" / f"{name}.onnx"
    export_report = options.output_root / "onnx" / f"{name}.export.json"
    export = Task(
        name=f"export_{name}",
        stage="deployment",
        command=[
            options.python,
            "tools/export_onnx.py",
            "-c",
            config,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(onnx),
            "--report",
            str(export_report),
        ],
        dependencies=(training_task.name,),
        completion_kind="onnx",
        completion_paths=(onnx, export_report),
        source_checkpoint=checkpoint,
    )
    engine = options.output_root / "engines" / f"{name}.engine"
    trt_report = options.output_root / "benchmarks" / f"{name}-tensorrt.json"
    trt_log = options.output_root / "benchmarks" / f"{name}-trtexec.log"
    tensorrt = Task(
        name=f"tensorrt_{name}",
        stage="deployment",
        command=[
            options.python,
            "tools/benchmark_tensorrt.py",
            "--onnx",
            str(onnx),
            "--onnx-report",
            str(export_report),
            "--engine",
            str(engine),
            "--fp16",
            "--output",
            str(trt_report),
            "--log",
            str(trt_log),
        ],
        dependencies=(export.name,),
        completion_kind="tensorrt",
        completion_paths=(engine, trt_report, trt_log),
        source_checkpoint=checkpoint,
        source_artifact=onnx,
    )
    return export, tensorrt


def _onnx_evaluation_task(
    options: MatrixOptions,
    *,
    name: str,
    config: str,
    training_task: Task,
    export_task: Task,
    torch_evaluation_task: Task,
) -> Task:
    checkpoint = training_task.output_dir / "checkpoint_best.pth"
    onnx, export_report = export_task.completion_paths
    output = options.output_root / "evaluations" / f"{name}-onnx.json"
    command = [
        options.python,
        "tools/evaluate_onnx.py",
        "-c",
        config,
        "--onnx",
        str(onnx),
        "--onnx-report",
        str(export_report),
        "--torch-report",
        str(torch_evaluation_task.completion_paths[0]),
        "--dataset",
        f"wireframe={options.wireframe}",
        "--dataset",
        f"york={options.york}",
        "--num-workers",
        str(options.workers),
        "--max-ap-delta",
        "0.05",
        "--output",
        str(output),
    ]
    if options.device.startswith("cuda"):
        command.insert(-2, "--cuda-ort")
    return Task(
        name=f"evaluate_onnx_{name}",
        stage="deployment",
        command=command,
        dependencies=(export_task.name, torch_evaluation_task.name),
        completion_kind="onnx_ap",
        completion_paths=(output,),
        source_checkpoint=checkpoint,
        source_artifact=onnx,
        comparison_artifact=torch_evaluation_task.completion_paths[0],
    )


def _measurement_tasks(
    options: MatrixOptions,
    *,
    config: str,
    training_task: Task,
    evaluation_stage: str = "deployment",
    benchmark_stage: str = "deployment",
) -> tuple[Task, Task, Task, Task, Task]:
    evaluation = _evaluation_task(
        options,
        name=training_task.name,
        stage=evaluation_stage,
        config=config,
        training_task=training_task,
    )
    benchmark = _benchmark_task(
        options,
        name=training_task.name,
        config=config,
        training_task=training_task,
        stage=benchmark_stage,
    )
    export, tensorrt = _export_tasks(
        options,
        name=training_task.name,
        config=config,
        training_task=training_task,
    )
    onnx_evaluation = _onnx_evaluation_task(
        options,
        name=training_task.name,
        config=config,
        training_task=training_task,
        export_task=export,
        torch_evaluation_task=evaluation,
    )
    tensorrt.dependencies = (export.name, onnx_evaluation.name)
    return evaluation, benchmark, export, onnx_evaluation, tensorrt


def _pareto_task(
    options: MatrixOptions,
    *,
    variant: str,
    records: tuple[tuple[str, Task, Task], ...],
) -> Task:
    output = options.output_root / "pareto" / f"lineae_{variant.lower()}_tuning.json"
    pareto_records = tuple(
        (label, evaluation.completion_paths[0], benchmark.completion_paths[0])
        for label, evaluation, benchmark in records
    )
    command = [options.python, "tools/analyze_pareto.py"]
    for label, evaluation, benchmark in pareto_records:
        command.extend(["--model", f"{label}={evaluation},{benchmark}"])
    command.extend(["--output", str(output)])
    return Task(
        name=f"pareto_lineae_{variant.lower()}_tuning",
        stage="tuning",
        command=command,
        dependencies=tuple(
            task.name
            for _, evaluation, benchmark in records
            for task in (evaluation, benchmark)
        ),
        completion_kind="pareto",
        completion_paths=(output,),
        pareto_records=pareto_records,
    )


def build_matrix(options: MatrixOptions) -> list[Task]:
    tasks = []
    linea = _train_task(
        options,
        name="linea_hgnetv2_n_nokd",
        stage="baseline",
        config="configs/linea/linea_hgnetv2_n.py",
    )
    s_baseline = _train_task(
        options,
        name="lineae_s_nokd",
        stage="baseline",
        config="configs/lineae/lineae_s.py",
    )
    xl = _train_task(
        options,
        name="lineae_xl_teacher_candidate",
        stage="teacher",
        config="configs/lineae/lineae_xl.py",
        dependencies=(linea.name, s_baseline.name),
    )
    xl_frozen = _train_task(
        options,
        name="lineae_xl_frozen_ablation",
        stage="teacher",
        config="configs/lineae/ablations/lineae_xl_frozen.py",
        dependencies=(linea.name, s_baseline.name),
    )
    teacher_candidates = [
        (xl, "configs/lineae/lineae_xl.py"),
        (xl_frozen, "configs/lineae/ablations/lineae_xl_frozen.py"),
    ]
    tasks.extend((linea, s_baseline, xl, xl_frozen))
    if options.include_ablations:
        for name, config in (
            ("lineae_xl_ema_ablation", "configs/lineae/ablations/lineae_xl_ema.py"),
            (
                "lineae_xl_photometric_ablation",
                "configs/lineae/ablations/lineae_xl_photometric.py",
            ),
        ):
            candidate = _train_task(
                options,
                name=name,
                stage="teacher",
                config=config,
                dependencies=(linea.name, s_baseline.name),
            )
            tasks.append(candidate)
            teacher_candidates.append((candidate, config))

    linea_eval = _evaluation_task(
        options,
        name=linea.name,
        stage="teacher",
        config="configs/linea/linea_hgnetv2_n.py",
        training_task=linea,
    )
    candidate_evaluations = []
    for candidate, config in teacher_candidates:
        evaluation = _evaluation_task(
            options,
            name=candidate.name,
            stage="teacher",
            config=config,
            training_task=candidate,
        )
        candidate_evaluations.append((candidate, config, evaluation))

    candidate_arguments = []
    for candidate, config, evaluation in candidate_evaluations:
        candidate_arguments.extend([
            "--candidate",
            ",".join((
                config,
                str(candidate.output_dir / "checkpoint_best.pth"),
                str(evaluation.completion_paths[0]),
            )),
        ])
    qualification_report = options.teacher.with_suffix(".qualification.json")
    qualify = Task(
        name="qualify_lineae_xl",
        stage="teacher",
        command=[
            options.python,
            "tools/qualify_best_teacher.py",
            *candidate_arguments,
            "--baseline-metrics",
            str(linea_eval.completion_paths[0]),
            "--baseline-checkpoint",
            str(linea.output_dir / "checkpoint_best.pth"),
            "--baseline-config",
            "configs/linea/linea_hgnetv2_n.py",
            "--minimum-ap10-gain",
            str(options.minimum_ap10_gain),
            "--device",
            options.device,
            "--output",
            str(options.teacher),
        ],
        dependencies=(
            linea_eval.name,
            *(evaluation.name for _, _, evaluation in candidate_evaluations),
        ),
        completion_kind="teacher",
        completion_paths=(options.teacher, qualification_report),
        source_checkpoint=linea.output_dir / "checkpoint_best.pth",
        candidate_checkpoints=tuple(
            candidate.output_dir / "checkpoint_best.pth"
            for candidate, _, _ in candidate_evaluations
        ),
    )
    tasks.extend((
        linea_eval,
        *(evaluation for _, _, evaluation in candidate_evaluations),
        qualify,
    ))

    previous_kd = qualify.name
    student_train_tasks = []
    for variant in STUDENT_ORDER:
        lower = variant.lower()
        baseline_config = f"configs/lineae/lineae_{lower}.py"
        if variant == "S":
            no_kd = s_baseline
            kd_dependencies = (no_kd.name, qualify.name, previous_kd)
        else:
            no_kd = _train_task(
                options,
                name=f"lineae_{lower}_nokd",
                stage="distillation",
                config=baseline_config,
                dependencies=(previous_kd,),
            )
            tasks.append(no_kd)
            kd_dependencies = (no_kd.name, qualify.name)
        kd = _train_task(
            options,
            name=f"lineae_{lower}_kd",
            stage="distillation",
            config=f"configs/lineae/distill/lineae_{lower}.py",
            dependencies=kd_dependencies,
            config_options=(f"distill_teacher_checkpoint={options.teacher}",),
        )
        tasks.append(kd)
        student_train_tasks.extend(((no_kd, baseline_config), (kd, f"configs/lineae/distill/lineae_{lower}.py")))
        previous_kd = kd.name

    if options.include_ablations:
        training_by_name = {
            task.name: task
            for task, _ in student_train_tasks
        }
        x_intermediate = _train_task(
            options,
            name="lineae_x_intermediate_ablation",
            stage="distillation",
            config="configs/lineae/ablations/lineae_x_intermediate.py",
            dependencies=(training_by_name["lineae_x_nokd"].name,),
        )
        x_feature_kd = _train_task(
            options,
            name="lineae_x_feature_kd_ablation",
            stage="distillation",
            config="configs/lineae/ablations/lineae_x_feature_kd.py",
            dependencies=(training_by_name["lineae_x_kd"].name, qualify.name),
            config_options=(f"distill_teacher_checkpoint={options.teacher}",),
        )
        tasks.extend((x_intermediate, x_feature_kd))
        student_train_tasks.extend((
            (x_intermediate, "configs/lineae/ablations/lineae_x_intermediate.py"),
            (x_feature_kd, "configs/lineae/ablations/lineae_x_feature_kd.py"),
        ))

    training_by_name = {task.name: task for task, _ in student_train_tasks}
    evaluations_by_name = {}
    benchmarks_by_name = {}
    direct_kd_names = {
        f"lineae_{variant.lower()}_kd"
        for variant in TUNING_STUDENT_ORDER
    }
    for training_task, config in student_train_tasks:
        if (
            options.include_cascade
            and training_task.name in {"lineae_x_nokd", "lineae_x_kd"}
        ):
            evaluation_stage = "cascade"
        elif options.include_tuning and training_task.name in direct_kd_names:
            evaluation_stage = "tuning"
        else:
            evaluation_stage = "deployment"
        benchmark_stage = (
            "tuning"
            if options.include_tuning and training_task.name in direct_kd_names
            else "deployment"
        )
        measurements = _measurement_tasks(
            options,
            config=config,
            training_task=training_task,
            evaluation_stage=evaluation_stage,
            benchmark_stage=benchmark_stage,
        )
        evaluations_by_name[training_task.name] = measurements[0]
        benchmarks_by_name[training_task.name] = measurements[1]
        tasks.extend(measurements)

    if options.include_cascade:
        x_no_kd = training_by_name["lineae_x_nokd"]
        x_kd = training_by_name["lineae_x_kd"]
        x_no_kd_evaluation = evaluations_by_name[x_no_kd.name]
        x_kd_evaluation = evaluations_by_name[x_kd.name]
        cascade_report = options.cascade_teacher.with_suffix(".qualification.json")
        qualify_x = Task(
            name="qualify_lineae_x_cascade_teacher",
            stage="cascade",
            command=[
                options.python,
                "tools/qualify_teacher.py",
                "-c",
                "configs/lineae/distill/lineae_x.py",
                "--inference-config",
                "configs/lineae/lineae_x.py",
                "--candidate",
                str(x_kd.output_dir / "checkpoint_best.pth"),
                "--candidate-metrics",
                str(x_kd_evaluation.completion_paths[0]),
                "--baseline-checkpoint",
                str(x_no_kd.output_dir / "checkpoint_best.pth"),
                "--baseline-metrics",
                str(x_no_kd_evaluation.completion_paths[0]),
                "--baseline-config",
                "configs/lineae/lineae_x.py",
                "--minimum-ap10-gain",
                str(options.minimum_ap10_gain),
                "--device",
                options.device,
                "--output",
                str(options.cascade_teacher),
            ],
            dependencies=(x_no_kd_evaluation.name, x_kd_evaluation.name),
            completion_kind="teacher",
            completion_paths=(options.cascade_teacher, cascade_report),
            source_checkpoint=x_no_kd.output_dir / "checkpoint_best.pth",
            candidate_checkpoints=(x_kd.output_dir / "checkpoint_best.pth",),
        )
        tasks.append(qualify_x)

        previous_cascade = qualify_x.name
        for variant in CASCADE_STUDENT_ORDER[1:]:
            lower = variant.lower()
            direct_kd = training_by_name[f"lineae_{lower}_kd"]
            config = f"configs/lineae/cascade/lineae_{lower}.py"
            cascade = _train_task(
                options,
                name=f"lineae_{lower}_cascade_x",
                stage="cascade",
                config=config,
                dependencies=(direct_kd.name, previous_cascade),
                config_options=(
                    "distill_teacher_config=configs/lineae/lineae_x.py",
                    f"distill_teacher_checkpoint={options.cascade_teacher}",
                ),
            )
            tasks.append(cascade)
            tasks.extend(_measurement_tasks(
                options,
                config=config,
                training_task=cascade,
            ))
            previous_cascade = cascade.name

    if options.include_tuning:
        previous_tuning = previous_cascade if options.include_cascade else qualify.name
        for variant in TUNING_STUDENT_ORDER:
            lower = variant.lower()
            direct_kd = training_by_name[f"lineae_{lower}_kd"]
            records = [(
                f"{variant}-direct-xl",
                evaluations_by_name[direct_kd.name],
                benchmarks_by_name[direct_kd.name],
            )]
            for profile in ("speed", "accuracy"):
                name = f"lineae_{lower}_tune_{profile}"
                config = f"configs/lineae/tuning/{name}.py"
                candidate = _train_task(
                    options,
                    name=name,
                    stage="tuning",
                    config=config,
                    dependencies=(direct_kd.name, previous_tuning),
                    config_options=(f"distill_teacher_checkpoint={options.teacher}",),
                )
                evaluation = _evaluation_task(
                    options,
                    name=name,
                    stage="tuning",
                    config=config,
                    training_task=candidate,
                )
                benchmark = _benchmark_task(
                    options,
                    name=name,
                    config=config,
                    training_task=candidate,
                    stage="tuning",
                )
                tasks.extend((candidate, evaluation, benchmark))
                records.append((f"{variant}-{profile}", evaluation, benchmark))
                previous_tuning = benchmark.name
            pareto = _pareto_task(
                options,
                variant=variant,
                records=tuple(records),
            )
            tasks.append(pareto)
            previous_tuning = pareto.name
    return tasks


def selected_tasks(tasks: list[Task], stage: str) -> list[Task]:
    if stage == "all":
        return tasks
    maximum_rank = STAGE_RANK[stage]
    return [task for task in tasks if STAGE_RANK[task.stage] <= maximum_rank]


def write_plan(tasks: list[Task], options: MatrixOptions, path: Path) -> dict:
    report = {
        "format": "lineae_experiment_matrix_v1",
        "hardware_profile": "single_gpu_96gb",
        "include_ablations": options.include_ablations,
        "include_cascade": options.include_cascade,
        "include_tuning": options.include_tuning,
        "seed": options.seed,
        "wireframe": str(options.wireframe.resolve()),
        "york": str(options.york.resolve()),
        "teacher": str(options.teacher.resolve()),
        "cascade_teacher": str(options.cascade_teacher.resolve()),
        "tasks": [task.record() for task in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGE_RANK, "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/full_matrix_seed42"))
    parser.add_argument("--wireframe", type=Path, default=Path("data/wireframe_processed"))
    parser.add_argument("--york", type=Path, default=Path("data/york_processed"))
    parser.add_argument("--teacher", type=Path, default=Path("ckpts/lineae_xl_teacher.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-ap10-gain", type=float, default=0.0)
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="add EMA, photometric, intermediate-fusion, and feature-KD experiments",
    )
    parser.add_argument(
        "--include-cascade",
        action="store_true",
        help="after direct-XL controls, qualify X and distill smaller variants from it",
    )
    parser.add_argument(
        "--include-tuning",
        action="store_true",
        help="screen speed/accuracy bundles for every directly distilled student",
    )
    parser.add_argument(
        "--cascade-teacher",
        type=Path,
        default=Path("ckpts/lineae_x_teacher.pth"),
        help="qualified X teacher artifact written/read by the optional cascade",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--extra-train-arg", action="append", default=[])
    args = parser.parse_args()
    options = MatrixOptions(
        output_root=args.output_root,
        wireframe=args.wireframe,
        york=args.york,
        teacher=args.teacher,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        amp=not args.no_amp,
        minimum_ap10_gain=args.minimum_ap10_gain,
        include_ablations=args.include_ablations,
        include_cascade=args.include_cascade,
        include_tuning=args.include_tuning,
        cascade_teacher=args.cascade_teacher,
        extra_train_args=args.extra_train_arg,
    )
    tasks = selected_tasks(build_matrix(options), args.stage)
    plan_path = options.output_root / "matrix_plan.json"
    report = write_plan(tasks, options, plan_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
