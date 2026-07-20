"""Strict schemas for artifacts used to make LINEAE experiment decisions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def _mapping(value, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sha256(value, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def _number(value, label: str, *, minimum=None, maximum=None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return number


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _checkpoint(report: Mapping, expected: str | None = None) -> str:
    checkpoint_hash = _sha256(report.get("checkpoint_sha256"), "checkpoint_sha256")
    if expected is not None and checkpoint_hash != expected:
        raise ValueError("artifact checkpoint SHA-256 does not match its source")
    return checkpoint_hash


def _checkpoint_record(value, label: str) -> str:
    record = _mapping(value, label)
    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label}.path must be a non-empty string")
    return _sha256(record.get("sha256"), f"{label}.sha256")


def _positive_integer(value, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _sample_summary(value, label: str, count: int, *, maximum=None) -> list[float]:
    summary = _mapping(value, label)
    samples = summary.get("samples")
    if not isinstance(samples, list) or len(samples) != count:
        raise ValueError(f"{label}.samples must contain exactly {count} values")
    numeric = [
        _number(sample, f"{label}.samples[{index}]", minimum=0, maximum=maximum)
        for index, sample in enumerate(samples)
    ]
    expected = {
        "mean": math.fsum(numeric) / len(numeric),
        "p50": _percentile(numeric, 50),
        "p95": _percentile(numeric, 95),
        "min": min(numeric),
        "max": max(numeric),
    }
    for metric, expected_value in expected.items():
        actual = _number(summary.get(metric), f"{label}.{metric}", minimum=0, maximum=maximum)
        if not math.isclose(actual, expected_value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{label}.{metric} does not match raw samples")
    return numeric


def _config(report: Mapping, expected: str | None = None) -> str:
    config = report.get("config")
    if not isinstance(config, str) or not config:
        raise ValueError("artifact lacks config")
    if expected is not None and config != expected:
        raise ValueError("artifact config does not match its task")
    return config


def _datasets(report: Mapping, required: Iterable[str]) -> None:
    datasets = _mapping(report.get("datasets"), "datasets")
    for name in required:
        dataset = _mapping(datasets.get(name), f"datasets.{name}")
        _sha256(dataset.get("annotation_sha256"), f"datasets.{name}.annotation_sha256")
        samples = dataset.get("samples")
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise ValueError(f"datasets.{name}.samples must be a positive integer")
        for metric in ("sap5", "sap10", "sap15"):
            _number(dataset.get(metric), f"datasets.{name}.{metric}", minimum=0, maximum=100)


def validate_evaluation_report(
    report,
    *,
    expected_checkpoint: str | None = None,
    expected_config: str | None = None,
    required_datasets: Iterable[str] = ("wireframe", "york"),
) -> None:
    report = _mapping(report, "evaluation report")
    if report.get("format") != "lineae_evaluation_v1":
        raise ValueError("unsupported evaluation report format")
    _checkpoint(report, expected_checkpoint)
    _config(report, expected_config)
    _datasets(report, required_datasets)


def validate_torch_benchmark_report(
    report,
    *,
    expected_checkpoint: str | None = None,
    expected_config: str | None = None,
    require_cuda: bool = False,
    require_samples: bool = False,
) -> None:
    report = _mapping(report, "PyTorch benchmark report")
    if report.get("format") != "lineae_torch_benchmark_v1":
        raise ValueError("unsupported PyTorch benchmark report format")
    _checkpoint(report, expected_checkpoint)
    _config(report, expected_config)
    if report.get("batch_size") != 1:
        raise ValueError("PyTorch benchmark must use batch size one")
    if report.get("deploy_mode") is not True:
        raise ValueError("PyTorch benchmark did not use the fused deploy model")
    _positive_integer(report.get("num_select"), "num_select")
    spatial_size = report.get("spatial_size")
    if not isinstance(spatial_size, int) or isinstance(spatial_size, bool) or spatial_size <= 0:
        raise ValueError("PyTorch benchmark spatial_size must be a positive integer")
    iterations = report.get("iterations")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise ValueError("PyTorch benchmark iterations must be a positive integer")
    warmup = report.get("warmup")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("PyTorch benchmark warmup must be a non-negative integer")
    latency = _mapping(report.get("latency_ms"), "latency_ms")
    for metric in ("p50", "p95"):
        _number(latency.get(metric), f"latency_ms.{metric}", minimum=0)
    parameters = report.get("parameters")
    if not isinstance(parameters, int) or isinstance(parameters, bool) or parameters <= 0:
        raise ValueError("PyTorch benchmark parameters must be a positive integer")
    if require_cuda:
        if not str(report.get("device", "")).startswith("cuda") or not report.get("gpu"):
            raise ValueError("PyTorch benchmark did not run on CUDA")
        _number(report.get("peak_memory_mib"), "peak_memory_mib", minimum=0)
    if require_samples:
        samples = report.get("samples_ms")
        if not isinstance(samples, list) or len(samples) != iterations:
            raise ValueError("PyTorch benchmark lacks the complete raw latency samples")
        for index, sample in enumerate(samples):
            _number(sample, f"samples_ms[{index}]", minimum=0)
        numeric_samples = [float(sample) for sample in samples]
        for metric, q in (("p50", 50), ("p95", 95)):
            if not math.isclose(
                float(latency[metric]),
                _percentile(numeric_samples, q),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(f"latency_ms.{metric} does not match raw samples")


def validate_onnx_export_report(
    report,
    *,
    expected_checkpoint: str | None = None,
    expected_onnx: str | None = None,
    expected_config: str | None = None,
    require_simplified: bool = True,
) -> None:
    report = _mapping(report, "ONNX export report")
    if report.get("format") != "lineae_onnx_export_v1":
        raise ValueError("unsupported ONNX export report format")
    _checkpoint(report, expected_checkpoint)
    _config(report, expected_config)
    onnx_hash = _sha256(report.get("onnx_sha256"), "onnx_sha256")
    if expected_onnx is not None and onnx_hash != expected_onnx:
        raise ValueError("ONNX report SHA-256 does not match its model")
    if report.get("onnxruntime_version") != "1.26.0":
        raise ValueError("ONNX export did not use onnxruntime 1.26.0")
    if report.get("deploy_mode") is not True:
        raise ValueError("ONNX export did not use the fused deploy model")
    if str(report.get("onnxsim_version", "")).lstrip("v") != "0.6.5":
        raise ValueError("ONNX export did not use onnxsim 0.6.5")
    if require_simplified and report.get("onnx_simplified") is not True:
        raise ValueError("ONNX export is not simplified")
    num_select = _positive_integer(report.get("num_select"), "num_select")
    seed = report.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("ONNX export seed must be a non-negative integer")
    parity = _mapping(report.get("parity"), "parity")
    for output in ("pred_logits", "pred_lines"):
        if parity.get(output) is not True:
            raise ValueError(f"ONNX {output} parity did not pass")
    shape = report.get("input_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or shape[0] != 1
        or not all(isinstance(value, int) and value > 0 for value in shape)
        or shape[-2] != shape[-1]
    ):
        raise ValueError("ONNX export input_shape must be fixed, square, and batch one")
    output_shapes = _mapping(report.get("output_shapes"), "output_shapes")
    logits_shape = output_shapes.get("pred_logits")
    lines_shape = output_shapes.get("pred_lines")
    if (
        not isinstance(logits_shape, list)
        or len(logits_shape) != 3
        or logits_shape[0] != 1
        or logits_shape[1] != num_select
        or not isinstance(logits_shape[2], int)
        or logits_shape[2] <= 0
    ):
        raise ValueError("ONNX pred_logits shape must be [1,num_select,C]")
    if lines_shape != [1, num_select, 4]:
        raise ValueError("ONNX pred_lines shape must be [1,num_select,4]")


def validate_onnx_evaluation_report(
    report,
    *,
    expected_checkpoint: str | None = None,
    expected_onnx: str | None = None,
    expected_torch_report: str | None = None,
    expected_config: str | None = None,
    require_cuda: bool = False,
) -> None:
    report = _mapping(report, "ONNX evaluation report")
    if report.get("format") != "lineae_onnx_evaluation_v1":
        raise ValueError("unsupported ONNX evaluation report format")
    _checkpoint(report, expected_checkpoint)
    _config(report, expected_config)
    onnx_hash = _sha256(report.get("onnx_sha256"), "onnx_sha256")
    if expected_onnx is not None and onnx_hash != expected_onnx:
        raise ValueError("ONNX evaluation SHA-256 does not match its model")
    torch_report_hash = _sha256(
        report.get("torch_report_sha256"),
        "torch_report_sha256",
    )
    if expected_torch_report is not None and torch_report_hash != expected_torch_report:
        raise ValueError("ONNX evaluation does not match its PyTorch report")
    providers = report.get("providers")
    if not isinstance(providers, list) or not providers:
        raise ValueError("ONNX evaluation lacks execution providers")
    if require_cuda and "CUDAExecutionProvider" not in providers:
        raise ValueError("ONNX evaluation did not activate CUDAExecutionProvider")
    if require_cuda and report.get("cpu_ep_fallback_disabled") is not True:
        raise ValueError("ONNX CUDA evaluation allowed CPU execution-provider fallback")
    if require_cuda and report.get("provider_options", {}).get(
        "CUDAExecutionProvider", {}
    ).get("use_tf32") != "0":
        raise ValueError("ONNX CUDA evaluation did not disable TF32")
    _datasets(report, ("wireframe", "york"))
    parity = _mapping(report.get("ap_parity"), "ap_parity")
    tolerance = _number(parity.get("tolerance"), "ap_parity.tolerance", minimum=0)
    maximum = _number(parity.get("maximum_delta"), "ap_parity.maximum_delta", minimum=0)
    absolute = _mapping(parity.get("absolute_delta"), "ap_parity.absolute_delta")
    deltas = []
    for dataset in ("wireframe", "york"):
        dataset_deltas = _mapping(
            absolute.get(dataset),
            f"ap_parity.absolute_delta.{dataset}",
        )
        for metric in ("sap5", "sap10", "sap15"):
            deltas.append(_number(
                dataset_deltas.get(metric),
                f"ap_parity.absolute_delta.{dataset}.{metric}",
                minimum=0,
            ))
    if not math.isclose(maximum, max(deltas), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("ap_parity.maximum_delta does not match absolute deltas")
    if maximum > tolerance:
        raise ValueError("ONNX evaluation exceeds its AP tolerance")


def validate_tensorrt_report(
    report,
    *,
    expected_checkpoint: str | None = None,
    expected_onnx: str | None = None,
    expected_engine: str | None = None,
    require_fp16: bool = False,
) -> None:
    report = _mapping(report, "TensorRT benchmark report")
    if report.get("format") != "lineae_tensorrt_benchmark_v1":
        raise ValueError("unsupported TensorRT benchmark report format")
    _checkpoint(report, expected_checkpoint)
    onnx_hash = _sha256(report.get("onnx_sha256"), "onnx_sha256")
    engine_hash = _sha256(report.get("engine_sha256"), "engine_sha256")
    if expected_onnx is not None and onnx_hash != expected_onnx:
        raise ValueError("TensorRT report ONNX SHA-256 does not match its source")
    if expected_engine is not None and engine_hash != expected_engine:
        raise ValueError("TensorRT report engine SHA-256 does not match its engine")
    if require_fp16 and report.get("fp16") is not True:
        raise ValueError("TensorRT benchmark did not use FP16")
    if report.get("tf32_disabled") is not True:
        raise ValueError("TensorRT benchmark did not disable TF32")
    if report.get("onnxruntime_version") != "1.26.0":
        raise ValueError("TensorRT parity did not use onnxruntime 1.26.0")
    parity = _mapping(report.get("parity"), "parity")
    for output in ("pred_logits", "pred_lines"):
        if parity.get(output) is not True:
            raise ValueError(f"TensorRT {output} parity did not pass")
    latency = _mapping(report.get("latency_ms"), "latency_ms")
    for metric in ("median", "percentile_95"):
        _number(latency.get(metric), f"latency_ms.{metric}", minimum=0)


_TRAINING_PROFILE_PHASES = (
    "transfer_ms",
    "student_supervised_ms",
    "teacher_forward_ms",
    "kd_loss_ms",
    "online_kd_ms",
    "backward_ms",
    "optimizer_ms",
    "step_ms",
    "throughput_images_per_second",
)


def validate_training_profile_report(report, *, require_cuda: bool = False) -> None:
    """Validate a bounded real-step profile and all decision-bearing raw samples."""
    report = _mapping(report, "training profile")
    if report.get("format") != "lineae_training_profile_v1":
        raise ValueError("unsupported training profile format")
    config = report.get("config")
    if not isinstance(config, str) or not config:
        raise ValueError("training profile lacks config")
    _sha256(report.get("resolved_config_sha256"), "resolved_config_sha256")
    variant = report.get("variant")
    if not isinstance(variant, str) or not variant:
        raise ValueError("training profile lacks variant")
    dataset = _mapping(report.get("dataset"), "dataset")
    if not isinstance(dataset.get("root"), str) or not dataset["root"]:
        raise ValueError("dataset.root must be a non-empty string")
    _sha256(dataset.get("annotation_sha256"), "dataset.annotation_sha256")
    _positive_integer(dataset.get("samples"), "dataset.samples")
    _checkpoint_record(report.get("initialization"), "initialization")
    if report.get("teacher") is not None:
        _checkpoint_record(report["teacher"], "teacher")

    device = report.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if require_cuda and (not device.startswith("cuda") or not report.get("gpu")):
        raise ValueError("training profile did not run on CUDA")
    for field in ("python", "torch"):
        if not isinstance(report.get(field), str) or not report[field]:
            raise ValueError(f"{field} must be a non-empty string")
    seed = _positive_integer(report.get("seed"), "seed", allow_zero=True)
    del seed
    _positive_integer(report.get("epoch"), "epoch", allow_zero=True)
    start_global_step = _positive_integer(
        report.get("start_global_step"),
        "start_global_step",
        allow_zero=True,
    )
    warmup = _positive_integer(report.get("warmup"), "warmup", allow_zero=True)
    iterations = _positive_integer(report.get("iterations"), "iterations")
    batch_size = _positive_integer(report.get("batch_size"), "batch_size")
    if report.get("gradient_accumulation_steps") != 1:
        raise ValueError("training profile must use one-step gradient accumulation")
    if not isinstance(report.get("optimizer_fused"), bool):
        raise ValueError("optimizer_fused must be boolean")
    trainable_depth = report.get("trainable_depth")
    if (
        isinstance(trainable_depth, bool)
        or not isinstance(trainable_depth, int)
        or trainable_depth < -1
    ):
        raise ValueError("trainable_depth must be an integer >= -1")
    _positive_integer(report.get("trainable_parameters"), "trainable_parameters")

    phase_ms = _mapping(report.get("phase_ms"), "phase_ms")
    phase_samples = {
        phase: _sample_summary(phase_ms.get(phase), f"phase_ms.{phase}", iterations)
        for phase in _TRAINING_PROFILE_PHASES
    }
    if any(value <= 0 for value in phase_samples["step_ms"]):
        raise ValueError("phase_ms.step_ms samples must be positive")
    if any(value <= 0 for value in phase_samples["throughput_images_per_second"]):
        raise ValueError("throughput samples must be positive")
    for index, (teacher, kd_loss, online) in enumerate(zip(
        phase_samples["teacher_forward_ms"],
        phase_samples["kd_loss_ms"],
        phase_samples["online_kd_ms"],
    )):
        if not math.isclose(online, teacher + kd_loss, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"phase_ms.online_kd_ms.samples[{index}] is inconsistent")

    peak_memory = report.get("peak_memory_mib")
    if require_cuda and peak_memory is None:
        raise ValueError("CUDA training profile lacks peak memory samples")
    if peak_memory is not None:
        _sample_summary(peak_memory, "peak_memory_mib", iterations)
    kd_fraction = _sample_summary(
        report.get("online_kd_fraction"),
        "online_kd_fraction",
        iterations,
        maximum=1,
    )

    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != iterations:
        raise ValueError(f"samples must contain exactly {iterations} measured steps")
    has_teacher = report.get("teacher") is not None
    temperature_steps = report.get("distill_temperature_steps_resolved")
    if has_teacher:
        _positive_integer(
            temperature_steps,
            "distill_temperature_steps_resolved",
            allow_zero=True,
        )
    elif temperature_steps is not None:
        raise ValueError("no-KD training profile contains a temperature horizon")
    for index, value in enumerate(samples):
        sample = _mapping(value, f"samples[{index}]")
        for phase in _TRAINING_PROFILE_PHASES:
            measured = _number(sample.get(phase), f"samples[{index}].{phase}", minimum=0)
            if not math.isclose(
                measured,
                phase_samples[phase][index],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(f"samples[{index}].{phase} does not match phase summary")
        if sample.get("optimizer_stepped") is not True:
            raise ValueError(f"samples[{index}] did not complete an optimizer step")
        if sample.get("batch_size") != batch_size:
            raise ValueError(f"samples[{index}].batch_size does not match the report")
        input_size = sample.get("input_size")
        if (
            not isinstance(input_size, list)
            or len(input_size) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in input_size
            )
        ):
            raise ValueError(f"samples[{index}].input_size is invalid")
        expected_step = start_global_step + warmup + index
        if sample.get("global_step") != expected_step:
            raise ValueError(f"samples[{index}].global_step is not contiguous")
        _number(sample.get("loss"), f"samples[{index}].loss")
        expected_fraction = (
            phase_samples["online_kd_ms"][index] / phase_samples["step_ms"][index]
        )
        if not math.isclose(
            kd_fraction[index],
            expected_fraction,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"online_kd_fraction.samples[{index}] is inconsistent")
        expected_throughput = batch_size * 1000.0 / phase_samples["step_ms"][index]
        if not math.isclose(
            phase_samples["throughput_images_per_second"][index],
            expected_throughput,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"samples[{index}] throughput is inconsistent")
        if has_teacher:
            _positive_integer(
                sample.get("kd_matches"),
                f"samples[{index}].kd_matches",
                allow_zero=True,
            )
            _number(sample.get("kd_weight"), f"samples[{index}].kd_weight", minimum=0)
            _number(
                sample.get("kd_temperature"),
                f"samples[{index}].kd_temperature",
                minimum=0,
            )
        elif (
            sample.get("kd_matches") is not None
            or sample.get("kd_weight") != 0.0
            or sample.get("kd_temperature") is not None
            or phase_samples["online_kd_ms"][index] != 0.0
        ):
            raise ValueError(f"samples[{index}] contains KD data without a teacher")


def validate_training_profile_comparison(report) -> None:
    report = _mapping(report, "training profile comparison")
    if report.get("format") != "lineae_training_profile_comparison_v1":
        raise ValueError("unsupported training profile comparison format")
    for label in ("baseline", "distilled"):
        source = _mapping(report.get(label), label)
        if not isinstance(source.get("path"), str) or not source["path"]:
            raise ValueError(f"{label}.path must be a non-empty string")
        _sha256(source.get("sha256"), f"{label}.sha256")
        if not isinstance(source.get("config"), str) or not source["config"]:
            raise ValueError(f"{label}.config must be a non-empty string")
        _sha256(source.get("resolved_config_sha256"), f"{label}.resolved_config_sha256")
    _sha256(report["distilled"].get("teacher_sha256"), "distilled.teacher_sha256")
    _sha256(report.get("dataset_annotation_sha256"), "dataset_annotation_sha256")
    _sha256(report.get("initialization_sha256"), "initialization_sha256")
    context = _mapping(report.get("matched_context"), "matched_context")
    _positive_integer(context.get("iterations"), "matched_context.iterations")
    _positive_integer(context.get("batch_size"), "matched_context.batch_size")
    if context.get("gradient_accumulation_steps") != 1:
        raise ValueError("comparison must describe one-step gradient accumulation")
    input_sizes = report.get("input_sizes")
    if not isinstance(input_sizes, list) or len(input_sizes) != context["iterations"]:
        raise ValueError("input_sizes must match the measured iteration count")
    for index, shape in enumerate(input_sizes):
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in shape
            )
        ):
            raise ValueError(f"input_sizes[{index}] is invalid")
    phase_delta = _mapping(report.get("phase_delta_ms"), "phase_delta_ms")
    for phase in (
        "transfer_ms",
        "student_supervised_ms",
        "teacher_forward_ms",
        "kd_loss_ms",
        "online_kd_ms",
        "backward_ms",
        "optimizer_ms",
        "step_ms",
    ):
        _number(phase_delta.get(phase), f"phase_delta_ms.{phase}")
    _number(report.get("direct_online_kd_ms"), "direct_online_kd_ms", minimum=0)
    _number(
        report.get("direct_online_kd_fraction"),
        "direct_online_kd_fraction",
        minimum=0,
        maximum=1,
    )
    _number(report.get("observed_step_overhead_ms"), "observed_step_overhead_ms")
    _number(
        report.get("observed_step_overhead_percent"),
        "observed_step_overhead_percent",
    )
    _number(report.get("step_time_ratio"), "step_time_ratio", minimum=0)
    _number(report.get("throughput_ratio"), "throughput_ratio", minimum=0)
    if report.get("peak_memory_delta_mib") is not None:
        _number(report["peak_memory_delta_mib"], "peak_memory_delta_mib")


__all__ = [
    "validate_evaluation_report",
    "validate_onnx_evaluation_report",
    "validate_onnx_export_report",
    "validate_tensorrt_report",
    "validate_torch_benchmark_report",
    "validate_training_profile_comparison",
    "validate_training_profile_report",
]
