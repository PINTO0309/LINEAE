import numpy as np
import pytest

from tools.deployment_parity import compare_line_sets
from util.artifact_validation import (
    validate_evaluation_report,
    validate_onnx_evaluation_report,
    validate_onnx_export_report,
    validate_tensorrt_report,
    validate_torch_benchmark_report,
)
from util.onnx_runtime import create_ort_session


def _datasets():
    return {
        name: {
            "annotation_sha256": "a" * 64,
            "samples": 10,
            "sap5": 50.0,
            "sap10": 51.0,
            "sap15": 52.0,
        }
        for name in ("wireframe", "york")
    }


def test_evaluation_schema_rejects_nonfinite_decision_metric():
    report = {
        "format": "lineae_evaluation_v1",
        "config": "/test/config.py",
        "checkpoint_sha256": "b" * 64,
        "datasets": _datasets(),
    }
    validate_evaluation_report(report)
    report["datasets"]["wireframe"]["sap10"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        validate_evaluation_report(report)


def test_deployment_parity_is_query_and_endpoint_order_invariant():
    logits = np.asarray([[[1.0, -1.0], [2.0, -2.0]]], dtype=np.float32)
    lines = np.asarray(
        [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]],
        dtype=np.float32,
    )
    actual_logits = logits[:, [1, 0]]
    actual_lines = lines[:, [1, 0]][..., [2, 3, 0, 1]]
    report = compare_line_sets(
        logits,
        lines,
        actual_logits,
        actual_lines,
        atol=1e-6,
        rtol=1e-6,
        max_outlier_fraction=0,
    )
    assert all(report["parity"].values())


def test_deployment_parity_uses_logits_to_disambiguate_duplicate_lines():
    logits = np.asarray([[[4.0, -2.0], [-3.0, 1.0]]], dtype=np.float32)
    lines = np.asarray(
        [[[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]],
        dtype=np.float32,
    )
    report = compare_line_sets(
        logits,
        lines,
        logits[:, [1, 0]],
        lines[:, [1, 0]],
        atol=1e-6,
        rtol=1e-6,
        max_outlier_fraction=0,
    )
    assert all(report["parity"].values())


def test_torch_benchmark_schema_requires_complete_raw_cuda_samples():
    report = {
        "format": "lineae_torch_benchmark_v1",
        "config": "/test/config.py",
        "checkpoint_sha256": "b" * 64,
        "device": "cuda:0",
        "gpu": "test GPU",
        "batch_size": 1,
        "deploy_mode": True,
        "num_select": 300,
        "spatial_size": 640,
        "warmup": 1,
        "iterations": 2,
        "parameters": 10,
        "peak_memory_mib": 100.0,
        "latency_ms": {"p50": 1.0, "p95": 2.0},
        "samples_ms": [1.0],
    }
    with pytest.raises(ValueError, match="complete raw latency samples"):
        validate_torch_benchmark_report(
            report,
            require_cuda=True,
            require_samples=True,
        )


def test_onnx_and_tensorrt_schemas_bind_runtime_versions_and_latency():
    onnx_report = {
        "format": "lineae_onnx_export_v1",
        "config": "/test/config.py",
        "checkpoint_sha256": "b" * 64,
        "onnx_sha256": "c" * 64,
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
        "parity": {"pred_logits": True, "pred_lines": True},
    }
    validate_onnx_export_report(onnx_report)
    onnx_report["onnxruntime_version"] = "1.25.0"
    with pytest.raises(ValueError, match="onnxruntime 1.26.0"):
        validate_onnx_export_report(onnx_report)

    trt_report = {
        "format": "lineae_tensorrt_benchmark_v1",
        "checkpoint_sha256": "b" * 64,
        "onnx_sha256": "c" * 64,
        "engine_sha256": "d" * 64,
        "fp16": True,
        "tf32_disabled": True,
        "onnxruntime_version": "1.26.0",
        "parity": {"pred_logits": True, "pred_lines": True},
        "latency_ms": {"median": 1.0, "percentile_95": None},
    }
    with pytest.raises(ValueError, match="percentile_95 must be numeric"):
        validate_tensorrt_report(trt_report, require_fp16=True)


def test_onnx_ap_schema_binds_torch_report_and_pure_cuda_execution():
    deltas = {
        dataset: {metric: 0.01 for metric in ("sap5", "sap10", "sap15")}
        for dataset in ("wireframe", "york")
    }
    report = {
        "format": "lineae_onnx_evaluation_v1",
        "config": "/test/config.py",
        "checkpoint_sha256": "a" * 64,
        "onnx_sha256": "b" * 64,
        "torch_report_sha256": "c" * 64,
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "provider_options": {"CUDAExecutionProvider": {"use_tf32": "0"}},
        "cpu_ep_fallback_disabled": True,
        "datasets": _datasets(),
        "ap_parity": {
            "tolerance": 0.05,
            "maximum_delta": 0.01,
            "absolute_delta": deltas,
        },
    }
    validate_onnx_evaluation_report(
        report,
        expected_checkpoint="a" * 64,
        expected_onnx="b" * 64,
        expected_torch_report="c" * 64,
        require_cuda=True,
    )
    report["cpu_ep_fallback_disabled"] = False
    with pytest.raises(ValueError, match="allowed CPU"):
        validate_onnx_evaluation_report(report, require_cuda=True)


class _FakeSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


class _FakeOrt:
    def __init__(self, available, active=None):
        self.available = available
        self.active = active
        self.requested = None
        self.session_options = None

    def get_available_providers(self):
        return self.available

    class SessionOptions:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    def InferenceSession(self, _path, *, sess_options, providers):
        self.requested = providers
        self.session_options = sess_options
        active = self.active if self.active is not None else [
            provider[0] if isinstance(provider, tuple) else provider
            for provider in providers
        ]
        return _FakeSession(active)


def test_cuda_ort_request_never_silently_falls_back_to_cpu():
    missing = _FakeOrt(["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="requested but is unavailable"):
        create_ort_session(missing, "model.onnx", require_cuda=True)

    failed = _FakeOrt(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        active=["CPUExecutionProvider"],
    )
    with pytest.raises(RuntimeError, match="failed to activate"):
        create_ort_session(failed, "model.onnx", require_cuda=True)

    runtime = _FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    session, _, requested, options = create_ort_session(
        runtime,
        "model.onnx",
        require_cuda=True,
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    assert requested[0] == "CUDAExecutionProvider"
    assert options["CUDAExecutionProvider"]["use_tf32"] == "0"
    assert runtime.requested[0] == (
        "CUDAExecutionProvider",
        {"use_tf32": "0"},
    )
    assert runtime.session_options.entries["session.disable_cpu_ep_fallback"] == "1"


def test_cpu_ort_request_is_explicit():
    runtime = _FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    session, available, requested, options = create_ort_session(
        runtime,
        "model.onnx",
        require_cuda=False,
    )
    assert available == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert requested == ["CPUExecutionProvider"]
    assert options == {}
    assert runtime.session_options is None
    assert session.get_providers() == ["CPUExecutionProvider"]
