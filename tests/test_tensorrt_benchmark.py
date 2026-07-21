import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tools.benchmark_tensorrt as benchmark_module
from util.experiment import sha256_file


class _ReferenceSession:
    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_outputs(self):
        return [SimpleNamespace(name="pred_logits"), SimpleNamespace(name="pred_lines")]

    def run(self, _outputs, _feeds):
        return [
            np.asarray([[[1.0, -1.0], [2.0, -2.0]]], dtype=np.float32),
            np.asarray(
                [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]],
                dtype=np.float32,
            ),
        ]


def test_tensorrt_benchmark_binds_engine_latency_and_set_parity(tmp_path, monkeypatch):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    onnx_report = tmp_path / "model.parity.json"
    onnx_report.write_text(json.dumps({
        "format": "lineae_onnx_export_v2",
        "image_preprocess_schema": "opencv_rgb_inter_linear_v2",
        "opencv_version": "4.13.0",
        "config": "/test/config.py",
        "checkpoint_sha256": "a" * 64,
        "onnx_sha256": sha256_file(onnx),
        "onnxruntime_version": "1.26.0",
        "onnxsim_version": "v0.6.5",
        "onnx_simplified": True,
        "deploy_mode": True,
        "seed": 0,
        "input_shape": [1, 3, 32, 32],
        "num_select": 2,
        "output_shapes": {
            "pred_logits": [1, 2, 2],
            "pred_lines": [1, 2, 4],
        },
        "parity": {"pred_logits": True, "pred_lines": True},
    }), encoding="utf-8")
    monkeypatch.setattr(benchmark_module.shutil, "which", lambda _name: "/trtexec")
    monkeypatch.setattr(
        benchmark_module,
        "create_ort_session",
        lambda *_args, **_kwargs: (_ReferenceSession(), [], [], {}),
    )

    engine = tmp_path / "model.engine"

    def fake_run(command, **_kwargs):
        Path(next(value.split("=", 1)[1] for value in command if value.startswith("--saveEngine="))).write_bytes(
            b"engine"
        )
        output_path = Path(next(
            value.split("=", 1)[1]
            for value in command
            if value.startswith("--exportOutput=")
        ))
        output_path.write_text(json.dumps([
            {
                "name": "pred_logits",
                "dimensions": "1x2x2",
                "values": [2.0, -2.0, 1.0, -1.0],
            },
            {
                "name": "pred_lines",
                "dimensions": "1x2x4",
                "values": [0.7, 0.8, 0.5, 0.6, 0.3, 0.4, 0.1, 0.2],
            },
        ]), encoding="utf-8")
        stdout = "\n".join([
            "mean = 1.0 ms",
            "median = 1.1 ms",
            "percentile(90%) = 1.2 ms",
            "percentile(95%) = 1.3 ms",
            "percentile(99%) = 1.4 ms",
        ])
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_run)
    args = SimpleNamespace(
        trtexec="trtexec",
        onnx=onnx,
        onnx_report=onnx_report,
        engine=engine,
        output=tmp_path / "trt.json",
        log=tmp_path / "trt.log",
        fp16=True,
        warmup_ms=100,
        duration_seconds=1,
        atol=1e-6,
        rtol=1e-6,
        max_outlier_fraction=0.0,
    )

    report = benchmark_module.benchmark(args)

    assert report["format"] == "lineae_tensorrt_benchmark_v1"
    assert report["engine_sha256"] == sha256_file(engine)
    assert report["latency_ms"]["percentile_95"] == 1.3
    assert all(report["parity"].values())
    assert "--noTF32" in report["command"]
