"""ONNX Runtime session construction without silent CUDA fallback."""

from __future__ import annotations


def create_ort_session(ort, model_path, *, require_cuda: bool):
    available = list(ort.get_available_providers())
    if require_cuda and "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider was requested but is unavailable; "
            f"available providers: {available}"
        )
    requested_names = ["CUDAExecutionProvider"] if require_cuda else ["CPUExecutionProvider"]
    provider_options = (
        {"CUDAExecutionProvider": {"use_tf32": "0"}}
        if require_cuda
        else {}
    )
    requested = [
        (name, provider_options[name]) if name in provider_options else name
        for name in requested_names
    ]
    session_options = None
    if require_cuda:
        session_options = ort.SessionOptions()
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=requested,
    )
    active = list(session.get_providers())
    if require_cuda and "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            "CUDAExecutionProvider was requested but failed to activate; "
            f"active providers: {active}"
        )
    return session, available, requested_names, provider_options


__all__ = ["create_ort_session"]
