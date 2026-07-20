"""Set-aware numeric parity for unordered LINEAE prediction queries."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from models.lineae.distillation import endpoint_swap, pairwise_endpoint_l1


def compare_line_sets(
    expected_logits,
    expected_lines,
    actual_logits,
    actual_lines,
    *,
    atol: float,
    rtol: float,
    max_outlier_fraction: float,
) -> dict:
    arrays = [
        np.asarray(value)
        for value in (expected_logits, expected_lines, actual_logits, actual_lines)
    ]
    expected_logits, expected_lines, actual_logits, actual_lines = arrays
    if expected_logits.shape != actual_logits.shape:
        raise ValueError(
            f"logit shape mismatch: {expected_logits.shape} != {actual_logits.shape}"
        )
    if expected_lines.shape != actual_lines.shape:
        raise ValueError(
            f"line shape mismatch: {expected_lines.shape} != {actual_lines.shape}"
        )
    if (
        expected_logits.ndim != 3
        or expected_lines.ndim != 3
        or expected_lines.shape[-1] != 4
        or expected_logits.shape[:2] != expected_lines.shape[:2]
    ):
        raise ValueError("LINEAE parity expects [B,Q,C] logits and [B,Q,4] lines")
    if atol < 0 or rtol < 0 or not 0 <= max_outlier_fraction <= 1:
        raise ValueError("invalid parity tolerance")

    matched_logits = []
    matched_expected_lines = []
    matched_actual_lines = []
    for batch_index in range(expected_logits.shape[0]):
        expected_line_tensor = torch.from_numpy(expected_lines[batch_index])
        actual_line_tensor = torch.from_numpy(actual_lines[batch_index])
        expected_logit_tensor = torch.from_numpy(expected_logits[batch_index]).float()
        actual_logit_tensor = torch.from_numpy(actual_logits[batch_index]).float()
        # Untrained and low-confidence models can emit several nearly identical
        # lines with very different confidence logits. Endpoint cost alone then
        # admits multiple optimal assignments and reports a false logit mismatch.
        # Joint cost keeps the comparison permutation-invariant while selecting
        # the corresponding query when either signal disambiguates it.
        line_cost = pairwise_endpoint_l1(expected_line_tensor, actual_line_tensor)
        logit_cost = torch.cdist(expected_logit_tensor, actual_logit_tensor, p=1)
        cost = line_cost + logit_cost
        expected_index, actual_index = linear_sum_assignment(cost.numpy())
        expected_selected = expected_line_tensor[expected_index]
        actual_selected = actual_line_tensor[actual_index]
        swapped = endpoint_swap(actual_selected)
        use_swapped = (
            (expected_selected - swapped).abs().sum(-1)
            < (expected_selected - actual_selected).abs().sum(-1)
        )
        actual_selected = torch.where(use_swapped[:, None], swapped, actual_selected)
        matched_expected_lines.append(expected_selected.numpy())
        matched_actual_lines.append(actual_selected.numpy())
        matched_logits.append((
            expected_logits[batch_index, expected_index],
            actual_logits[batch_index, actual_index],
        ))

    expected_lines_matched = np.concatenate(matched_expected_lines)
    actual_lines_matched = np.concatenate(matched_actual_lines)
    expected_logits_matched = np.concatenate([pair[0] for pair in matched_logits])
    actual_logits_matched = np.concatenate([pair[1] for pair in matched_logits])
    logit_delta = np.abs(expected_logits_matched - actual_logits_matched)
    line_delta = np.abs(expected_lines_matched - actual_lines_matched)
    logit_tolerance = atol + rtol * np.abs(expected_logits_matched)
    logit_outlier_fraction = float(np.mean(logit_delta > logit_tolerance))
    logits_match = (
        float(np.percentile(logit_delta, 99)) <= atol + rtol
        and logit_outlier_fraction <= max_outlier_fraction
    )
    lines_match = np.allclose(
        expected_lines_matched,
        actual_lines_matched,
        atol=atol,
        rtol=rtol,
    )
    return {
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_error": {
            "pred_logits": float(logit_delta.max()),
            "pred_lines": float(line_delta.max()),
        },
        "p99_abs_error": {
            "pred_logits": float(np.percentile(logit_delta, 99)),
            "pred_lines": float(np.percentile(line_delta, 99)),
        },
        "logit_outlier_fraction": logit_outlier_fraction,
        "max_outlier_fraction": float(max_outlier_fraction),
        "comparison": "joint-logit endpoint-swap-invariant Hungarian set matching",
        "parity": {
            "pred_logits": bool(logits_match),
            "pred_lines": bool(lines_match),
        },
    }


__all__ = ["compare_line_sets"]
