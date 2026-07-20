"""Select the best matched XL recipe and pass it through strict qualification."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from models.lineae.variants import validate_variant_config
from tools.qualify_teacher import _load_report, _validate_matched_evaluation, qualify
from util.experiment import sha256_file
from util.slconfig import SLConfig


@dataclass(frozen=True)
class Candidate:
    config: Path
    checkpoint: Path
    metrics: Path


def candidate_argument(value: str) -> Candidate:
    parts = value.split(",", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "candidate must be CONFIG,CHECKPOINT,METRICS_JSON"
        )
    return Candidate(*(Path(part) for part in parts))


def select_candidate(candidates: list[Candidate], baseline_metrics: Path) -> Candidate:
    if not candidates:
        raise ValueError("at least one XL candidate is required")
    baseline = _load_report(baseline_metrics)
    scored = []
    for candidate in candidates:
        if not candidate.checkpoint.is_file():
            raise FileNotFoundError(f"XL candidate checkpoint not found: {candidate.checkpoint}")
        report = _load_report(candidate.metrics)
        if report.get("checkpoint_sha256") != sha256_file(candidate.checkpoint):
            raise ValueError(f"stale metrics for XL candidate: {candidate.checkpoint}")
        expected_config = str(candidate.config.resolve())
        if report.get("config") != expected_config:
            raise ValueError(
                f"candidate metrics config mismatch: {report.get('config')!r} "
                f"!= {expected_config!r}"
            )
        config = SLConfig.fromfile(str(candidate.config))
        spec = validate_variant_config(config)
        if spec is None or spec.name != "XL":
            raise ValueError(f"teacher candidate is not LINEAE-XL: {candidate.config}")
        _validate_matched_evaluation(report, baseline)
        scored.append((float(report["datasets"]["wireframe"]["sap10"]), candidate))
    return max(scored, key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=candidate_argument, action="append", required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--baseline-config", default="configs/linea/linea_hgnetv2_n.py"
    )
    parser.add_argument("--inference-config", default="configs/lineae/lineae_xl.py")
    parser.add_argument("--minimum-ap10-gain", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("ckpts/lineae_xl_teacher.pth"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = select_candidate(args.candidate, args.baseline_metrics)
    result = qualify(SimpleNamespace(
        config=str(selected.config),
        candidate=selected.checkpoint,
        candidate_metrics=selected.metrics,
        baseline_metrics=args.baseline_metrics,
        baseline_checkpoint=args.baseline_checkpoint,
        baseline_config=args.baseline_config,
        inference_config=args.inference_config,
        minimum_ap10_gain=args.minimum_ap10_gain,
        device=args.device,
        output=args.output,
        force=args.force,
    ))
    result["selected_config"] = str(selected.config.resolve())
    result["selected_checkpoint"] = str(selected.checkpoint.resolve())
    args.output.with_suffix(".qualification.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
