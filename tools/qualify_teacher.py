"""Promote a LINEAE candidate only after recorded validation superiority."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch

from main import create
from models.lineae.backbones.base import unwrap_state_dict
from models.lineae.variants import validate_variant_config
from util.artifact_validation import validate_evaluation_report
from util.experiment import config_fingerprint, sha256_file
from util.slconfig import SLConfig
from util.training_state import atomic_torch_save


def _load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    validate_evaluation_report(report)
    checkpoint_hash = report.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError(f"qualification report {path} has no valid checkpoint SHA-256")
    for dataset in ("wireframe", "york"):
        if dataset not in report.get("datasets", {}):
            raise ValueError(f"qualification report {path} lacks dataset {dataset!r}")
        dataset_report = report["datasets"][dataset]
        annotation_hash = dataset_report.get("annotation_sha256")
        if not isinstance(annotation_hash, str) or len(annotation_hash) != 64:
            raise ValueError(
                f"qualification report {path} has no valid {dataset}.annotation_sha256"
            )
        if not isinstance(dataset_report.get("samples"), int) or dataset_report["samples"] <= 0:
            raise ValueError(f"qualification report {path} has invalid {dataset}.samples")
        for metric in ("sap5", "sap10", "sap15"):
            if metric not in dataset_report:
                raise ValueError(f"qualification report {path} lacks {dataset}.{metric}")
            value = float(dataset_report[metric])
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(
                    f"qualification report {path} has invalid {dataset}.{metric}={value!r}"
                )
    return report


def _validate_matched_evaluation(candidate: dict, baseline: dict) -> None:
    for dataset in ("wireframe", "york"):
        candidate_dataset = candidate["datasets"][dataset]
        baseline_dataset = baseline["datasets"][dataset]
        for field in ("annotation_sha256", "samples"):
            if candidate_dataset[field] != baseline_dataset[field]:
                raise ValueError(
                    f"candidate and baseline use different {dataset}.{field}: "
                    f"{candidate_dataset[field]!r} != {baseline_dataset[field]!r}"
                )


def _load_compatible_teacher_configs(
    source_config_path: Path,
    inference_config_path: Path,
):
    source_config = SLConfig.fromfile(str(source_config_path))
    source_spec = validate_variant_config(source_config)
    if source_spec is None:
        raise ValueError("teacher source config must select a registered LINEAE variant")
    inference_config = SLConfig.fromfile(str(inference_config_path))
    inference_spec = validate_variant_config(inference_config)
    if inference_spec is None:
        raise ValueError("teacher inference config must select a registered LINEAE variant")
    if source_spec.name != inference_spec.name:
        raise ValueError(
            "teacher source and inference configs select different variants: "
            f"{source_spec.name} != {inference_spec.name}"
        )
    return source_config, source_spec, inference_config


def qualify(args) -> dict:
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing teacher: {args.output}")
    if not math.isfinite(args.minimum_ap10_gain) or args.minimum_ap10_gain < 0:
        raise ValueError("minimum sAP10 gain must be finite and non-negative")
    candidate_path = Path(args.candidate)
    candidate_hash = sha256_file(candidate_path)
    candidate_report = _load_report(args.candidate_metrics)
    baseline_report = _load_report(args.baseline_metrics)
    if candidate_report.get("checkpoint_sha256") != candidate_hash:
        raise ValueError("candidate metrics were produced from a different checkpoint")
    baseline_checkpoint = getattr(args, "baseline_checkpoint", None)
    baseline_checkpoint_hash = None
    if baseline_checkpoint is not None:
        baseline_checkpoint = Path(baseline_checkpoint)
        if not baseline_checkpoint.is_file():
            raise FileNotFoundError(f"baseline checkpoint not found: {baseline_checkpoint}")
        baseline_checkpoint_hash = sha256_file(baseline_checkpoint)
        if baseline_report.get("checkpoint_sha256") != baseline_checkpoint_hash:
            raise ValueError("baseline metrics were produced from a different checkpoint")
    _validate_matched_evaluation(candidate_report, baseline_report)
    expected_config = str(Path(args.config).resolve())
    if candidate_report.get("config") != expected_config:
        raise ValueError(
            "candidate metrics were produced with a different config: "
            f"{candidate_report.get('config')!r} != {expected_config!r}"
        )
    baseline_config_path = Path(
        getattr(args, "baseline_config", "configs/linea/linea_hgnetv2_n.py")
    )
    if not baseline_config_path.is_file():
        raise FileNotFoundError(f"baseline config not found: {baseline_config_path}")
    expected_baseline_config = str(baseline_config_path.resolve())
    if baseline_report.get("config") != expected_baseline_config:
        raise ValueError(
            "baseline metrics were produced with a different config: "
            f"{baseline_report.get('config')!r} != {expected_baseline_config!r}"
        )
    candidate_ap10 = float(candidate_report["datasets"]["wireframe"]["sap10"])
    baseline_ap10 = float(baseline_report["datasets"]["wireframe"]["sap10"])
    if candidate_ap10 <= baseline_ap10 + args.minimum_ap10_gain:
        raise ValueError(
            f"candidate Wireframe sAP10 {candidate_ap10:.4f} does not exceed baseline "
            f"{baseline_ap10:.4f} by required gain {args.minimum_ap10_gain:.4f}"
        )
    if baseline_checkpoint_hash is None:
        raise ValueError("baseline checkpoint is required for teacher qualification")

    source_config_path = Path(args.config)
    inference_config_path = Path(
        getattr(args, "inference_config", "configs/lineae/lineae_xl.py")
    )
    config, spec, inference_config = _load_compatible_teacher_configs(
        source_config_path,
        inference_config_path,
    )
    checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, Mapping) and checkpoint.get("epoch_complete", True) is not True:
        raise ValueError("a partial-epoch checkpoint cannot be promoted as a teacher")
    config.pretrained = False
    model, _ = create(config, "modelname")
    state = unwrap_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(1, 3, spec.input_size, spec.input_size, generator=generator).to(device)
    with torch.inference_mode():
        first = model(images)
        model.load_state_dict(state, strict=True)
        second = model(images)
    for key in ("pred_logits", "pred_lines"):
        if not torch.equal(first[key], second[key]):
            raise RuntimeError(f"checkpoint reload changed {key}")

    inference_config_sha256 = config_fingerprint(inference_config)
    inference_config.pretrained = False
    if inference_config_path.resolve() == source_config_path.resolve():
        inference_outputs = second
    else:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        inference_model, _ = create(inference_config, "modelname")
        inference_model.load_state_dict(state, strict=True)
        inference_model.to(device).eval()
        with torch.inference_mode():
            inference_outputs = inference_model(images)
    for key in ("pred_logits", "pred_lines"):
        if not torch.equal(first[key], inference_outputs[key]):
            raise RuntimeError(f"canonical teacher inference config changed {key}")

    payload = {
        "format": "lineae_teacher_v2",
        "variant": spec.name,
        "model": {key: value.detach().cpu() for key, value in state.items()},
        "source_checkpoint": str(candidate_path.resolve()),
        "source_checkpoint_sha256": candidate_hash,
        "source_config": str(source_config_path.resolve()),
        "source_config_file_sha256": sha256_file(source_config_path),
        "source_config_sha256": config_fingerprint(SLConfig.fromfile(args.config)),
        "baseline_checkpoint_sha256": baseline_checkpoint_hash,
        "baseline_config": expected_baseline_config,
        "baseline_config_file_sha256": sha256_file(baseline_config_path),
        "baseline_config_sha256": config_fingerprint(
            SLConfig.fromfile(str(baseline_config_path))
        ),
        "inference_config": str(inference_config_path.resolve()),
        "inference_config_file_sha256": sha256_file(inference_config_path),
        "inference_config_sha256": inference_config_sha256,
        "qualification": {
            "candidate": candidate_report,
            "baseline": baseline_report,
            "minimum_ap10_gain": args.minimum_ap10_gain,
            "reload_identical": True,
            "canonical_inference_identical": True,
        },
    }
    atomic_torch_save(payload, args.output)
    result = {
        "format": "lineae_teacher_qualification_v2",
        "variant": spec.name,
        "teacher": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "source_sha256": candidate_hash,
        "baseline_source_sha256": baseline_checkpoint_hash,
        "source_config": str(source_config_path.resolve()),
        "source_config_file_sha256": sha256_file(source_config_path),
        "baseline_config": expected_baseline_config,
        "baseline_config_file_sha256": sha256_file(baseline_config_path),
        "wireframe_sap10": candidate_ap10,
        "baseline_wireframe_sap10": baseline_ap10,
        "reload_identical": True,
        "inference_config": str(inference_config_path.resolve()),
        "inference_config_file_sha256": sha256_file(inference_config_path),
        "inference_config_sha256": inference_config_sha256,
        "canonical_inference_identical": True,
    }
    args.output.with_suffix(".qualification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="configs/lineae/lineae_xl.py")
    parser.add_argument(
        "--inference-config",
        default="configs/lineae/lineae_xl.py",
        help="same-variant canonical architecture config embedded in the artifact",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--baseline-config",
        default="configs/linea/linea_hgnetv2_n.py",
        help="matched baseline config used by the recorded comparison",
    )
    parser.add_argument("--minimum-ap10-gain", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("ckpts/lineae_xl_teacher.pth"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(qualify(args), indent=2))


if __name__ == "__main__":
    main()
