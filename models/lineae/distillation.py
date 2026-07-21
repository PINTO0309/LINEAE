"""Knowledge distillation for unordered sets of line-segment proposals.

LINEAE queries have no stable identity across independently trained models.  This
module therefore filters teacher proposals and computes a fresh bipartite match
for every image.  Line costs and losses are invariant to exchanging a segment's
two endpoints.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
import torch.nn.functional as F

from .linea_utils import endpoint_invariant_loss, endpoint_swap, pairwise_endpoint_l1


@dataclass(frozen=True)
class DistillationSchedule:
    weight: float
    temperature: float


def resolve_distillation_temperature_steps(
    configured_steps: int,
    *,
    optimizer_steps_per_epoch: int,
    epochs: int,
) -> int:
    """Resolve ``-1`` to the last optimizer-step index of the full run."""
    configured_steps = int(configured_steps)
    if configured_steps < -1:
        raise ValueError("distillation temperature steps must be -1 or non-negative")
    if configured_steps >= 0:
        return configured_steps
    if optimizer_steps_per_epoch <= 0 or epochs <= 0:
        raise ValueError("automatic temperature scheduling requires positive steps and epochs")
    return max(0, int(optimizer_steps_per_epoch) * int(epochs) - 1)


class TeacherTargetCache:
    """Disk cache keyed by every byte of the exact augmented teacher input."""

    def __init__(self, directory: str | Path, *, namespace: object, read_only: bool = False):
        namespace_json = json.dumps(namespace, sort_keys=True, separators=(",", ":"))
        self.namespace = hashlib.sha256(namespace_json.encode()).hexdigest()
        self.directory = Path(directory) / self.namespace
        self.read_only = bool(read_only)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if not self.read_only:
            self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, sample: Tensor, *, context: str = "") -> str:
        value = sample.detach().contiguous().cpu()
        digest = hashlib.sha256()
        digest.update(self.namespace.encode())
        if context:
            digest.update(b"context:")
            digest.update(context.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        # NumPy exposes the contiguous CPU tensor through the buffer protocol;
        # hashlib can consume it directly without duplicating the full image in
        # a temporary ``bytes`` object.
        digest.update(value.view(torch.uint8).numpy())
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.pth"

    @staticmethod
    def _validate(payload) -> dict:
        if not isinstance(payload, dict):
            raise TypeError("cached teacher target must be a dictionary")
        for key in ("pred_logits", "pred_lines"):
            if not isinstance(payload.get(key), Tensor):
                raise TypeError(f"cached teacher target lacks tensor {key!r}")
        features = payload.get("distill_features")
        if features is not None and (
            not isinstance(features, list)
            or len(features) != 3
            or not all(isinstance(feature, Tensor) for feature in features)
        ):
            raise TypeError("cached distill_features must be a list of three tensors")
        return payload

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.hits += 1
        return self._validate(payload)

    def put(self, key: str, payload: dict) -> None:
        if self.read_only:
            return
        payload = self._validate(payload)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.writes += 1

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}


def _pairwise_bernoulli_cross_entropy(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    teacher_prob = teacher_logits.float().sigmoid()
    student = student_logits.float()
    # [student query, teacher query, class]
    return (
        teacher_prob.unsqueeze(0) * F.softplus(-student.unsqueeze(1))
        + (1.0 - teacher_prob).unsqueeze(0) * F.softplus(student.unsqueeze(1))
    ).mean(dim=-1)


def _batched_linear_sum_assignment(costs: list[Tensor]):
    """Solve differently shaped costs after one device-to-CPU synchronization."""
    if not costs:
        return []
    if any(cost.ndim != 2 or min(cost.shape) == 0 for cost in costs):
        raise ValueError("Hungarian cost matrices must be non-empty and rank two")
    device = costs[0].device
    if any(cost.device != device for cost in costs):
        raise ValueError("Hungarian cost matrices must share one device")

    sizes = [cost.numel() for cost in costs]
    flattened = torch.cat([cost.detach().reshape(-1) for cost in costs])
    flattened_cpu = flattened.cpu().numpy()
    assignments = []
    offset = 0
    for cost, size in zip(costs, sizes, strict=True):
        matrix = flattened_cpu[offset:offset + size].reshape(cost.shape)
        assignments.append(linear_sum_assignment(matrix))
        offset += size
    return assignments


class LineSetDistillationCriterion(nn.Module):
    """Output-level KD for LINEAE's unordered line proposal sets."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.3,
        top_k: int = 300,
        match_cost_class: float = 2.0,
        match_cost_line: float = 5.0,
        class_weight: float = 1.0,
        line_weight: float = 5.0,
        feature_weight: float = 0.0,
        feature_loss: str = "cosine",
        total_weight: float = 1.0,
        warmup_steps: int = 0,
        temperature_start: float = 1.0,
        temperature_end: float = 4.0,
        temperature_steps: int = 0,
    ) -> None:
        super().__init__()
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if match_cost_class < 0 or match_cost_line < 0:
            raise ValueError("matching costs must be non-negative")
        if match_cost_class == 0 and match_cost_line == 0:
            raise ValueError("at least one matching cost must be positive")
        if min(class_weight, line_weight, feature_weight, total_weight) < 0:
            raise ValueError("distillation weights must be non-negative")
        if feature_loss not in {"cosine", "mse"}:
            raise ValueError("feature_loss must be 'cosine' or 'mse'")
        if min(temperature_start, temperature_end) <= 0:
            raise ValueError("distillation temperatures must be positive")

        self.confidence_threshold = float(confidence_threshold)
        self.top_k = int(top_k)
        self.match_cost_class = float(match_cost_class)
        self.match_cost_line = float(match_cost_line)
        self.class_weight = float(class_weight)
        self.line_weight = float(line_weight)
        self.feature_weight = float(feature_weight)
        self.feature_loss = feature_loss
        self.total_weight = float(total_weight)
        self.warmup_steps = max(0, int(warmup_steps))
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        temperature_steps = int(temperature_steps)
        if temperature_steps < -1:
            raise ValueError("temperature_steps must be -1 or non-negative")
        self.temperature_steps = None if temperature_steps == -1 else temperature_steps

    def set_temperature_steps(self, steps: int) -> None:
        steps = int(steps)
        if steps < 0:
            raise ValueError("resolved temperature steps must be non-negative")
        self.temperature_steps = steps

    def schedule(self, global_step: int) -> DistillationSchedule:
        step = max(0, int(global_step))
        if self.warmup_steps:
            ramp = min(1.0, step / self.warmup_steps)
        else:
            ramp = 1.0
        if self.temperature_steps is None:
            raise RuntimeError("automatic distillation temperature steps were not resolved")
        if self.temperature_steps:
            progress = min(1.0, step / self.temperature_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            temperature = self.temperature_end + (
                self.temperature_start - self.temperature_end
            ) * cosine
        else:
            temperature = self.temperature_end
        return DistillationSchedule(self.total_weight * ramp, temperature)

    @torch.no_grad()
    def _teacher_indices(self, logits: Tensor) -> Tensor:
        # LINEA's Wireframe/York annotations contain only category 0, and both
        # LineEvaluator and PostProcess define line confidence as channel 0.
        # Channel 1 is always supervised as a negative target; allowing it to
        # select KD proposals would admit detections that can never score at
        # inference time.
        confidence = logits[..., 0].float().sigmoid()
        selected = torch.nonzero(
            confidence >= self.confidence_threshold,
            as_tuple=False,
        ).flatten()
        if selected.numel() == 0:
            return selected
        selected = selected[torch.argsort(confidence[selected], descending=True, stable=True)]
        if self.top_k:
            selected = selected[: self.top_k]
        return selected

    @torch.no_grad()
    def match(self, student_outputs: dict[str, Tensor], teacher_outputs: dict[str, Tensor]):
        self._validate_outputs(student_outputs, teacher_outputs)
        batch_size = student_outputs["pred_logits"].shape[0]
        matches = [None] * batch_size
        pending_costs = []
        pending = []
        for batch_index, (student_logits, student_lines, teacher_logits, teacher_lines) in enumerate(zip(
            student_outputs["pred_logits"],
            student_outputs["pred_lines"],
            teacher_outputs["pred_logits"],
            teacher_outputs["pred_lines"],
            strict=True,
        )):
            teacher_indices = self._teacher_indices(teacher_logits)
            if teacher_indices.numel() == 0 or student_logits.shape[0] == 0:
                empty = torch.empty(0, dtype=torch.long, device=student_logits.device)
                matches[batch_index] = (empty, empty)
                continue
            filtered_logits = teacher_logits[teacher_indices]
            filtered_lines = teacher_lines[teacher_indices]
            class_cost = _pairwise_bernoulli_cross_entropy(
                student_logits[..., :1],
                filtered_logits[..., :1],
            )
            line_cost = pairwise_endpoint_l1(student_lines, filtered_lines)
            cost = self.match_cost_class * class_cost + self.match_cost_line * line_cost
            pending_costs.append(cost)
            pending.append((batch_index, student_logits.device, teacher_indices))

        assignments = _batched_linear_sum_assignment(pending_costs)
        for (batch_index, device, teacher_indices), (student_index, filtered_index) in zip(
            pending,
            assignments,
            strict=True,
        ):
            student_index = torch.as_tensor(student_index, dtype=torch.long, device=device)
            filtered_index = torch.as_tensor(filtered_index, dtype=torch.long, device=device)
            matches[batch_index] = (student_index, teacher_indices[filtered_index])
        if any(match is None for match in matches):
            raise RuntimeError("distillation matching did not resolve every batch item")
        return matches

    @staticmethod
    def _validate_outputs(student_outputs, teacher_outputs) -> None:
        for name, outputs in (("student", student_outputs), ("teacher", teacher_outputs)):
            missing = {"pred_logits", "pred_lines"} - outputs.keys()
            if missing:
                raise KeyError(f"{name} outputs missing keys: {sorted(missing)}")
            if outputs["pred_logits"].ndim != 3 or outputs["pred_lines"].ndim != 3:
                raise ValueError(f"{name} predictions must be rank-three tensors")
            if outputs["pred_logits"].shape[-1] < 1:
                raise ValueError(f"{name} pred_logits must contain the line-confidence channel")
            if outputs["pred_lines"].shape[-1] != 4:
                raise ValueError(f"{name} pred_lines must have four coordinates")
        if student_outputs["pred_logits"].shape[0] != teacher_outputs["pred_logits"].shape[0]:
            raise ValueError("student and teacher batch sizes differ")
        if student_outputs["pred_logits"].shape[-1] != teacher_outputs["pred_logits"].shape[-1]:
            raise ValueError("student and teacher class counts differ")

    @staticmethod
    def _bernoulli_kl(student_logits: Tensor, teacher_logits: Tensor, temperature: float) -> Tensor:
        student = student_logits.float() / temperature
        teacher = teacher_logits.detach().float() / temperature
        probability = teacher.sigmoid()
        cross_entropy = probability * F.softplus(-student) + (1.0 - probability) * F.softplus(student)
        teacher_entropy = probability * F.softplus(-teacher) + (1.0 - probability) * F.softplus(teacher)
        return (cross_entropy - teacher_entropy).clamp_min(0).mean() * (temperature ** 2)

    @staticmethod
    def _line_loss(student_lines: Tensor, teacher_lines: Tensor) -> Tensor:
        teacher_lines = teacher_lines.detach()
        direct = F.smooth_l1_loss(student_lines, teacher_lines, reduction="none").sum(dim=-1)
        swapped = F.smooth_l1_loss(
            student_lines,
            endpoint_swap(teacher_lines),
            reduction="none",
        ).sum(dim=-1)
        return endpoint_invariant_loss(direct, swapped).mean()

    def _feature_loss(self, student_outputs, teacher_outputs) -> Tensor | None:
        if self.feature_weight == 0:
            return None
        for name, outputs in (("student", student_outputs), ("teacher", teacher_outputs)):
            features = outputs.get("distill_features")
            if not isinstance(features, (list, tuple)) or len(features) != 3:
                raise ValueError(f"{name} must provide three distill_features for feature KD")
        losses = []
        for level, (student, teacher) in enumerate(zip(
            student_outputs["distill_features"],
            teacher_outputs["distill_features"],
            strict=True,
        )):
            if student.ndim != 4 or teacher.ndim != 4:
                raise ValueError(
                    f"feature KD level {level} must use BCHW tensors: "
                    f"student={tuple(student.shape)}, teacher={tuple(teacher.shape)}"
                )
            if student.shape[:2] != teacher.shape[:2]:
                raise ValueError(
                    f"feature KD level {level} batch/channel mismatch: "
                    f"student={tuple(student.shape)}, teacher={tuple(teacher.shape)}"
                )
            if student.shape[-2:] != teacher.shape[-2:]:
                teacher = F.interpolate(
                    teacher.detach().float(),
                    size=student.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            student = F.normalize(student.float(), dim=1)
            teacher = F.normalize(teacher.detach().float(), dim=1)
            if self.feature_loss == "cosine":
                losses.append((1.0 - (student * teacher).sum(dim=1)).mean())
            else:
                losses.append(F.mse_loss(student, teacher))
        return torch.stack(losses).mean()

    def forward(
        self,
        student_outputs: dict[str, Tensor],
        teacher_outputs: dict[str, Tensor],
        *,
        global_step: int,
    ) -> dict[str, Tensor]:
        matches = self.match(student_outputs, teacher_outputs)
        self.last_match_count = sum(student_index.numel() for student_index, _ in matches)
        student_logits = []
        teacher_logits = []
        student_lines = []
        teacher_lines = []
        for batch_index, (student_index, teacher_index) in enumerate(matches):
            if student_index.numel() == 0:
                continue
            student_logits.append(
                student_outputs["pred_logits"][batch_index, student_index, :1]
            )
            teacher_logits.append(
                teacher_outputs["pred_logits"][batch_index, teacher_index, :1]
            )
            student_lines.append(student_outputs["pred_lines"][batch_index, student_index])
            teacher_lines.append(teacher_outputs["pred_lines"][batch_index, teacher_index])

        schedule = self.schedule(global_step)
        zero = student_outputs["pred_logits"].sum() * 0.0
        feature_loss = self._feature_loss(student_outputs, teacher_outputs)
        result = {}
        if feature_loss is not None:
            result["loss_kd_feature"] = (
                feature_loss * self.feature_weight * schedule.weight
            )
        if not student_logits:
            result.update({
                "loss_kd_logits": zero,
                "loss_kd_line": zero,
            })
            return result

        kd_logits = self._bernoulli_kl(
            torch.cat(student_logits),
            torch.cat(teacher_logits),
            schedule.temperature,
        )
        kd_line = self._line_loss(torch.cat(student_lines), torch.cat(teacher_lines))
        result.update({
            "loss_kd_logits": kd_logits * self.class_weight * schedule.weight,
            "loss_kd_line": kd_line * self.line_weight * schedule.weight,
        })
        return result


class DistillationTeacher(nn.Module):
    """Frozen teacher with explicit normalization and canonical-size conversion."""

    def __init__(
        self,
        model: nn.Module,
        source_mean,
        source_std,
        target_mean,
        target_std,
        target_spatial_size=None,
        cache: TeacherTargetCache | None = None,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("source_mean", torch.tensor(source_mean).view(1, 3, 1, 1))
        self.register_buffer("source_std", torch.tensor(source_std).view(1, 3, 1, 1))
        self.register_buffer("target_mean", torch.tensor(target_mean).view(1, 3, 1, 1))
        self.register_buffer("target_std", torch.tensor(target_std).view(1, 3, 1, 1))
        if target_spatial_size is None:
            self.target_spatial_size = None
        else:
            if isinstance(target_spatial_size, int):
                target_spatial_size = (target_spatial_size, target_spatial_size)
            if (
                not isinstance(target_spatial_size, (list, tuple))
                or len(target_spatial_size) != 2
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in target_spatial_size
                )
            ):
                raise ValueError("teacher target_spatial_size must contain two positive integers")
            self.target_spatial_size = tuple(target_spatial_size)
        self.cache = cache
        cache_transform = {
            "schema": "lineae_teacher_preprocess_v3",
            "source_mean": self.source_mean.flatten().tolist(),
            "source_std": self.source_std.flatten().tolist(),
            "target_mean": self.target_mean.flatten().tolist(),
            "target_std": self.target_std.flatten().tolist(),
            "target_spatial_size": self.target_spatial_size,
            "resize_mode": "opencv_inter_linear_equivalent",
        }
        cache_transform_json = json.dumps(
            cache_transform,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._cache_key_context = hashlib.sha256(cache_transform_json.encode()).hexdigest()

    @staticmethod
    def _sample_payload(outputs: dict, index: int) -> dict:
        payload = {
            "pred_logits": outputs["pred_logits"][index].detach().cpu(),
            "pred_lines": outputs["pred_lines"][index].detach().cpu(),
        }
        if "distill_features" in outputs:
            payload["distill_features"] = [
                feature[index].detach().cpu() for feature in outputs["distill_features"]
            ]
        return payload

    @staticmethod
    def _batch_payloads(payloads: list[dict], device: torch.device) -> dict:
        result = {
            "pred_logits": torch.stack([item["pred_logits"] for item in payloads]).to(device),
            "pred_lines": torch.stack([item["pred_lines"] for item in payloads]).to(device),
        }
        if "distill_features" in payloads[0]:
            if not all("distill_features" in item for item in payloads):
                raise ValueError("cached teacher feature schema is inconsistent")
            result["distill_features"] = [
                torch.stack([item["distill_features"][level] for item in payloads]).to(device)
                for level in range(3)
            ]
        return result

    def _prepare_samples(self, samples: Tensor) -> Tensor:
        source_mean = self.source_mean.to(dtype=samples.dtype)
        source_std = self.source_std.to(dtype=samples.dtype)
        target_mean = self.target_mean.to(dtype=samples.dtype)
        target_std = self.target_std.to(dtype=samples.dtype)
        pixels = samples * source_std + source_mean
        teacher_samples = (pixels - target_mean) / target_std
        if (
            self.target_spatial_size is not None
            and teacher_samples.shape[-2:] != self.target_spatial_size
        ):
            # Gazelle runs a teacher at its canonical input even when a smaller
            # student consumes the same augmented image. LINEAE predictions use
            # normalized endpoints, so no target-coordinate conversion is needed.
            teacher_samples = F.interpolate(
                teacher_samples,
                size=self.target_spatial_size,
                mode="bilinear",
                align_corners=False,
            )
        return teacher_samples

    def forward(self, samples: Tensor, targets=None):
        if self.cache is None:
            return self.model(self._prepare_samples(samples), None)

        # The exact augmented source tensor plus the preprocessing fingerprint
        # uniquely determines the teacher tensor. Key before canonical resizing
        # so smaller students hash fewer bytes and all-hit batches skip
        # normalization/interpolation entirely.
        keys = [
            self.cache.key(sample, context=self._cache_key_context)
            for sample in samples
        ]
        payloads = [self.cache.get(key) for key in keys]
        missing = [index for index, payload in enumerate(payloads) if payload is None]
        if missing:
            indices = torch.tensor(missing, device=samples.device, dtype=torch.long)
            teacher_samples = self._prepare_samples(samples.index_select(0, indices))
            outputs = self.model(teacher_samples, None)
            for local_index, batch_index in enumerate(missing):
                payload = self._sample_payload(outputs, local_index)
                payloads[batch_index] = payload
                self.cache.put(keys[batch_index], payload)
        if not all(isinstance(payload, dict) for payload in payloads):
            raise RuntimeError("teacher cache failed to resolve every batch item")
        return self._batch_payloads(payloads, samples.device)

    @property
    def cache_stats(self) -> dict[str, int] | None:
        return None if self.cache is None else self.cache.stats


def build_distillation_criterion(args) -> LineSetDistillationCriterion:
    return LineSetDistillationCriterion(
        confidence_threshold=args.distill_confidence_threshold,
        top_k=args.distill_top_k,
        match_cost_class=args.distill_match_cost_class,
        match_cost_line=args.distill_match_cost_line,
        class_weight=args.distill_class_weight,
        line_weight=args.distill_line_weight,
        feature_weight=getattr(args, "distill_feature_weight", 0.0),
        feature_loss=getattr(args, "distill_feature_loss", "cosine"),
        total_weight=args.distill_weight,
        warmup_steps=args.distill_warmup_steps,
        temperature_start=args.distill_temperature_start,
        temperature_end=args.distill_temperature_end,
        temperature_steps=args.distill_temperature_steps,
    )


__all__ = [
    "DistillationTeacher",
    "DistillationSchedule",
    "LineSetDistillationCriterion",
    "TeacherTargetCache",
    "build_distillation_criterion",
    "endpoint_swap",
    "pairwise_endpoint_l1",
    "resolve_distillation_temperature_steps",
]
