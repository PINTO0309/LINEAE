"""Authoritative coarse P4 screening candidates for distilled LINEAE variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TuningCandidate:
    variant: str
    profile: str
    input_size: int
    num_queries: int
    num_select: int
    decoder_layers: int

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.input_size % 32:
            raise ValueError("tuning input size must be a positive multiple of 32")
        if self.num_queries <= 0:
            raise ValueError("tuning query count must be positive")
        if not 0 < self.num_select <= self.num_queries:
            raise ValueError("tuning num_select must be in [1, num_queries]")
        if self.decoder_layers <= 0:
            raise ValueError("tuning decoder depth must be positive")


_VALUES = {
    ("A", "speed"): (320, 600, 200, 2),
    ("A", "accuracy"): (384, 1400, 400, 4),
    ("F", "speed"): (352, 600, 200, 2),
    ("F", "accuracy"): (480, 1400, 400, 4),
    ("P", "speed"): (512, 700, 200, 2),
    ("P", "accuracy"): (768, 1400, 400, 4),
    ("N", "speed"): (512, 700, 200, 2),
    ("N", "accuracy"): (768, 1400, 400, 4),
    ("S", "speed"): (512, 700, 200, 2),
    ("S", "accuracy"): (768, 1400, 400, 4),
    ("M", "speed"): (512, 700, 200, 3),
    ("M", "accuracy"): (768, 1500, 450, 5),
    ("L", "speed"): (512, 700, 200, 3),
    ("L", "accuracy"): (768, 1500, 450, 5),
    ("X", "speed"): (576, 800, 250, 4),
    ("X", "accuracy"): (800, 1600, 500, 7),
}

TUNING_CANDIDATES = {
    key: TuningCandidate(key[0], key[1], *values)
    for key, values in _VALUES.items()
}


def get_tuning_candidate(variant: str, profile: str) -> TuningCandidate:
    key = (variant.upper(), profile.lower())
    try:
        return TUNING_CANDIDATES[key]
    except KeyError as error:
        raise ValueError(
            f"unknown LINEAE tuning candidate {key!r}; expected one of "
            f"{tuple(TUNING_CANDIDATES)}"
        ) from error


__all__ = ["TUNING_CANDIDATES", "TuningCandidate", "get_tuning_candidate"]
