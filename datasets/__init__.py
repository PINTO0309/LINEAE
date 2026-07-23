from .coco import build as build_coco, resolve_ensemble_training_sources
from .line_eval import DualLineEvaluator, LineEvaluator, SAP_EVALUATION_PROTOCOL
from .collate import BatchImageCollateFunction

__all__ = [
    "BatchImageCollateFunction",
    "DualLineEvaluator",
    "LineEvaluator",
    "SAP_EVALUATION_PROTOCOL",
    "build_dataset",
    "resolve_ensemble_training_sources",
]


def build_dataset(image_set, args):
    return build_coco(image_set, args)
