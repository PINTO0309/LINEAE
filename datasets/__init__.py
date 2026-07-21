from .coco import build as build_coco
from .line_eval import DualLineEvaluator, LineEvaluator, SAP_EVALUATION_PROTOCOL
from .collate import BatchImageCollateFunction

__all__ = [
    "BatchImageCollateFunction",
    "DualLineEvaluator",
    "LineEvaluator",
    "SAP_EVALUATION_PROTOCOL",
    "build_dataset",
]


def build_dataset(image_set, args):
    return build_coco(image_set, args)
