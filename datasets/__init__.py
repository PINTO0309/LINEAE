from .coco import build as build_coco
from .line_eval import LineEvaluator
from .collate import BatchImageCollateFunction

__all__ = ["BatchImageCollateFunction", "LineEvaluator", "build_dataset"]


def build_dataset(image_set, args):
    return build_coco(image_set, args)
