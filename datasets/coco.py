# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

import torch
import torch.utils.data
from pycocotools.coco import COCO

import datasets.transforms as T
from util.image_preprocess import (
    IMAGE_PREPROCESS_SCHEMA,
    read_rgb_image,
    validate_image_preprocess_schema,
)


__all__ = ['build', 'resolve_ensemble_training_sources']


ENSEMBLE_DATASET_SCHEMA = 'wireframe_york_auto_discovery_v3'
ENSEMBLE_LINE_FORMAT = 'xy_dxdy_v1'


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dxdy_annotations(
    annotation: dict,
    images_by_id: dict[int, dict],
    annotation_file: Path,
    dataset_label: str,
) -> None:
    line_format = annotation.get('line_format')
    if line_format is not None and line_format != ENSEMBLE_LINE_FORMAT:
        raise ValueError(
            f'ensemble {dataset_label} annotation has unsupported line_format '
            f'{line_format!r}; expected {ENSEMBLE_LINE_FORMAT!r}: '
            f'{annotation_file}'
        )
    annotations = annotation.get('annotations')
    if not isinstance(annotations, list):
        raise ValueError(
            f'ensemble {dataset_label} annotation must contain an annotations list: '
            f'{annotation_file}'
        )
    categories = annotation.get('categories')
    if not isinstance(categories, list) or not any(
        isinstance(category, dict) and str(category.get('id')) == '0'
        for category in categories
    ):
        raise ValueError(
            f'ensemble {dataset_label} annotation must contain category id 0: '
            f'{annotation_file}'
        )
    tolerance = 1e-6
    annotation_ids = set()
    for index, record in enumerate(annotations):
        if not isinstance(record, dict):
            raise ValueError(
                f'ensemble {dataset_label} annotation has an invalid record at '
                f'index {index}: {annotation_file}'
            )
        annotation_id = record.get('id')
        if (
            isinstance(annotation_id, bool)
            or not isinstance(annotation_id, int)
            or annotation_id in annotation_ids
        ):
            raise ValueError(
                f'ensemble {dataset_label} annotation has an invalid or duplicate '
                f'annotation id at index {index}: {annotation_file}'
            )
        annotation_ids.add(annotation_id)
        image_id = record.get('image_id')
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id not in images_by_id
        ):
            raise ValueError(
                f'ensemble {dataset_label} annotation record {index} references unknown '
                f'image id {image_id!r}: {annotation_file}'
            )
        category_id = record.get('category_id')
        if isinstance(category_id, bool) or category_id != 0:
            raise ValueError(
                f'ensemble {dataset_label} annotation record {index} must use '
                f'category_id=0: {annotation_file}'
            )
        area = record.get('area')
        if (
            isinstance(area, bool)
            or not isinstance(area, (int, float))
            or not math.isfinite(area)
        ):
            raise ValueError(
                f'ensemble {dataset_label} annotation record {index} must contain '
                f'a finite area: {annotation_file}'
            )
        line = record.get('line')
        if (
            not isinstance(line, list)
            or len(line) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in line
            )
        ):
            raise ValueError(
                f'ensemble {dataset_label} annotation record {index} must contain four '
                f'finite line values: {annotation_file}'
            )
        x, y, dx, dy = line
        image = images_by_id[image_id]
        if not (
            -tolerance <= x <= image['width'] + tolerance
            and -tolerance <= x + dx <= image['width'] + tolerance
            and -tolerance <= y <= image['height'] + tolerance
            and -tolerance <= y + dy <= image['height'] + tolerance
        ):
            raise ValueError(
                f'ensemble {dataset_label} annotation record {index} has an endpoint '
                f'outside image {image_id}: {annotation_file}'
            )


def _inspect_ensemble_source(
    root: Path,
    split: str,
    *,
    source_name: str,
    dataset_label: str,
    validate_line_format: bool = False,
) -> dict:
    image_dir = root / f'{split}2017'
    annotation_file = root / 'annotations' / f'lines_{split}2017.json'
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f'ensemble {dataset_label} image directory does not exist: {image_dir}'
        )
    if not annotation_file.is_file():
        raise FileNotFoundError(
            f'ensemble {dataset_label} annotation does not exist: {annotation_file}'
        )
    try:
        with annotation_file.open('r', encoding='utf-8') as stream:
            annotation = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f'could not read ensemble {dataset_label} annotation '
            f'{annotation_file}: {error}'
        ) from error
    images = annotation.get('images') if isinstance(annotation, dict) else None
    if not isinstance(images, list):
        raise ValueError(
            f'ensemble {dataset_label} annotation must contain an images list: '
            f'{annotation_file}'
        )
    missing_images = []
    images_by_id = {}
    for index, record in enumerate(images):
        if not isinstance(record, dict) or not isinstance(record.get('file_name'), str):
            raise ValueError(
                f'ensemble {dataset_label} annotation has an invalid image record: '
                f'{annotation_file}'
            )
        if validate_line_format:
            image_id = record.get('id')
            width = record.get('width')
            height = record.get('height')
            if (
                isinstance(image_id, bool)
                or not isinstance(image_id, int)
                or image_id in images_by_id
                or isinstance(width, bool)
                or not isinstance(width, int)
                or width <= 0
                or isinstance(height, bool)
                or not isinstance(height, int)
                or height <= 0
            ):
                raise ValueError(
                    f'ensemble {dataset_label} annotation has invalid image '
                    f'metadata at index {index}: {annotation_file}'
                )
            images_by_id[image_id] = record
        image_path = image_dir / record['file_name']
        if not image_path.is_file():
            missing_images.append(str(image_path))
    if missing_images:
        raise FileNotFoundError(
            f'ensemble {dataset_label} annotation references missing images: '
            f'{missing_images[:5]}'
        )
    if validate_line_format:
        _validate_dxdy_annotations(
            annotation,
            images_by_id,
            annotation_file,
            dataset_label,
        )
    return {
        'name': source_name,
        'dataset_name': dataset_label,
        'split': split,
        'root': str(root),
        'image_dir': str(image_dir),
        'annotation_file': str(annotation_file),
        'annotation_sha256': _sha256_file(annotation_file),
        'samples': len(images),
    }


def _normalize_ensemble_dataset_name(root: Path) -> str:
    name = root.name
    if name.lower().endswith('_processed'):
        name = name[:-len('_processed')]
    normalized = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    if not normalized:
        raise ValueError(
            f'could not derive an ensemble dataset name from directory: {root}'
        )
    return normalized


def _discover_ensemble_datasets(
    discovery_root: Path,
    *,
    excluded_roots: set[Path],
) -> list[dict]:
    """Discover and preflight generic sibling datasets deterministically."""
    if not discovery_root.is_dir():
        raise FileNotFoundError(
            f'ensemble discovery root does not exist: {discovery_root}'
        )
    discovered = []
    names = set()
    for root in sorted(
        (path.resolve() for path in discovery_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        if root in excluded_roots:
            continue
        train_images = root / 'train2017'
        train_annotation = root / 'annotations' / 'lines_train2017.json'
        has_train_images = train_images.is_dir()
        has_train_annotation = train_annotation.is_file()
        if not has_train_images and not has_train_annotation:
            continue
        if has_train_images != has_train_annotation:
            missing = train_annotation if has_train_images else train_images
            raise FileNotFoundError(
                f'incomplete auto-discovered ensemble dataset {root}: missing {missing}'
            )

        dataset_name = _normalize_ensemble_dataset_name(root)
        if dataset_name in {'wireframe', 'york'}:
            raise ValueError(
                f'auto-discovered ensemble dataset uses reserved name '
                f'{dataset_name!r}: {root}'
            )
        if dataset_name in names:
            raise ValueError(
                f'duplicate auto-discovered ensemble dataset name '
                f'{dataset_name!r}: {root}'
            )
        names.add(dataset_name)

        train_source = _inspect_ensemble_source(
            root,
            'train',
            source_name=f'{dataset_name}_train',
            dataset_label=dataset_name,
            validate_line_format=True,
        )
        if train_source['samples'] == 0:
            raise ValueError(
                f'auto-discovered ensemble dataset has no train images: {root}'
            )

        val_images = root / 'val2017'
        val_annotation = root / 'annotations' / 'lines_val2017.json'
        has_val_images = val_images.is_dir()
        has_val_annotation = val_annotation.is_file()
        if has_val_images != has_val_annotation:
            missing = val_annotation if has_val_images else val_images
            raise FileNotFoundError(
                f'incomplete validation split for auto-discovered ensemble '
                f'dataset {dataset_name!r}: missing {missing}'
            )
        val_source = None
        if has_val_images:
            val_source = _inspect_ensemble_source(
                root,
                'val',
                source_name=f'{dataset_name}_val',
                dataset_label=dataset_name,
                validate_line_format=True,
            )
            if val_source['samples'] == 0:
                raise ValueError(
                    f'auto-discovered ensemble dataset has an empty validation '
                    f'split: {root}'
                )
        discovered.append(
            {
                'name': dataset_name,
                'root': str(root),
                'train': train_source,
                'val': val_source,
            }
        )
    return sorted(discovered, key=lambda dataset: dataset['name'])


def resolve_ensemble_training_sources(args) -> list[dict]:
    """Preflight and record every auxiliary source added to training."""
    if not getattr(args, 'ensemble', False):
        args.ensemble_dataset_schema = None
        args.ensemble_discovery_root = None
        args.ensemble_discovered_datasets = []
        args.ensemble_validation_sources = []
        args.ensemble_annotation_sha256 = None
        args.ensemble_split_samples = None
        args.ensemble_training_sample_count = 0
        args.ensemble_training_sources = []
        return []
    if getattr(args, 'use_lmap', False):
        raise ValueError('--ensemble cannot be combined with use_lmap=True')
    args.ensemble_dataset_schema = ENSEMBLE_DATASET_SCHEMA
    york_root = Path(
        getattr(args, 'ensemble_york_path', 'data/york_processed')
    ).resolve()
    args.ensemble_york_path = str(york_root)
    york_sources = [
        _inspect_ensemble_source(
            york_root,
            'train',
            source_name='york_train',
            dataset_label='YorkUrban',
        ),
        _inspect_ensemble_source(
            york_root,
            'val',
            source_name='york_val',
            dataset_label='YorkUrban',
        ),
    ]
    york_total = sum(source['samples'] for source in york_sources)
    if york_total == 0:
        raise ValueError(
            f'ensemble YorkUrban dataset contains no images in train or val: '
            f'{york_root}'
        )
    args.ensemble_annotation_sha256 = {
        source['split']: source['annotation_sha256'] for source in york_sources
    }
    args.ensemble_split_samples = {
        source['split']: source['samples'] for source in york_sources
    }

    discovery_root = Path(args.coco_path).resolve().parent
    args.ensemble_discovery_root = str(discovery_root)
    discovered = _discover_ensemble_datasets(
        discovery_root,
        excluded_roots={Path(args.coco_path).resolve(), york_root},
    )
    args.ensemble_discovered_datasets = discovered
    args.ensemble_validation_sources = [
        dataset['val'] for dataset in discovered if dataset['val'] is not None
    ]
    sources = [
        *york_sources,
        *(dataset['train'] for dataset in discovered),
    ]

    args.ensemble_training_sample_count = sum(
        source['samples'] for source in sources
    )
    args.ensemble_training_sources = sources
    return sources


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms, include_lmap):
        self.root = Path(img_folder)
        self.coco = COCO(str(ann_file))
        self.ids = sorted(self.coco.imgs)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()

        with open(ann_file, 'r') as file:
            data = json.load(file)
            id2imgfile = {d['id']: d['file_name'].split('.')[0] for d in data['images']}
        self.id2imgfile = id2imgfile
        self.lmap_folder_dir = str(img_folder).replace('processed', 'extras')

        self.include_lmap = include_lmap

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_record = self.coco.loadImgs([image_id])[0]
        image_path = self.root / image_record["file_name"]
        img = read_rgb_image(image_path)
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(annotation_ids)
        target = annotations
        target = {'image_id': image_id, 'annotations': target}

        if self.include_lmap:
            name = self.id2imgfile[image_id]
            lmaps = []
            for downsampling in [8, 4, 32]:
                npz = np.load(f'{self.lmap_folder_dir}/{name}_downsample{downsampling}_label.npz')
                lmap = npz['lmap']
                lmaps.append(lmap)
            target.update({'lmap': lmaps})

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCocoPolysToMask(object):

    def __call__(self, image, target):
        h, w = image.shape[:2]

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno]
 
        lines = [obj["line"] for obj in anno]
        lines = torch.as_tensor(lines, dtype=torch.float32).reshape(-1, 4)

        lines[:, 2:] += lines[:, :2] #xyxy

        lines[:, 0::2].clamp_(min=0, max=w)
        lines[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if 'lmap' in target:
            lmaps = [torch.as_tensor(lmap).unsqueeze(0) for lmap in target['lmap']]

            target = {}
            target["lines"] = lines

            target["labels"] = classes
            
            target["image_id"] = image_id
            target['lmap'] = lmaps

        else:
            target = {}

            target["lines"] = lines

            target["labels"] = classes
            
            target["image_id"] = image_id
        


        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno ])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args=None):

    validate_image_preprocess_schema(
        getattr(args, "image_preprocess_schema", IMAGE_PREPROCESS_SCHEMA)
    )

    image_mean = getattr(args, 'image_mean', [0.538, 0.494, 0.453])
    image_std = getattr(args, 'image_std', [0.257, 0.263, 0.273])
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(image_mean, image_std)
    ])

    # update args from config files
    scales = args.data_aug_scales
    max_size = args.data_aug_max_size
    scales2_resize = args.data_aug_scales2_resize
    scales2_crop = args.data_aug_scales2_crop
    if len(scales2_crop) != 2:
        raise ValueError("data_aug_scales2_crop must contain [min_size, max_size]")
    test_size = args.eval_spatial_size
    photometric = T.ColorJitter()
    if getattr(args, 'use_photometric_distort', False):
        photometric = T.RandomPhotometricDistort(
            p=getattr(args, 'photometric_distort_probability', 0.5)
        )

    if image_set == 'train':
        return T.Compose([
            T.RandomSelect(
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                ),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            photometric,
            normalize,
        ])

    if image_set in ['val', 'test']:
        return T.Compose([
            T.RandomResize([test_size], max_size=max_size),
            normalize,
        ])



    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    mode = 'lines'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
    }

    # add some hooks to datasets
    img_folder, ann_file = PATHS[image_set]

    if 'train' not in image_set:
        use_lmap = False
    else:
        use_lmap = args.use_lmap

    bs = getattr(args, f'batch_size_{image_set}') 
    print(f'building {image_set}_dataloader with batch_size={bs}...')
    dataset = CocoDetection(img_folder, ann_file, 
            transforms=make_coco_transforms(image_set, args=args),
            include_lmap=use_lmap
        )

    if image_set == 'train' and getattr(args, 'ensemble', False):
        sources = getattr(args, 'ensemble_training_sources', None)
        if not sources:
            sources = resolve_ensemble_training_sources(args)
        datasets = [dataset]
        for source in sources:
            if source['samples'] == 0:
                continue
            datasets.append(
                CocoDetection(
                    source['image_dir'],
                    source['annotation_file'],
                    transforms=make_coco_transforms('train', args=args),
                    include_lmap=False,
                )
            )
        dataset = torch.utils.data.ConcatDataset(datasets)
        args.training_dataset_sources = [
            {
                'name': 'primary_train',
                'split': 'train',
                'root': str(root),
                'image_dir': str(img_folder),
                'annotation_file': str(ann_file),
                'samples': len(datasets[0]),
            },
            *sources,
        ]
        args.training_dataset_sample_count = len(dataset)
        source_counts = ', '.join(
            f'{source["name"]}={source["samples"]}' for source in sources
        )
        print(
            'ensemble training dataset: '
            f'primary_train={len(datasets[0])}, '
            f'{source_counts}, '
            f'total={len(dataset)}'
        )

    return dataset



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    dataset_debug = CocoDetection(
            '../data/wireframe_processed/val2017',
            '../data/wireframe_processed/annotations/lines_val2017.json',
            transforms=T.Compose([
                # T.RandomResize([400, 500, 600]),
                # T.RandomSizeCrop(384, 600),
                T.RandomResize([(640, 640)], max_size=1333),
                T.ToTensor()
                ]),
            include_lmap=False
        )
    i = 0
    for sample, target in dataset_debug:
        if 'lmap' in target:
            print(target['lmap'].shape, sample.shape)
            h, w = sample.shape[-2:]
            plt.imshow(sample.permute(1, 2, 0), extent=[-1, 1, -1, 1])
            plt.imshow(np.array(target['lmap'][0, 0]), alpha=0.4, extent=[-1, 1, -1, 1])
            for line in target['lines']:
                x1, y1, x2, y2 = line
                x1 = x1 / w * 2 - 1
                x2 = x2 / w * 2 - 1
                y1 = -(y1 / h * 2 - 1)
                y2 = -(y2 / h * 2 - 1)
                plt.plot((x1, x2), (y1, y2), c='r')
            plt.show()
            i+=1
        if 'lneg' in target:
            h, w = sample.shape[-2:]
            plt.imshow(sample.permute(1, 2, 0))#, extent=[-1, 1, -1, 1])
            for line in target['lneg'][:500]:
                x1, y1, x2, y2 = line 
                # x1 = x1 / w * 2 - 1
                # x2 = x2 / w * 2 - 1
                # y1 = -(y1 / h * 2 - 1)
                # y2 = -(y2 / h * 2 - 1)
                plt.plot((x1, x2), (y1, y2), c='r')
            plt.show()


    print(i)
