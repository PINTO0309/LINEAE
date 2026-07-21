# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import json
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


__all__ = ['build']


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
