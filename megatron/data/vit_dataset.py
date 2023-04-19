# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import datasets
from megatron import get_args
from megatron.data.image_folder import ImageFolder
from megatron.data.autoaugment import ImageNetPolicy
from megatron.data.data_samplers import RandomSeedDataset
from PIL import Image, ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class ClassificationTransform():
    def __init__(self, image_size, train=True):
        args = get_args()
        assert args.fp16 or args.bf16
        self.data_type = torch.half if args.fp16 else torch.bfloat16
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                ImageNetPolicy(),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.ConvertImageDtype(self.data_type)
            ])
        else:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.ConvertImageDtype(self.data_type)
            ])

    def __call__(self, input):
        output = self.transform(input)
        return output


class DinoTransform(object):
    def __init__(self, image_size, train=True):
        args = get_args()
        self.data_type = torch.half if args.fp16 else torch.bfloat16

        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4,
			       saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        if args.fp16 or args.bf16:
            normalize = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.ConvertImageDtype(self.data_type)
            ])
        else:
            normalize = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        # first global crop
        scale_const = 0.4
        self.global_transform1 = T.Compose([
            T.RandomResizedCrop(image_size,
                                scale=(scale_const, 1),
                                interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize
        ])
        # second global crop
        self.global_transform2 = T.Compose([
            T.RandomResizedCrop(image_size,
                                scale=(scale_const, 1),
                                interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize
        ])
        # transformation for the local small crops
        self.local_crops_number = args.dino_local_crops_number
        self.local_transform = T.Compose([
            T.RandomResizedCrop(args.dino_local_img_size,
                                scale=(0.05, scale_const),
                                interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


def build_train_valid_datasets(data_path, image_size=224):
    args = get_args()

    if args.vision_pretraining_type == 'classify':
        train_transform = ClassificationTransform(image_size)
        val_transform = ClassificationTransform(image_size, train=False)
    elif args.vision_pretraining_type == 'dino':
        train_transform = DinoTransform(image_size, train=True)
        val_transform = ClassificationTransform(image_size, train=False)
    else:
        raise Exception('{} vit pretraining type is not supported.'.format(
                args.vit_pretraining_type))

    # training dataset
    train_data_path = data_path[0] if len(data_path) <= 2 else data_path[2]
    train_data = ImageFolder(
        root=train_data_path,
        transform=train_transform,
        classes_fraction=args.classes_fraction,
        data_per_class_fraction=args.data_per_class_fraction
    )
    train_data = RandomSeedDataset(train_data)

    # validation dataset
    val_data_path = data_path[1]
    val_data = ImageFolder(
        root=val_data_path,
        transform=val_transform
    )
    val_data = RandomSeedDataset(val_data)

    return train_data, val_data
