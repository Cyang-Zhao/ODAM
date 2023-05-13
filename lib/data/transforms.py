# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F

def build_transforms(cfg, is_train=True):
    if is_train:
        if cfg.train_image_min_size_range[0] == -1:
            min_size = cfg.train_image_min_size
        else:
            assert len(cfg.train_image_min_size_range) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = list(range(
                cfg.train_image_min_size_range,
                cfg.train_image_min_size_range + 1
            ))
        max_size = cfg.train_image_max_size
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.eval_image_min_size
        max_size = cfg.eval_image_max_size
        flip_prob = 0

    to_bgr255 = cfg.to_bgr255
    tobgr255_transform = ToBGR255(
        to_bgr255=to_bgr255
    )

    transform = Compose(
        [
            Resize(min_size, max_size),
            RandomHorizontalFlip(flip_prob),
            ToTensor(),
            tobgr255_transform,
        ]
    )
    return transform

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if isinstance(target, list):
            target = [t.resize(image.size) for t in target]
        elif target is not None:
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        return F.to_tensor(image), target


# class Normalize(object):
#     def __init__(self, mean, std, to_bgr=True):
#         self.mean = mean
#         self.std = std
#         self.to_bgr = to_bgr

#     def __call__(self, image, target=None):
#         if self.to_bgr:
#             image = image[[2, 1, 0]]
#         image = F.normalize(image, mean=self.mean, std=self.std)
#         return image, target

class ToBGR255(object):
    def __init__(self, to_bgr255=True):
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255.
        return image, target