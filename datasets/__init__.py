import numbers
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

sys.path.append("External")

from SST.utils import AudioDataset


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    dataset, test_dataset = None, None
    if config.data.dataset == "AUDIO":
        if type(config.data.path) is not str:
            raise Exception(f"Need to provide path of data. get {config.data.path}")
        if not os.path.isdir(config.data.path):
            raise NotADirectoryError(f"{config.data.path} is not a directory")
        if not os.listdir(config.data.path):
            raise FileNotFoundError(f"{config.data.path} do not contains files")

        class Dummy_Wrapping_Dataset(AudioDataset):
            def __getitem__(self, *args, **kwargs):
                x = super().__getitem__(*args, **kwargs)
                return x, 0

        dataset = Dummy_Wrapping_Dataset(
            **config.data.dataset_kwargs,
        )

    else:
        dataset, test_dataset = None, None

    if test_dataset is None and dataset is not None:
        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(1010)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X, as_uint8=True):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0
    return (
        (
            torch.clamp(X, 0.0, 1.0)
            .mul_(255)
            .add_(0.5)
            .clamp_(0, 255)
            .to("cpu", torch.uint8)
            .permute(0, 2, 3, 1)
            .numpy()
        )
        if as_uint8
        else X.to("cpu").permute(0, 2, 3, 1).numpy()
    )
