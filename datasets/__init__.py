import os
import sys

import numpy as np
from torch.utils.data import Subset

sys.path.append("External")

from SST.utils import AudioDataset


def get_dataset(args, config):
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
            path=config.data.path,
            **vars(config.data.dataset_kwargs),
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
