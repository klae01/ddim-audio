import os
import sys

import numpy as np
from torch.utils.data import Subset

sys.path.append("External")

from SST.utils import AudioDataset, config as SST_config


def get_dataset(args, config, mapping):
    # take only data related config
    def get_log_data_spec(data):
        if mapping.log_polar:
            eps = np.exp(-16).tolist()
            C_axis = config.axis.index("C")
            if config.dataset_kwargs.use_numpy:
                X = np.linalg.norm(data, ord=2, axis=C_axis)
                X = np.log(X + eps)
                get_log_data_spec.log_data_spec = {
                    "eps": eps,
                    "mean": X.mean().tolist(),
                    "std": X.std().tolist(),
                }
            else:
                X = data.norm(p=2, dim=C_axis)
                X = (X + eps).log()
                get_log_data_spec.log_data_spec = {
                    "eps": eps,
                    "mean": X.mean().item(),
                    "std": X.std().item(),
                }
        else:
            get_log_data_spec.log_data_spec = None

        return data

    get_log_data_spec.log_data_spec = None

    dataset, test_dataset = None, None
    if config.dataset == "AUDIO":
        if type(config.path) is not str:
            raise Exception(f"Need to provide path of data. get {config.path}")
        if not os.path.isdir(config.path):
            raise NotADirectoryError(f"{config.path} is not a directory")
        if not os.listdir(config.path):
            raise FileNotFoundError(f"{config.path} do not contains files")

        class Dummy_Wrapping_Dataset(AudioDataset):
            def __getitem__(self, *args, **kwargs):
                x = super().__getitem__(*args, **kwargs)
                return x, 0

        dataset = Dummy_Wrapping_Dataset(
            path=config.path,
            config=SST_config(**vars(config.dataset_kwargs)),
            transform=get_log_data_spec,
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

    return dataset, test_dataset, get_log_data_spec.log_data_spec
