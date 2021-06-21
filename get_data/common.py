import os

import numpy as np
import torch
from autots import preprocessing
from transformers import Interpolation


def make_directory(loc, file=False):
    """Makes a directory if it doesn't already exist. If loc is specified as a file, ensure the file=True option is set.

    Args:
        loc (str): The file/folder for which the folder location needs to be created.

    Returns:
        True if exists, False if did not exist before.
    """
    existed = True
    loc_ = os.path.dirname(loc) if file else loc
    if not os.path.exists(loc):
        os.makedirs(loc_, exist_ok=True)
        existed = False
    return existed


def open_npz(npz, key):
    """Gets npz files and converts to tensor format."""
    data = npz[key]
    if data.dtype == "O":
        data = [torch.tensor(x, dtype=torch.float) for x in data]
    else:
        try:
            data = torch.tensor(data, dtype=torch.float)
        except Exception as e:
            raise Exception(
                "Could not convert key={} to a tensor with error {}.".format(key, e)
            )
    return data


def static_pipeline(static_data, return_as_numpy=True):
    assert isinstance(static_data, torch.Tensor)
    assert static_data.dim() == 2

    static_out = preprocessing.SimplePipeline(
        [
            preprocessing.NegativeFilter(),
            preprocessing.TensorScaler(method="stdsc"),
            preprocessing.SimpleImputer(strategy="constant", fill_value=0.0),
        ]
    ).fit_transform(static_data)

    if return_as_numpy:
        static_out = static_out.numpy()

    return static_out


def temporal_pipeline(
    temporal_data,
    interpolation_method="linear",
    return_as_numpy=True,
):
    assert len(temporal_data[0].shape) == 2

    # Apply
    temporal_out = Interpolation(method=interpolation_method).fit_transform(
        temporal_data
    )

    # Numpy
    if return_as_numpy:
        if all([len(x) == len(temporal_out[0]) for x in temporal_out]):
            temporal_out = np.stack(temporal_out).astype(np.float32)
        else:
            temporal_out = [x.numpy().astype(np.float32) for x in temporal_out]

    return temporal_out


def normalise(data):
    # Normalise
    if not isinstance(data, torch.Tensor):
        cat_data = torch.cat(data)
    else:
        cat_data = data.clone().reshape(-1, data.shape[-1])
    mean = np.nanmean(cat_data, axis=0)
    std = np.nanstd(cat_data, axis=0)
    data = [(d - mean) / (std + 1e-6) for d in data]
    return data


def process_all_interpolations(
    static_data, temporal_data, stratification_labels=None, split=True
):
    # Create pipelines and apply, save into processed data with ('name', data)
    interpolation_methods = ["linear", "rectilinear", "cubic", "linear_forward_fill"]
    keys = ["static_data"]
    keys += ["temporal_data_{}".format(x) for x in interpolation_methods]
    processed_data = dict.fromkeys(keys)

    # Process static
    if static_data is not None:
        processed_data["static_data"] = static_pipeline(static_data)

    # Temporal
    temporal_data = normalise(temporal_data)
    processed_data["temporal_data_raw"] = temporal_data
    for method in interpolation_methods:
        processed_data["temporal_data_{}".format(method)] = temporal_pipeline(
            temporal_data, method, return_as_numpy=True
        )

    # Get split indices
    if split:
        train_idxs, val_idxs, test_idxs = get_train_test_val_indices(
            len(temporal_data), stratification_labels
        )
        processed_data["train_idxs"] = train_idxs
        processed_data["val_idxs"] = val_idxs
        processed_data["test_idxs"] = test_idxs

    return processed_data


def get_train_test_val_indices(length, stratification_labels=None):
    """Simple train test val splitter."""
    tensors = [torch.arange(length)]
    stratify_index = None
    if stratification_labels is not None:
        tensors.append(stratification_labels)
        stratify_index = 1
    splits = preprocessing.train_val_test_split(
        tensors, stratify_idx=stratify_index, random_state=0
    )
    train_idxs, val_idxs, test_idxs = [s[0] for s in splits]
    return train_idxs, val_idxs, test_idxs


def reduce_tensor_samples(tensors, num_samples=100):
    """Reduce number of samples in each tensor, useful for testing."""
    test_tensors = []
    for tensor in tensors:
        test_tensors.append(tensor[:num_samples])
    return test_tensors


def rolling_window(x, dimension, window_size, step_size=1, return_same_size=True):
    """Outputs an expanded tensor to perform rolling window operations on a pytorch tensor.
    Given an input tensor of shape [N, L, C] and a window length W, computes an output tensor of shape [N, L-W, C, W]
    where the final dimension contains the values from the current timestep to timestep - W + 1.
    Args:
        x (torch.Tensor): Tensor of shape [N, L, C].
        dimension (int): Dimension to open.
        window_size (int): Length of the rolling window.
        step_size (int): Window step, defaults to 1.
        return_same_size (bool): Set True to return a tensor of the same size as the input tensor with nan values filled
                                 where insufficient prior window lengths existed. Otherwise returns a reduced size
                                 tensor from the paths that had sufficient data.
    Returns:
        torch.Tensor: Tensor of shape [N, L, C, W] where the window values are opened into the fourth W dimension.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if return_same_size:
        x_dims = list(x.size())
        x_dims[dimension] = window_size - 1
        nans = np.nan * torch.zeros(x_dims)
        x = torch.cat((nans, x), dim=dimension)

    # Unfold ready for mean calculations
    unfolded = x.unfold(dimension, window_size, step_size)

    return unfolded


def add_time(temporal_data):
    """Add a time column."""
    times = np.repeat(
        np.arange(temporal_data.shape[1]).reshape(1, -1, 1), len(temporal_data), 0
    )
    temporal_data = np.concatenate([times, temporal_data], axis=-1)
    return temporal_data
