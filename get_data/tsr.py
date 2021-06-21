import numpy as np
import torch
from common import (
    add_time,
    make_directory,
    open_npz,
    process_all_interpolations,
    reduce_tensor_samples,
    rolling_window,
)
from tqdm import tqdm

TSR_DATASETS = [
    "BeijingPM10Quality",
    "BeijingPM25Quality",
    "BenzeneConcentration",
    # "PPGDalia",
    # "IEEEPPG",
]


def handle_ppgdalia_sampling(temporal_data):
    """PPGDalia has variations in sampling rate, here we mean the accelerometer signal to the same rate."""
    temporal_data[..., :256, 0] = torch.tensor(
        np.nanmean(rolling_window(temporal_data[..., 0], -1, 2, step_size=2), axis=-1),
        dtype=torch.float,
    )
    return temporal_data[..., :256, :]


if __name__ == "__main__":
    # Load data and all labels_split
    for dataset in tqdm(TSR_DATASETS, "TSR processing status"):
        npz = np.load("../data/raw/TSR/{}/data.npz".format(dataset), allow_pickle=True)
        static_data = None
        temporal_data = open_npz(npz, "data")
        labels = open_npz(npz, "labels").reshape(-1, 1)

        # Additional fixes
        if dataset == "PPGDalia":
            temporal_data = handle_ppgdalia_sampling(temporal_data)

        # Add time
        temporal_data = torch.tensor(add_time(temporal_data)).float()

        # Reduce num samples
        test = False
        test_string = ""
        if test:
            temporal_data, labels = reduce_tensor_samples(
                [temporal_data, labels], num_samples=10
            )
            test_string += "_test"

        # Get all interpolation schemes
        processed_data = process_all_interpolations(static_data, temporal_data, None)

        # Save
        directory = "../data/processed/TSR/{}".format(dataset)
        make_directory(directory)
        np.savez(
            "{}/improved-neural-cdes_data{}.npz".format(directory, test_string),
            **processed_data,
            labels=labels,
        )
