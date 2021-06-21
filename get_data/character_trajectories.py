import numpy as np
import torch
from common import (
    add_time,
    make_directory,
    open_npz,
    process_all_interpolations,
    reduce_tensor_samples,
)


if __name__ == "__main__":
    # Load data and all labels_split
    npz = np.load("../data/raw/UEA/CharacterTrajectories/data.npz", allow_pickle=True)
    static_data = None
    temporal_data = open_npz(npz, "data")
    labels = open_npz(npz, "labels").reshape(-1, 1)

    # Add time
    for i in range(len(temporal_data)):
        temporal_data[i] = torch.cat(
            [torch.arange(len(temporal_data[i])).reshape(-1, 1), temporal_data[i]],
            axis=1,
        )

    # Reduce num samples
    test = True
    test_string = ""
    if test:
        temporal_data, labels = reduce_tensor_samples(
            [temporal_data, labels], num_samples=50
        )
        test_string += "_test"

    # Get all interpolation schemes
    processed_data = process_all_interpolations(static_data, temporal_data, None)

    # Save
    directory = "../data/processed/UEA/CharacterTrajectories"
    make_directory(directory)
    np.savez(
        "{}/improved-neural-cdes_data{}.npz".format(directory, test_string),
        **processed_data,
        labels=labels,
    )
