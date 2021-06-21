import numpy as np
import torch
from common import add_time, open_npz, process_all_interpolations, reduce_tensor_samples

if __name__ == "__main__":
    # Load data and all labels_split
    npz = np.load(
        "../data/raw/SpeechCommands/SpeechCommands/data_len=89.npz", allow_pickle=True
    )
    static_data = None
    temporal_data = open_npz(npz, "data")
    labels = open_npz(npz, "labels")

    # Reduce num samples
    test = False
    test_str = "_test" if test else ""
    if test:
        temporal_data, labels = reduce_tensor_samples(
            (temporal_data, labels), num_samples=100
        )

    # Get all interpolation schemes
    temporal_data = torch.tensor(add_time(temporal_data))
    processed_data = process_all_interpolations(
        static_data, temporal_data, torch.tensor(labels)
    )

    # Save
    np.savez(
        "../data/processed/SpeechCommands/SpeechCommands/improved-neural-cdes_data{}.npz".format(
            test_str
        ),
        **processed_data,
        labels=labels.to(torch.long).numpy(),
    )
