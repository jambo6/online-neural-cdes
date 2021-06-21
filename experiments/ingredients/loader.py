# flake8: noqa
import numpy as np
import torch
from autots import preprocessing
from sacred import Ingredient
from torch.utils.data import DataLoader

data_ingredient = Ingredient("dataset")


@data_ingredient.config
def data_config():
    use_static = True
    interpolation = "linear"
    batch_size = 1024
    test_mode = False


@data_ingredient.capture
def load_data(
    data_loc,
    problem=None,
    use_static=None,
    interpolation=None,
    batch_size=None,
    test_mode=None,
):
    # Load
    static_data, temporal_data, labels, *in_out_static_interp = load_numpy_data(
        data_loc,
        interpolation,
        use_static=use_static,
        problem=problem,
        test_mode=test_mode,
    )

    # Create dataloaders, use full batch size for validation and test loaders
    dataloaders = []
    for i, (static, temporal, labels_) in enumerate(
        zip(static_data, temporal_data, labels)
    ):
        dataloader = create_dataloader(
            static, temporal, labels_, interpolation, batch_size
        )
        dataloaders.append(dataloader)

    # Return sequences only if online
    return_sequences = True if problem == "online" else False

    return dataloaders, *in_out_static_interp, return_sequences


def parse_interpolation_string(interpolation):
    """Return the data interpolation string and the model interpolation string."""
    if interpolation in ["linear", "rectilinear", "cubic"]:
        d, m = interpolation, interpolation
    elif interpolation == "rectilinear-intensity":
        d, m = "rectilinear", interpolation
    elif interpolation in ["linear_cubic_smoothing", "linear_quintic_smoothing"]:
        d, m = "linear", interpolation
    elif interpolation == "linear_forward_fill":
        d, m = interpolation, "linear"
    elif interpolation == "cubic_forward_fill":
        d, m = "linear_forward_fill", "linear_cubic_smoothing"
    elif interpolation is None:
        d, m = "raw", None
    else:
        raise NotImplementedError(
            "Not implemented for interpolation {}".format(interpolation)
        )
    return d, m


def load_numpy_data(data_loc, interpolation, use_static, problem, test_mode):
    """Load numpy data and convert to tensor format."""
    # Load all the data in
    npz = np.load(
        "../data/processed/{}/improved-neural-cdes_data{}.npz".format(
            data_loc, "_test" if test_mode else ""
        ),
        allow_pickle=True,
    )

    # Parse interpolation string
    data_interpolation, model_interpolation = parse_interpolation_string(interpolation)

    # Get the relevant pieces
    # Bit of a hack, no static if no interpolation
    if all([use_static, "static_data" in npz.files, interpolation is not None]):
        static_data = npz["static_data"]
    else:
        use_static = False
    temporal_data = npz["temporal_data_{}".format(data_interpolation)]
    labels = npz["labels"]
    splits = [npz[x] for x in ("train_idxs", "val_idxs", "test_idxs")]

    # Intensity for interpolation schemes, cumsummed so that the delta is 1 if value has been measured
    # Hacky hacky but cant get the information from the rectilinear scheme, needed to be put there originally
    # Todo: just make a rectilinear-intensity array in get_data
    if model_interpolation == "rectilinear-intensity":
        model_interpolation = "rectilinear"
        raw_data = npz["temporal_data_raw"]
        dtype = temporal_data[0].dtype
        for i in range(len(temporal_data)):
            tdata = torch.tensor(np.copy(raw_data[i]))
            tdata[0, :][tdata[0, :] == 0] = float("nan")
            intensity_cumsum = (
                (~tdata[:, 1:].isnan()).cumsum(axis=0).repeat_interleave(2, 0)
            )
            intensity_cumsum = intensity_cumsum[:-1]
            temporal_data[i] = np.concatenate(
                [temporal_data[i], intensity_cumsum.numpy().astype(dtype)], axis=1
            )

    # Convert to torch
    static_data = torch.tensor(static_data).to(torch.float) if use_static else None
    if not isinstance(temporal_data, np.ndarray):
        temporal_data = np.array([x for x in temporal_data], dtype=object)
    if problem == "online":
        assert isinstance(labels, np.ndarray)
    else:
        labels = torch.tensor(labels).to(torch.float)

    # Get dimension information
    input_dim = (
        int(temporal_data[0].shape[-1] / 4)
        if data_interpolation == "cubic"
        else temporal_data[0].shape[-1]
    )
    output_dim = 1
    static_dim = static_data.shape[-1] if use_static else None

    # Split with conversion just to be sure
    static_data = (
        [static_data[idxs] for idxs in splits] if use_static else [None, None, None]
    )
    temporal_data = [temporal_data[idxs] for idxs in splits]
    labels = [labels[idxs] for idxs in splits]

    # Perform sorting on PhysioNet due to unequal lengths
    # This is all a bit hacky and should have been handling in the data processing scripts
    if any([x in data_loc for x in ("MIMIC-IV", "CharacterTrajectories")]):
        for i, (static, temporal, labs) in enumerate(
            zip(static_data, temporal_data, labels)
        ):
            # Sort
            static_data[i], temporal_data[i], labels[i], indices = sort_unequal_lengths(
                static, temporal, labs
            )
        if "CharacterTrajectories" in data_loc:
            output_dim = 20
            labels = [x.to(torch.long).reshape(-1) - 1 for x in labels]
    elif "SpeechCommands" in data_loc:
        output_dim = 10
        labels = [x.to(torch.long) for x in labels]
    else:
        assert all([len(x) == len(temporal_data[0][0]) for x in temporal_data[0]])

    return (
        static_data,
        temporal_data,
        labels,
        input_dim,
        output_dim,
        static_dim,
        model_interpolation,
    )


def sort_unequal_lengths(static, temporal, labels):
    # If the lengths are unequal, sort to shortest first
    lengths = [len(x) for x in temporal]
    sorted_indices = sorted(range(len(lengths)), key=lambda k: lengths[k])
    static = static[sorted_indices] if static is not None else None
    temporal = temporal[sorted_indices]
    labels = labels[sorted_indices]
    return static, temporal, labels, sorted_indices


def create_dataloader(static, temporal, labels, interpolation, batch_size):
    # If the data is unequal length, first sorts the data and pads up to batch size
    if not isinstance(temporal, torch.Tensor):
        assert all(
            [len(temporal[i]) <= len(temporal[i + 1]) for i in range(len(temporal) - 1)]
        ), (
            "Data is of unequal length and has not been sorted. This will lead to slow training, "
            "please sort the data in order of length first."
        )

        # A bit hacky, we only want forward fill if an NCDE
        if interpolation is None:
            temporal_pipe = preprocessing.PadRaggedTensors()
        else:
            temporal_pipe = preprocessing.SimplePipeline(
                [preprocessing.PadRaggedTensors(), preprocessing.ForwardFill()]
            )
        labels_pipe = preprocessing.PadRaggedTensors()

        def padder(data_list, pipe):
            padded = [
                pipe.transform(data_list[i : i + batch_size])
                for i in range(0, len(data_list), batch_size)
            ]
            return [x for tens in padded for x in tens]

        temporal = padder(temporal, temporal_pipe)
        if not isinstance(labels, torch.Tensor):
            labels = padder(labels, labels_pipe)

    dataset = StaticTemporalDataset(static, temporal, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


class StaticTemporalDataset:
    """Can handle static temporal tuple outputs."""

    def __init__(self, static_data=None, temporal_data=None, labels=None):
        self.static_data = static_data
        self.temporal_data = temporal_data
        self.labels = labels

    def __len__(self):
        return len(self.temporal_data)

    def __getitem__(self, item):
        if self.static_data is None:
            return self.temporal_data[item], self.labels[item]
        else:
            return (self.static_data[item], self.temporal_data[item]), self.labels[item]
