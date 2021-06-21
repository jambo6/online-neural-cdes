# flake8: noqa
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sacredex.parse import average_metrics_over_seed, get_dataframe
from utils import get_client, load_datasets_json

# Table save loc
RESULTS_DIR = Path("./results")

# Standard metrics
METRICS = [
    "eval_metric.test",
    "eval_metric.val",
    "eval_metric.train",
    "elapsed_time",
    "memory_usage",
    "num_params",
    "nfe",
    "nfe_per_epoch",
]


# Configs to group and metrics to return for each run
SETTINGS = {
    "hyperopt": {
        "config": ["dataset_name", "dataset.interpolation", "model.model_string"],
    },
    "medical-sota-v2": {
        "metrics": METRICS,
        "config": ["dataset_name", "model.model_string", "dataset.interpolation"],
    },
    "interpolation-v2": {
        "metrics": METRICS,
        "config": ["dataset_name", "dataset.interpolation", "model.interpolation_eps"],
    },
    "sparsity-v2": {
        "metrics": METRICS,
        "config": [
            "dataset_name",
            "model.vector_field",
            "model.vector_field_type",
            "model.sparsity",
        ],
    },
}
for key, value_dict in SETTINGS.items():
    for inner_key, inner_list in SETTINGS[key].items():
        SETTINGS[key][inner_key] = ["{}.{}".format(inner_key, x) for x in inner_list]

# Categorise datasets and give rounding values for each
IRREGULAR_DATASETS = ["Mortality", "Sepsis", "LOS"]
REGULAR_DATASETS = [
    "BeijingPM10",
    "BeijingPM2pt5",
    # "BenzeneConcentration",
    "SpeechCommands",
    "CharacterTrajectories",
]
ROUNDING = {
    "BeijingPM10": 1,
    "BeijingPM2pt5": 1,
    "BenzeneConcentration": 3,
    "LOS": 3,
}
for dataset in IRREGULAR_DATASETS + REGULAR_DATASETS:
    if dataset not in ROUNDING.keys():
        ROUNDING[dataset] = 3


def create_dataset_metric_column(frame):
    """Datasets have different metrics, we put these under a common name 'eval_metric'."""
    dataset_configs = load_datasets_json()

    def func(x, suffix):
        metric = dataset_configs[x["config.dataset_name"]]["evaluation_metric"]
        return x["metrics.{}.{}".format(metric, suffix)]

    for name in ("train", "val", "test"):
        frame["metrics.eval_metric.{}".format(name)] = frame.apply(
            lambda x: func(x, name), axis=1
        )

    return frame


def round_sf(x, sf=3):
    # Round to sf significant figures
    return float(
        np.format_float_positional(
            x, precision=sf, unique=False, fractional=False, trim="k"
        )
    )


def _handle_sparsity(avg_frame):
    # Save loc
    folder = RESULTS_DIR / "sparsity"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # This will be useful for creating masks
    avg_frame_reset = avg_frame.reset_index()

    # First split out matmul sparsity, this is matmul with original/sparse/low-rank vector fields
    # We will return a frame for each dataset with num_params and metric val
    matmul_sparsity = avg_frame[
        avg_frame_reset["config.model.vector_field"]
        .isin(["original", "sparse", "low-rank"])
        .values
    ]
    matmul_sparsity = matmul_sparsity.xs("matmul", level=2)
    matmul_sparsity = matmul_sparsity[
        ["metrics.eval_metric.main", "metrics.num_params.mean"]
    ]
    matmul_sparsity["metrics.num_params.mean"] = matmul_sparsity[
        "metrics.num_params.mean"
    ].astype(int)

    # Renames
    matmul_sparsity.rename(
        columns={
            "metrics.num_params.mean": "Parameter fraction",
            "metrics.eval_metric.main": "Test metric",
        },
        inplace=True,
    )
    matmul_sparsity.index = matmul_sparsity.index.rename(
        ["Dataset", "Vector field", "Sparsity"]
    )

    # Re-orientate so that dataset_name is the column header followed by metric and num params
    matmul_sparsity = (
        matmul_sparsity.reset_index(level=0)
        .pivot(columns="Dataset")
        .swaplevel(0, 1, axis=1)
        .sort_values(by="Dataset", axis=1)
    )

    # Normalise each column by the original number of params
    idx = pd.IndexSlice
    original_param_count = (
        matmul_sparsity.loc[:, idx[:, "Parameter fraction"]].loc["original"].values
    )
    matmul_sparsity.loc[:, idx[:, "Parameter fraction"]] /= original_param_count
    matmul_sparsity.loc[:, idx[:, "Parameter fraction"]] = matmul_sparsity.loc[
        :, idx[:, "Parameter fraction"]
    ].round(2)

    # Make original come first, reduce the presented values
    matmul_sparsity = matmul_sparsity.loc[
        [("original", float("nan"))]
        + [x for x in matmul_sparsity.index if x[0] != "original"]
    ]
    mask = matmul_sparsity.index.get_level_values(1).isin([0.5, 0.7, 0.9, 0.95, 0.99])
    mask[0] = True
    matmul_sparsity = matmul_sparsity.loc[mask]
    matmul_sparsity.index = matmul_sparsity.index.rename(
        ["\textbf{Vector field}", "\textbf{Sparsity}"]
    )

    # (matmul, evaluate, derivative) with original gating
    no_sparse_frame = avg_frame[
        (
            avg_frame_reset["config.model.sparsity"]
            != avg_frame_reset["config.model.sparsity"]
        ).values
    ]
    gating_frame = (
        no_sparse_frame["metrics.eval_metric.main"]
        .droplevel(3)
        .reset_index(0)
        .pivot(columns="config.dataset_name")
        .droplevel(0, axis=1)
    )
    # Order happily works when sorted bwds
    gating_frame = gating_frame.swaplevel(i=0, j=1).sort_index(ascending=False)
    gating_frame.index = gating_frame.index.rename(
        ["\textbf{Vector field type}", "\textbf{Gating}"]
    )
    gating_frame.rename(
        index={
            "matmul": "$f(z) \dby X$",
            "evaluate": "$f(z, x) \dby t$",
            "derivative": "$f(z, \frac{dx}{dt})dt$",
            "original": "Original",
            "minimal": "Minimal",
            "gru": "GRU",
        },
        inplace=True,
    )

    # Save data
    matmul_sparsity[IRREGULAR_DATASETS].to_latex(
        folder / "sparsity_irregular.tex", escape=False
    )
    matmul_sparsity[REGULAR_DATASETS].to_latex(
        folder / "sparsity_regular.tex", escape=False
    )
    gating_frame[IRREGULAR_DATASETS].to_latex(
        folder / "gating_irregular.tex", escape=False
    )
    gating_frame[REGULAR_DATASETS].to_latex(folder / "gating_regular.tex", escape=False)


def _handle_medical_sota(avg_frame):
    sota = (
        avg_frame["metrics.eval_metric.main"]
        .reset_index(0)
        .pivot(columns="config.dataset_name")
    )

    new_index = []
    for ix in sota.index:
        if ix[1] == ix[1]:
            new_index.append("{}-{}".format(ix[0], ix[1]))
        else:
            new_index.append(ix[0])
    sota.index = new_index

    ordered = [
        "gru",
        "gru-dt",
        "gru-intensity",  # Leave out
        "gru-dt-intensity",
        "gru-d",
        "odernn",
        "ncde-rectilinear",
        "ncde-rectilinear-intensity",
    ]
    renamed = [
        "GRU",
        "GRU-dt",
        "GRU-intensity",
        "GRU-dt-intensity",
        "GRU-D",
        "ODE-RNN",
        "NCDE (rectilinear)",
        "NCDE (rectilinear-intensity)",
    ]
    assert set(ordered) == set(
        sota.index
    ), "New models seem to have been added, please update the `ordered` variable."
    sota = sota.loc[ordered].droplevel(0, axis=1)
    sota.index = renamed

    # Save
    sota.to_latex(RESULTS_DIR / "medical-sota.tex", escape=False)


def _handle_interpolation(avg_frame):
    # Save loc
    folder = RESULTS_DIR / "interpolation"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    renames = {
        "cubic": "Natural cubic",
        "linear_cubic_smoothing": "Cubic",
        "linear": "Linear",
        "rectilinear": "Rectilinear",
        "linear_quintic_smoothing": "Quintic",
    }

    # Some name neatening
    avg_frame.index.names = ["Dataset", "Interpolation", "Matching region"]

    # First mark the cases that have interpolation_eps set as null
    avg_frame_for_masking = avg_frame.reset_index()
    eps_bool = (
        avg_frame_for_masking["Matching region"]
        == avg_frame_for_masking["Matching region"]
    ).values
    interp_bool = (~eps_bool) | (avg_frame_for_masking["Matching region"] == 1).values

    # Metric and nfe
    metric_nfe = avg_frame[interp_bool].droplevel(2)[
        [
            "metrics.eval_metric.main",
            "metrics.nfe_per_epoch.mean",
            "metrics.nfe_per_epoch.std",
        ]
    ]
    metric_nfe["metrics.nfe_per_epoch"] = (
        (metric_nfe["metrics.nfe_per_epoch.mean"] / 1e3).round(1).astype(str)
        + " $\pm$ "
        + (metric_nfe["metrics.nfe_per_epoch.std"] / 1e3).round(1).astype(str)
    )
    metric_nfe = metric_nfe[["metrics.eval_metric.main", "metrics.nfe_per_epoch"]]
    metric_nfe = metric_nfe.rename(
        renames,
        level=1,
    )
    metric_nfe.columns = [
        "Metric",
        "NFEs per epoch $\\times 10^3$",
    ]

    # Interpolation eps
    eps_frame = (
        avg_frame[eps_bool]["metrics.eval_metric.main"]
        .reset_index(0)
        .pivot(columns="Dataset")
    )
    eps_frame = eps_frame.droplevel(0, axis=1).rename(renames, level=0)

    # Cant use sepsis here
    IRREGULAR = [
        x for x in IRREGULAR_DATASETS if x in metric_nfe.index.get_level_values(0)
    ]
    REGULAR = [x for x in REGULAR_DATASETS if x in metric_nfe.index.get_level_values(0)]

    # Save
    metric_nfe.loc[REGULAR].to_latex(folder / "metric_nfe_regular.tex", escape=False)
    metric_nfe.loc[IRREGULAR].to_latex(
        folder / "metric_nfe_irregular.tex", escape=False
    )
    # eps_frame[REGULAR].to_latex(folder / "eps_frame_regular.tex", escape=False)
    # eps_frame[IRREGULAR].to_latex(folder / "eps_frame_irregular.tex", escape=False)


FUNCTIONS = {
    "sparsity-v2": _handle_sparsity,
    "medical-sota-v2": _handle_medical_sota,
    "interpolation-v2": _handle_interpolation,
}


if __name__ == "__main__":
    if not os.path.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    # Some setup
    client = get_client()
    # run_names = ["medical-sota", "interpolation", "sparsity"]
    run_names = ["interpolation-v2"]
    test = False

    # Prepend test string if test is set
    db_names = run_names
    if test:
        db_names = ["test_{}".format(x) for x in run_names]

    # Iter
    for run_name, db_name in zip(run_names, db_names):
        # Get the raw frame
        frame = get_dataframe(client[db_name], open_metrics=True)

        # Create a common metric name
        frame = create_dataset_metric_column(frame)

        is_ncde = frame["config.model.model_string"] == "ncde"
        pd.set_option("display.max_columns", 10)

        # Build the seed-averaged metrics frame
        config_cols = SETTINGS[run_name]["config"]
        metric_cols = SETTINGS[run_name]["metrics"]
        avg_frame = average_metrics_over_seed(
            frame, config_cols=config_cols, metric_cols=metric_cols
        )

        # Eval metric -> mean +/- std with problem specific rounding
        cols = ["metrics.eval_metric.test.mean", "metrics.eval_metric.test.std"]
        for dataset in avg_frame.index.get_level_values(0).unique():
            avg_frame.loc[dataset, cols] = (
                avg_frame.loc[dataset, cols].round(ROUNDING[dataset]).values
            )
        mean, std = avg_frame[cols[0]], avg_frame[cols[1]]
        avg_frame["metrics.eval_metric.main"] = (
            mean.astype(str) + " $\pm$ " + std.astype(str)
        )

        # Perform run specific save handling
        FUNCTIONS[run_name](avg_frame)
