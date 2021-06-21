"""Builds the 4 MIMIC-IV problems. Each problem has different exclusions and label processing functions.

Task overview:
    1. Mortality - Only consider first 72 hours of the ICU stay, aim is to predict eventual mortality.
    2. LOS - Only consider patients who were in the ICU for < 72 hours total (otherwise we have a tail problem). This is
    slightly different to the above where we use all data truncated to 72 hours.
    3. Ventilation - Up to 72 hours and we censor 3 hours before ventilation.
    4. Sepsis - Prediction of sepsis in [-12, 6] window around the first occurrence. Filter patients who have a first
    t_sofa < 4 hours.
"""
# flake8: noqa
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("../")
import common

logging.root.setLevel(logging.INFO)


def _select_keep_idxs(data, keep_idxs):
    if isinstance(data, list):
        output = [data[idx] for idx in keep_idxs]
    else:
        output = data[keep_idxs]
    return output


def _assert_continuous_label_lengths(temporal_data, labels):
    # Check each label has the same length as the temporal data
    assert all([len(x) == len(y) for x, y in zip(temporal_data, labels)])


def _exclude_times(static_data, temporal_data, labels=None, max_time=72, method="drop"):
    """Exclusion for data points greater than a specified time.

    Args:
        max_time (float): The maximum allowed time in hours.
        method (str): One of ('drop', 'reduce'). If drop is set will drop patients with time > max_time, if reduce is
            set reduces the data to only contain times < max_time.

    Returns:
        A list containing the indexes to keep in the case of method='drop', or a list containing tensor masks
    """
    assert method in ("drop", "reduce")

    # Reduce
    drop_idxs = []
    for idx in range(len(temporal_data)):
        if method == "reduce":
            mask = temporal_data[idx][:, 0] <= max_time
            temporal_data[idx] = temporal_data[idx][mask]
            if labels is not None:
                labels[idx] = labels[idx][mask]
        else:
            if max(temporal_data[idx][:, 0]) > max_time:
                drop_idxs.append(idx)

    # Remove patients with time > max time if method is set to drop
    if len(drop_idxs) > 0:
        keep_idxs = [x for x in range(len(static_data)) if x not in drop_idxs]
        static_data = _select_keep_idxs(static_data, keep_idxs)
        temporal_data = _select_keep_idxs(temporal_data, keep_idxs)
        labels = _select_keep_idxs(labels, keep_idxs)

    return static_data, temporal_data, labels


def _los_exclusions(static_data, temporal_data, labels):
    """Additional exclusions for LOS else classification problem becomes a little weird.

    Here we take patients who have 24 < discharge time < 72 and aim to predict LOS from the 24th hour of data. If we had
    kept < 24hrs then it becomes easy to predict their discharge (if we do normal batching), and > 72 is a long tail so
    the results will become skewed.
    """
    # Patients with more than 24 hrs of data
    keep_idxs = []
    for idx in range(len(temporal_data)):
        temporal = temporal_data[idx]
        times = temporal[:, 0]
        max_time = max(times)
        if max_time > 24:
            continue
        else:
            new_temporal = temporal[times <= 24]
            if len(new_temporal) > 4:
                keep_idxs.append(idx)
                temporal_data[idx] = new_temporal

    # Apply the reduction
    static_data = _select_keep_idxs(static_data, keep_idxs)
    temporal_data = _select_keep_idxs(temporal_data, keep_idxs)
    labels = _select_keep_idxs(labels, keep_idxs)

    return static_data, temporal_data, labels


def _vent_exclusions(static_data, temporal_data, labels):
    """Exclusions for predicting whether or not someone will need ventilation in the next 12 hours.

    Here we consider patients who get some form of ventilation and are in the ICU between 24 and 72 hours. We then try
    to predict the shifted label problem.
    """
    keep_idxs = []
    for idx in tqdm(range(len(labels)), desc="Performing ventilation exclusions."):
        times, labels_ = labels[idx][:, 0], labels[idx][:, 1]
        max_time = times.max()
        # Keep ventilation and > 24 hours
        if (labels_.max() > 0) and (max_time > 24) and (max_time < 72):
            on_vent = np.isin(labels_, [1, 2, 3, 5])
            labels_[on_vent] = 1
            labels_[~on_vent] = 0

            # Now make the new labels be the time shifted labels
            # Here we are just making the label the label closest to 12 hours ahead of time
            # This is not perfect because not all times are equal
            new_labels = labels_.clone()
            for i in range(len(new_labels)):
                time = times[i]
                # If time exceeds max_time - 12, mask the ends and break
                if time >= max_time - 12:
                    mask = times <= max_time - 12
                    temporal_data[idx] = temporal_data[idx][mask]
                    update = new_labels[mask]
                    labels[idx] = update
                    # Make sure it hasnt been reduced to less than 4 points
                    if len(update) >= 4:
                        keep_idxs.append(idx)
                    break
                query_time = time.item() + 12
                query_idx = torch.argmin((times - query_time).abs()).item()
                new_labels[i] = labels_[query_idx]

    static_data = _select_keep_idxs(static_data, keep_idxs)
    temporal_data = _select_keep_idxs(temporal_data, keep_idxs)
    labels = _select_keep_idxs(labels, keep_idxs)

    return static_data, temporal_data, labels


def _process_continuous_labels(
    static_data, temporal_data, labels, exclude_before=4, lookback=12, lookforward=6
):
    """Handles the processing required for cfroiontinuous label exclusion and binarizing.

    Given some continuous binary labels such as [0, 0, 1, 0, 1, 1, 0, 1, 0] we perform as follows:
        1. Exclude all patients with postive labels occuring before t=`exclude_before`.
        2. Mark labels as 1 in t_first_pos - lookback < t < t_first_pos + lookforward.
        3. Censor data from t > t_first_pos + lookforward.
    """
    keep_idxs = []
    for idx, label in enumerate(labels):
        # Logic for positive labellings
        times = label[:, 0]
        ls = label[:, 1]
        if ls.max() > 0:
            # Time of first positive label
            first_time = min(times[ls == 1])
            # Drop if fist_time < exclude_before
            if first_time < exclude_before:
                continue
            else:
                # Mark 1 in the lookback lookforward region
                ones_mask = (times >= first_time - lookback) & (
                    times <= first_time + lookforward
                )
                label[ones_mask] = 1
                # Perform post censoring on labels and temporal data
                keep_mask = times <= first_time + lookforward
                update = ls[keep_mask]
                if len(update) < 4:
                    continue
                labels[idx] = ls[keep_mask]
                temporal_data[idx] = temporal_data[idx][keep_mask]
        else:
            labels[idx] = ls

        keep_idxs.append(idx)

    # Reduce
    static_data = _select_keep_idxs(static_data, keep_idxs)
    temporal_data = _select_keep_idxs(temporal_data, keep_idxs)
    labels = _select_keep_idxs(labels, keep_idxs)

    return static_data, temporal_data, labels


def perform_exclusions(name, static_data, temporal_data, labels):
    # 1. For every task, exclude any patient with more than 72 hours worth of data.
    static_data, temporal_data, labels = _exclude_times(
        static_data, temporal_data, labels, method="drop", max_time=72
    )

    # If LOS use people with stays in 24 < t < 72 and consider only the first 24 hours worth of data
    if name == "LOS":
        static_data, temporal_data, labels = _los_exclusions(
            static_data, temporal_data, labels
        )
    # Additionally handle continuous labellings
    elif name == "Sepsis":
        static_data, temporal_data, labels = _process_continuous_labels(
            static_data,
            temporal_data,
            labels,
            exclude_before=4,
            lookback=12,
            lookforward=6,
        )
    # Ventilation
    elif name == "Ventilation":
        static_data, temporal_data, labels = _vent_exclusions(
            static_data, temporal_data, labels
        )
    # Reshaping
    if name in ("LOS", "Mortality"):
        labels = labels.reshape(-1, 1)
    else:
        _assert_continuous_label_lengths(temporal_data, labels)
        labels = [x.reshape(-1, 1) for x in labels]

    return static_data, temporal_data, labels


def process_interpolate_and_save(name, labels, top_folder, test=False):
    """Processing function for a given problem with associated set of labels."""
    # Handle save locations
    save_folder = top_folder / name
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    fname = save_folder / "improved-neural-cdes_data{}.npz".format(
        "_test" if test else ""
    )
    # if os.path.exists(fname):
    #     logging.warning("{} already exists, delete to reconstruct.".format(fname))
    #     return None

    # Reload the data
    npz = np.load(raw_folder / "reduced_format.npz", allow_pickle=True)
    static_data = common.open_npz(npz, "static_data")
    temporal_data = common.open_npz(npz, "temporal_data")

    # Test mode
    if test:
        static_data, temporal_data, labels = common.reduce_tensor_samples(
            (static_data, temporal_data, labels),
            num_samples=100,
        )

    # Problem specific exclusions and label processing
    static_data, temporal_data, labels = perform_exclusions(
        name, static_data, temporal_data, labels
    )
    _assert_continuous_label_lengths(temporal_data, labels)

    # Build the interpolations
    logging.info("Processing interpolations, this is likely to take a LONG time.")
    processed_data = common.process_all_interpolations(static_data, temporal_data, None)

    # Save
    np.savez(
        fname,
        **processed_data,
        labels=labels,
    )


if __name__ == "__main__":
    # Location setup
    raw_folder = Path("../../data/raw/mimic-iv")
    processed_folder = Path("../../data/processed/MIMIC-IV")
    save_folder = Path("../../data/processed/MIMIC-IV")
    if not os.path.isdir(processed_folder):
        os.mkdir(processed_folder)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    # Load everything
    npz = np.load(raw_folder / "reduced_format.npz", allow_pickle=True)
    labels_los = common.open_npz(npz, "los_data")
    labels_mortality = common.open_npz(npz, "mortality_data")
    labels_vent = common.open_npz(npz, "ventilation_data")
    labels_sepsis = common.open_npz(npz, "sepsis_data")

    named_labels = [
        ("Sepsis", labels_sepsis),
        ("Mortality", labels_mortality),
        ("LOS", labels_los),
        ("Ventilation", labels_vent),
    ]

    test = False
    if test:
        logging.info("Running for test set!")

    for name, labels in tqdm(named_labels, desc="Processing and interpolating."):
        process_interpolate_and_save(name, labels, save_folder, test=test)
