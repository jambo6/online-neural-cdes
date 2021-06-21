"""
Build the raw csv from the gcloud split datasets.

Ventilation conversion is:
    NaN -> 0
    Oxygen -> 1
    InvasiveVent -> 2
    Trach -> 3
    HighFlow -> 4
    NonInvasiveVent -> 5
"""
import logging
import os
from pathlib import Path

import features
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.root.setLevel(logging.INFO)

RAW_DIR = Path("../../data/raw/mimic-iv")

# Features that go in the final dataset used by the model
FEATURES_TO_USE = {
    "static": [
        "gender",
        "ethnicity",
        "admission_age",
        "height",
        "weight",
        # Labels
        "mortality",
        "los",
    ],
    "temporal": [
        "time",
        # Vitals
        "dbp",
        "dbp_ni",
        "heart_rate",
        "mbp",
        "mbp_ni",
        "o2_flow",
        "resp_rate",
        "sbp",
        "spo2",
        "temperature",
        # Labs
        "alp",
        "ast",
        "baseexcess",
        "bicarbonate_bg",
        "bilirubin_direct",
        "bilirubin_total",
        "bun",
        "calcium",
        "chloride",
        "creatinine",
        "fibrinogen",
        "fio2",
        "glucose",
        "glucose_bg",
        "hematocrit",
        "hemoglobin",
        "lactate",
        "pco2",
        "ph",
        "platelet",
        "po2",
        "potassium_bg",
        "ptt",
        "so2",
        "sodium",
        "troponin_i",
        "wbc",
        # Labels, we will use t_sofa as the definition
        "sepsis",
        "ventilation",
    ],
}


def combine_and_dump():
    # First load everything into one big datafarme
    save_file = RAW_DIR / "raw_combined.csv"
    if os.path.exists(save_file):
        logging.info("Loading the combined data from save...")
        return pd.read_csv(save_file)

    logging.info("Combining the split CSV files...")

    # Create one big frame and save it
    frames = []
    for file in os.listdir(RAW_DIR):
        if file.endswith(".csv"):
            frames.append(pd.read_csv("{}/{}".format(RAW_DIR, file)))
    massive_frame = pd.concat(frames)
    massive_frame.to_csv(save_file)

    return massive_frame


def _build_times(frame):
    # Convert to pandas datetime
    for col in ["charttime", "icu_outtime", "icu_intime", "t_sofa"]:
        frame.loc[:, col] = pd.to_datetime(frame[col])

    # Return measurement time in hours
    frame["time"] = (frame["charttime"] - frame["icu_intime"]).dt.total_seconds() / (
        60 ** 2
    )

    # Make a LOS variable, this is given in days
    frame["los"] = (frame["icu_outtime"] - frame["icu_intime"]).dt.total_seconds() / (
        24 * 60 * 60
    )

    # Mark sepsis times
    frame["sepsis"] = (frame["t_sofa"] - frame["icu_intime"]).dt.total_seconds() / (
        60 ** 2
    )

    return frame


def _convert_categorical(s):
    # Converts a categorical series to binary
    unique_labels = s.unique()
    convert_ints = range(len(unique_labels))
    return s.replace(unique_labels, convert_ints)


def _remove_continuously_monitored(frame, consecutive_time=2, num_consecutive=5):
    # Some patients have continuously monitored vitals signs every minute. This poses a different challenge when dealing
    # with this ICU data and so for simplicity we remove these cases.
    # We define said cases as those that 5 consecutively measured vitals signs within 2 minutes of each measurement
    # These cases can be seen through an exploration of the data
    # (consecutive_time=2, num_consecutive=5) removes 1166 patients
    unique_ids = frame["id"].unique()
    keep_ids = []
    for id_ in tqdm(unique_ids, "Removing continuously monitored patients..."):
        id_frame = frame[frame["id"] == id_]
        times = id_frame[
            "time"
        ].unique()  # Careful here, duplicate times are handled later
        diffs = (times[1:] - times[:-1]) * 60
        mask = (diffs < consecutive_time).reshape(
            -1, 1
        )  # Marked true if within 2 mins measurement
        if len(diffs) > num_consecutive:
            # Perform the shift and if we have 5 in a row then skip
            main_bl = np.concatenate(
                [mask[i : -num_consecutive + i] for i in range(num_consecutive)], axis=1
            ).sum(axis=1)
            if (main_bl == num_consecutive).any():
                continue
        keep_ids.append(id_)
    logging.info(
        "Removed {} continuously monitored patients".format(
            len(unique_ids) - len(keep_ids)
        )
    )
    frame = frame[frame["id"].isin(keep_ids)]
    return frame


def _merge_nearby_vitals(frame):
    """Vitals signs are often measured extremely close to each other, here we merge in such edge cases.

    Often different vitals signs are measured within 1 minute of each other, this can be seen from the data and must
    happen by some artefact of the way the data is input. This function simply aims to combine such nearby
    measurements.

    This function is a little bit tricky, the basic idea is for each id we find the location of any measurement times
    that are within 2 minutes of each other, we then create a column that is indexed with the same number if said
    measurements are within two minutes of each other. A mean function is then applied to the groupby columns.
    """
    logging.info("Merging 'nearby' vitals. This will take a while (~ 4 hours)...")

    def grouped_func(multirow):
        # Function to apply to the grouped dataframe
        if len(multirow) > 1:
            # We dont want to mean time/vent/sepsis
            later_time = multirow["time"].iloc[-1]
            later_vent = multirow["ventilation"].iloc[-1]
            later_sepsis = multirow["sepsis"].iloc[-1]
            idx = multirow.index[-1]
            multirow = multirow.mean(skipna=True, axis=0).to_frame().T
            multirow.index = [idx]
            multirow["time"] = later_time
            multirow["ventilation"] = later_vent
            multirow["sepsis"] = later_sepsis
        return multirow

    def single_merge(df):
        times = df["time"]
        diffs = (times - times.shift(1)) * 60
        mask = diffs < 2
        if mask.any():
            # new_mask contains 0 if the previous measurement was within 2 minutes else 1
            new_mask = (~mask).fillna(True).astype(int)
            new_mask[new_mask == 0] = float("nan")
            # group_col is the range, we then make the 2 min measurements 0 by multiplying with new_mask, finally a
            # forward fill ensures 2 min consecutive measurements share the same value
            df["group_col"] = np.arange(len(df)) * new_mask.values
            df["group_col"] = df["group_col"].ffill()
            df = (
                df.groupby("group_col", as_index=False)
                .apply(grouped_func)
                .drop("group_col", axis=1)
            )
            df = df.droplevel(0)
        return df

    frame = frame.groupby("id", as_index=False).apply(single_merge)

    return frame


def main_processing(frame):
    # For saving
    folder = RAW_DIR / "reduced_feature_dataframes"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    else:
        if os.path.isfile(folder / "static.csv"):
            logging.info(
                "Loading data that exists in {}, remove to reprocess.".format(folder)
            )
            static_data = pd.read_csv(folder / "static.csv", index_col=0)
            temporal_data = pd.read_csv(folder / "temporal.csv", index_col=1).drop(
                "Unnamed: 0", axis=1
            )
            return static_data, temporal_data

    logging.warning(
        "Performing the main processing loop. This includes two big group-by operations that take around "
        "3 hours each. It takes an estimated 8 hours for full processing."
    )

    # Main processing function to convert to physionet challenge style
    frame = frame[features.USE_FEATURES]

    # Generate a times column in hrs and a length of stay
    frame = _build_times(frame)
    frame = frame[frame["time"] > 0]

    # Ventilation marker
    frame["ventilation"] = frame["ventilation_status"].replace(
        (
            float("nan"),
            "Oxygen",
            "InvasiveVent",
            "Trach",
            "HighFlow",
            "NonInvasiveVent",
        ),
        (0, 1, 2, 3, 4, 5),
    )

    # Convert gender and
    for name in ["gender", "ethnicity"]:
        frame[name] = _convert_categorical(frame[name])

    # Renames
    frame.rename(
        columns={"stay_id": "id", "hospital_expire_flag": "mortality"}, inplace=True
    )

    # Some reductions, merge any times that are repeated then remove those who appear continuously monitored
    frame.sort_values("time", inplace=True)  # Careful!
    frame = _remove_continuously_monitored(frame)

    # Make static indexed by id
    static_frame = frame[["id"] + FEATURES_TO_USE["static"]]
    static_frame = (
        static_frame.groupby("id", as_index=True)
        .apply(lambda x: x.iloc[0])
        .drop("id", axis=1)
    )
    static_frame.to_csv(folder / "static.csv")

    # Temporal with some removes
    temporal_frame = frame[["id"] + FEATURES_TO_USE["temporal"]]
    temporal_frame = _merge_nearby_vitals(
        temporal_frame
    )  # Done here so as not to muck up static
    temporal_frame.to_csv(folder / "temporal.csv")

    return static_frame, temporal_frame


def convert_to_numpy(static_frame, temporal_frame):
    fname = RAW_DIR / "reduced_format.npz"
    if os.path.isfile(fname):
        logging.info("numpy data already exists, delete to reconstruct.")
        return None

    logging.info("Preparing conversion to npz format...")

    # Sort
    temporal_frame.sort_values("time", inplace=True)
    unique_ids = static_frame.index.unique()

    # Store static, temporal and the four different label problems.
    static_data = []
    temporal_data = []
    los_data = []
    sepsis_data = []
    ventilation_data = []
    mortality_data = []
    for id_ in tqdm(unique_ids, desc="Converting to numpy format..."):
        # Get temporal first since we skip if < 4 hours or < 4 data points
        all_temporal = temporal_frame[temporal_frame["id"] == id_].drop("id", axis=1)
        if any([len(all_temporal) < 4, all_temporal["time"].max() < 4]):
            continue

        # Append static
        all_static = static_frame.loc[id_]
        static_data.append(all_static.drop(["los", "mortality"]).values)
        los_data.append(all_static["los"])
        mortality_data.append(all_static["mortality"])

        # Append temporal
        temporal_data.append(
            all_temporal.drop(["sepsis", "ventilation"], axis=1).values
        )
        ventilation_data.append(all_temporal[["time", "ventilation"]].values)

        # Sepsis needs some handling, we convert the labels to binary
        t_sepsis = all_temporal["sepsis"].iloc[0]
        sepsis = all_temporal[["time", "sepsis"]].values
        sepsis[:, 1] = 0
        # Update if there is a sepsis location
        if t_sepsis == t_sepsis:
            idx_update = np.argmin(np.abs(sepsis[:, 0] - t_sepsis))
            sepsis[idx_update, 1] = 1
        sepsis_data.append(sepsis)

    logging.info("Saving npz file...")
    np.savez(
        fname,
        static_data=np.stack(static_data),
        temporal_data=np.array(temporal_data, dtype=object),
        los_data=np.stack(los_data),
        mortality_data=np.stack(mortality_data),
        ventilation_data=np.array(ventilation_data, dtype=object),
        sepsis_data=np.array(sepsis_data, dtype=object),
        # Also include column names for completeness
        static_columns=list(static_frame.drop(["los", "mortality"], axis=1).columns),
        temporal_columns=list(
            temporal_frame.drop(["id", "ventilation", "sepsis"], axis=1).columns
        ),
    )


if __name__ == "__main__":
    # Error suppression
    pd.options.mode.chained_assignment = None

    # Load
    raw_combined = combine_and_dump()

    # Process
    static_frame, temporal_frame = main_processing(raw_combined)

    # Convert to numpy arrays
    convert_to_numpy(static_frame, temporal_frame)
