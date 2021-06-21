import numpy as np
import pandas as pd
from helpers import download_zip, unzip
from sktime.utils.data_io import load_from_tsfile_to_dataframe

RAW_DIR = "../../data/raw/UEA/CharacterTrajectories"
URL = "http://www.timeseriesclassification.com/Downloads/CharacterTrajectories.zip"

download_zip(RAW_DIR, "CharacterTrajectories", URL, unzip=False)
unzip(RAW_DIR + "/CharacterTrajectories.zip", RAW_DIR)

# Get ts format
ts_data_list = []
for name in ("TRAIN", "TEST"):
    ts_data_list.append(
        load_from_tsfile_to_dataframe(RAW_DIR + "/CharacterTrajectories_TEST.ts")
    )
ts_data = pd.concat([t[0] for t in ts_data_list])
labels = np.concatenate([[int(x) for x in t[1]] for t in ts_data_list])

# Convert to numpy format
data_list = []
for i in range(len(ts_data)):
    data_list.append(pd.concat(ts_data.iloc[i].values, axis=1).values)

# Save as npz
np.savez(RAW_DIR + "/data.npz", data=data_list, labels=labels)
