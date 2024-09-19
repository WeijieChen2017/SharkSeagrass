data_folder = "B100/TOFNAC_CTACIVV_part2/TC256_v2/"

import os
import glob

path_list = sorted(glob.glob(f"{data_folder}*_CTAC_256.nii.gz"))
n_cv = 5

# randomly divide the data into 5 folds
import random
random.seed(729)
random.shuffle(path_list)

data_cv = {
    "cv_0": [],
    "cv_1": [],
    "cv_2": [],
    "cv_3": [],
    "cv_4": [],
}

for idx, path in enumerate(path_list):
    cv_idx = idx % n_c
    case_name = os.path.basename(path).split("_")[0]
    data_cv[f"cv_{cv_idx}"].append(case_name)

data_div = {
    "cv_0": {
        "train": data_cv["cv_1"] + data_cv["cv_2"] + data_cv["cv_3"],
        "val": data_cv["cv_4"],
        "test": data_cv["cv_0"],
    },
    "cv_1": {
        "train": data_cv["cv_2"] + data_cv["cv_3"] + data_cv["cv_4"],
        "val": data_cv["cv_0"],
        "test": data_cv["cv_1"],
    },
    "cv_2": {
        "train": data_cv["cv_3"] + data_cv["cv_4"] + data_cv["cv_0"],
        "val": data_cv["cv_1"],
        "test": data_cv["cv_2"],
    },
    "cv_3": {
        "train": data_cv["cv_4"] + data_cv["cv_0"] + data_cv["cv_1"],
        "val": data_cv["cv_2"],
        "test": data_cv["cv_3"],
    },
    "cv_4": {
        "train": data_cv["cv_0"] + data_cv["cv_1"] + data_cv["cv_2"],
        "val": data_cv["cv_3"],
        "test": data_cv["cv_4"],
    },
}

import json
with open("UNetUNet_v1_data_split_acs.json", "w") as f:
    json.dump(data_div, f, indent=4)

print("Data split saved to UNetUNet_v1_data_split_acs.json")
