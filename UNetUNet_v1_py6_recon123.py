TOFNAC_data_folder = "B100/TOFNAC/"
CTAC_data_folder = "B100/CTACIVV/"
CTAC_resample_folder = "B100/CTACIVV_resample/"
pred_folder = "B100/UNetUnet_best/test/"

import os
import glob
import nibabel as nib
import numpy as np

tag_list = [
    "E4055", "E4058", "E4061",          "E4066",
    "E4068", "E4069", "E4073", "E4074", "E4077",
    "E4078", "E4079",          "E4081", "E4084",
             "E4091", "E4092", "E4094", "E4096",
             "E4098", "E4099",          "E4103",
    "E4105", "E4106", "E4114", "E4115", "E4118",
    "E4120", "E4124", "E4125", "E4128", "E4129",
    "E4130", "E4131", "E4134", "E4137", "E4138",
    "E4139",
]

save_folder = "B100/DLCTAC/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for tag in tag_list:
    # CTAC_path = f"{CTAC_data_folder}CTACIVV_{tag[1:]}.nii.gz"
    CTAC_resample_path = f"{CTAC_resample_folder}CTACIVV_{tag[1:]}.nii.gz"
    # check if the file exists
    if not os.path.exists(CTAC_path):
        print(f"CTAC file not found for {tag}")
        # continue
    pred_path = glob.glob(os.path.join(pred_folder, f"*{tag[3:]}_CTAC_pred*.nii.gz"))[0]
    print(f"{len(pred_path)} files found for {tag}, using {pred_path}")
