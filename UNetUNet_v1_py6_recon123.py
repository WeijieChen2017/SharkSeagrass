TOFNAC_data_folder = "B100/TOFNAC/"
CTAC_data_folder = "B100/CTACIVV/"
CTAC_resample_folder = "B100/CTACIVV_resample/"
TC256_folder = "B100/TC256/"
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


MAX_CT = 1976
MIN_CT = -1024
RANGE_CT = MAX_CT - MIN_CT

save_folder = "B100/DLCTAC/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for tag in tag_list:
    CTAC_path = f"{CTAC_data_folder}CTACIVV_{tag[1:]}_256.nii.gz"
    CTAC_file = nib.load(CTAC_path)
    CTAC_data = CTAC_file.get_fdata()

    # CTAC_resample_path = f"{CTAC_resample_folder}CTACIVV_{tag[1:]}.nii.gz"
    # check if the file exists
    # if not os.path.exists(CTAC_resample_path):
    #     print(f"CTAC file not found for {tag}")
        # continue
    # TC256_path = glob.glob(os.path.join(TC256_folder, f"*{tag[3:]}_CTAC_256.nii.gz"))[0]
    pred_path = glob.glob(os.path.join(pred_folder, f"*{tag[2:]}_CTAC_pred*.nii.gz"))[0]
    # print(f"TC256_path: {TC256_path}")
    # print(f"pred_path: {pred_path}")
    # print(f"{len(pred_path)} files found for {tag}, using {pred_path}")

    # CTAC_resample_file = nib.load(CTAC_resample_path)
    pred_file = nib.load(pred_path)

    # CTAC_resample_data = CTAC_resample_file.get_fdata()
    pred_data = pred_file.get_fdata()

    print("<" * 50)
    print(f"Processing {tag}")
    print(f"CTAC path: {CTAC_path}")
    print(f"pred path: {pred_path}")
    print(f"CTAC shape: {CTAC_data.shape}")
    print(f"pred shape: {pred_data.shape}")


    # pad to CTAC size
    full_data = np.zeros(CTAC_data.shape, dtype=np.float32)
    if CTAC_data.shape[1] == pred_data.shape[1]:
        full_data[21:277, 21:277, :] = pred_data
    else:
        len_CTAC = CTAC_data.shape[2]
        len_pred = pred_data.shape[2]
        pad = (len_CTAC - len_pred) // 2
        full_data[21:277, 21:277, pad:pad+len_pred] = pred_data
    full_data = np.clip(full_data, 0, 1)

    # rescale the data
    full_data = full_data * RANGE_CT + MIN_CT
    
    # save the data
    save_path = os.path.join(save_folder, f"E4{tag[2:]}_CTAC_DL.nii.gz")
    save_nii = nib.Nifti1Image(full_data, CTAC_file.affine, CTAC_file.header)
    nib.save(save_nii, save_path)
    print(f"Data saved at {save_path}")
