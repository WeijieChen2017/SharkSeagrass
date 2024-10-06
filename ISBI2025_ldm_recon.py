model_spec_list = ["f4", "f4-noattn", "f8", "f8-n256", "f16"]

case_list = [
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

# CT unit is HU
# PET unit is Bq/ml

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes
import json

def two_segment_scale(arr, MIN, MID, MAX, MIQ):
    # Create an empty array to hold the scaled results
    scaled_arr = np.zeros_like(arr, dtype=np.float32)

    # First segment: where arr <= MID
    mask1 = arr <= MID
    scaled_arr[mask1] = (arr[mask1] - MIN) / (MID - MIN) * MIQ

    # Second segment: where arr > MID
    mask2 = arr > MID
    scaled_arr[mask2] = MIQ + (arr[mask2] - MID) / (MAX - MID) * (1 - MIQ)
    
    return scaled_arr

def reverse_two_segment_scale(arr, MIN, MID, MAX, MIQ):
    # Create an empty array to hold the reverse scaled results
    reverse_scaled_arr = np.zeros_like(arr, dtype=np.float32)

    # First segment: where arr <= MIQ
    mask1 = arr <= MIQ
    reverse_scaled_arr[mask1] = arr[mask1] * (MID - MIN) / MIQ + MIN

    # Second segment: where arr > MIQ
    mask2 = arr > MIQ
    reverse_scaled_arr[mask2] = MID + (arr[mask2] - MIQ) * (MAX - MID) / (1 - MIQ)
    
    return reverse_scaled_arr

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MAX_CT = 3976
MIN_CT = -1024
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET

CT_mask_folder = "B100/CTACIVV_resample_mask/"
HU_boundary_valid_air = -500
HU_boundary_air_soft = -250
HU_boundary_soft_bone = 150



for model_spec in model_spec_list:

    metrics_dict = {
        "CT_MAE_whole": [],
        "PET_MAE_whole": [],
        "CT_MAE_air": [],
        "PET_MAE_air": [],
        "CT_MAE_soft": [],
        "PET_MAE_soft": [],
        "CT_MAE_bone": [],
        "PET_MAE_bone": [],
    }
    result_save_json = f"ISBI2025_ldm_recon_metrics_{model_spec}.json"

    print("Processing model spec: ", model_spec)
    recon_folder = f"B100/vq_{model_spec}_recon_nifit/"
    for casename in case_list:
        print("Processing case: ", casename)
        # vq_f8_E4077_CTr_recon.nii.gz
        CT_path = f"vq_{model_spec}_{casename}_CTr_recon.nii.gz"
        CT_path = os.path.join(recon_folder, CT_path)
        # vq_f8_E4077_PET_recon.nii.gz
        PET_path = f"vq_{model_spec}_{casename}_PET_recon.nii.gz"
        PET_path = os.path.join(recon_folder, PET_path)

        CT_file = nib.load(CT_path)
        PET_file = nib.load(PET_path)
        CT_data = CT_file.get_fdata()
        PET_data = PET_file.get_fdata()

        # CT data is from -1 to 1
        # PET data is from -1 to 1
        # scale the data to the original range
        CT_data_denorm = (CT_data + 1) / 2 * RANGE_CT + MIN_CT
        PET_data_denorm = (PET_data + 1) / 2
        PET_data_denorm = reverse_two_segment_scale(PET_data_denorm, MIN_PET, MID_PET, MAX_PET, MIQ_PET)

        CT_GT_path = f"B100/CTACIVV_resample/CTACIVV_{casename[1:]}.nii.gz"
        PET_GT_path = f"B100/TOFNAC_resample/PET_TOFNAC_{casename}.nii.gz"

        CT_GT_file = nib.load(CT_GT_path)
        PET_GT_file = nib.load(PET_GT_path)

        CT_GT_data = CT_GT_file.get_fdata()[33:433, 33:433, :]
        PET_GT_data = PET_GT_file.get_fdata()

        # clip the data to the original range
        CT_GT_data = np.clip(CT_GT_data, MIN_CT, MAX_CT)
        PET_GT_data = np.clip(PET_GT_data, MIN_PET, MAX_PET)
        CT_data_denorm = np.clip(CT_data_denorm, MIN_CT, MAX_CT)
        PET_data_denorm = np.clip(PET_data_denorm, MIN_PET, MAX_PET)

        # compute the mask using CT_GT_data if the mask does not exist
        mask_CT_whole_path = os.path.join(CT_mask_folder, f"CT_mask_{casename}.nii.gz")
        mask_CT_air_path = os.path.join(CT_mask_folder, f"CT_mask_air_{casename}.nii.gz")
        mask_CT_soft_path = os.path.join(CT_mask_folder, f"CT_mask_soft_{casename}.nii.gz")
        mask_CT_bone_path = os.path.join(CT_mask_folder, f"CT_mask_bone_{casename}.nii.gz")

        if os.path.exists(mask_CT_whole_path):
            mask_CT_whole_file = nib.load(mask_CT_whole_path)
            mask_CT_whole = mask_CT_whole_file.get_fdata()
            mask_CT_whole = mask_CT_whole > 0

            mask_CT_air_file = nib.load(mask_CT_air_path)
            mask_CT_air = mask_CT_air_file.get_fdata()
            mask_CT_air = mask_CT_air > 0

            mask_CT_soft_file = nib.load(mask_CT_soft_path)
            mask_CT_soft = mask_CT_soft_file.get_fdata()
            mask_CT_soft = mask_CT_soft > 0

            mask_CT_bone_file = nib.load(mask_CT_bone_path)
            mask_CT_bone = mask_CT_bone_file.get_fdata()
            mask_CT_bone = mask_CT_bone > 0
        else:
            mask_CT = CT_GT_data > -500
            for i in range(CT_GT_data.shape[2]):
                mask_CT[:, :, i] = binary_fill_holes(mask_CT[:, :, i])
            
            # save the mask
            mask_CT_file = nib.Nifti1Image(mask_CT.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
            nib.save(mask_CT_file, mask_CT_path)
            print("Saved mask to: ", mask_CT_path)

            # air mask is from MIN to HU_boundary_air_soft
            mask_CT_air = (CT_GT_data > MIN_CT) & (CT_GT_data < HU_boundary_air_soft)
            # intersection with the whole mask
            mask_CT_air = mask_CT_air & mask_CT
            # save the mask
            mask_CT_air_file = nib.Nifti1Image(mask_CT_air.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
            nib.save(mask_CT_air_file, mask_CT_air_path)
            print("Saved air mask to: ", mask_CT_air_path)

            # soft mask is from HU_boundary_air_soft to HU_boundary_soft_bone
            mask_CT_soft = (CT_GT_data > HU_boundary_air_soft) & (CT_GT_data < HU_boundary_soft_bone)
            # intersection with the whole mask
            mask_CT_soft = mask_CT_soft & mask_CT
            # save the mask
            mask_CT_soft_file = nib.Nifti1Image(mask_CT_soft.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
            nib.save(mask_CT_soft_file, mask_CT_soft_path)
            print("Saved soft mask to: ", mask_CT_soft_path)

            # bone mask is from HU_boundary_soft_bone to MAX
            mask_CT_bone = (CT_GT_data > HU_boundary_soft_bone) & (CT_GT_data < MAX_CT)
            # intersection with the whole mask
            mask_CT_bone = mask_CT_bone & mask_CT
            # save the mask
            mask_CT_bone_file = nib.Nifti1Image(mask_CT_bone.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
            nib.save(mask_CT_bone_file, mask_CT_bone_path)
            print("Saved bone mask to: ", mask_CT_bone_path)

        # start to compute the metrics
        # compute the metrics for the whole mask
        CT_MAE_whole = np.mean(np.abs(CT_GT_data[mask_CT_whole] - CT_data_denorm[mask_CT_whole]))
        PET_MAE_whole = np.mean(np.abs(PET_GT_data[mask_CT_whole] - PET_data_denorm[mask_CT_whole]))
        CT_MAE_air = np.mean(np.abs(CT_GT_data[mask_CT_air] - CT_data_denorm[mask_CT_air]))
        PET_MAE_air = np.mean(np.abs(PET_GT_data[mask_CT_air] - PET_data_denorm[mask_CT_air]))
        CT_MAE_soft = np.mean(np.abs(CT_GT_data[mask_CT_soft] - CT_data_denorm[mask_CT_soft]))
        PET_MAE_soft = np.mean(np.abs(PET_GT_data[mask_CT_soft] - PET_data_denorm[mask_CT_soft]))
        CT_MAE_bone = np.mean(np.abs(CT_GT_data[mask_CT_bone] - CT_data_denorm[mask_CT_bone]))
        PET_MAE_bone = np.mean(np.abs(PET_GT_data[mask_CT_bone] - PET_data_denorm[mask_CT_bone]))

        metrics_dict["CT_MAE_whole"].append(CT_MAE_whole)
        metrics_dict["PET_MAE_whole"].append(PET_MAE_whole)
        metrics_dict["CT_MAE_air"].append(CT_MAE_air)
        metrics_dict["PET_MAE_air"].append(PET_MAE_air)
        metrics_dict["CT_MAE_soft"].append(CT_MAE_soft)
        metrics_dict["PET_MAE_soft"].append(PET_MAE_soft)
        metrics_dict["CT_MAE_bone"].append(CT_MAE_bone)
        metrics_dict["PET_MAE_bone"].append(PET_MAE_bone)

    # save the dict
    metric_dict_name = f"ISBI2025_ldm_recon_metrics_dict_{model_spec}.npy"
    np.save(metric_dict_name, metrics_dict)
    print("Saved metrics dict to: ", metric_dict_name)

    for key in metrics_dict.keys():
        metrics_dict[key] = np.mean(metrics_dict[key])
    
    # in json, output metric names first per row
    with open(result_save_json, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print("Saved metrics to: ", result_save_json)

    print("Metrics: ", metrics_dict)
    print(">"*50)
    print()

 


        





