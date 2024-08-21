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

# here we load the data
import os
import nibabel as nib
import numpy as np

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MAX_CT = 3976
MIN_CT = -1024
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET


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

for tag in tag_list:
    CT_path = f"./B100/CTACIVV_resample/CTACIVV_{tag[1:]}.nii.gz"
    PET_path = f"./B100/TOFNAC_resample/PET_TOFNAC_{tag}.nii.gz"
    CT_file = nib.load(CT_path)
    PET_file = nib.load(PET_path)
    CT_data = CT_file.get_fdata()
    PET_data = PET_file.get_fdata()

    print("<"*50)
    print(f"File: CTACIVV_{tag[1:]}.nii.gz")
    print(f"CT shape: {CT_data.shape}, PET shape: {PET_data.shape}")

    # CT data is 467, 467, z, cut it to 400, 400, z
    new_CT_data = CT_data[33:433, 33:433, :]
    print(f"New CT shape: {new_CT_data.shape}")

    # normalize CT data
    output_CT = np.clip(new_CT_data, MIN_CT, MAX_CT)
    output_CT = (output_CT - MIN_CT) / RANGE_CT
    # output_CT = (output_CT - 0.5) * 2

    # normalize PET data
    output_PET = np.clip(PET_data, MIN_PET, MAX_PET)
    output_PET = two_segment_scale(output_PET, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
    # output_PET = (output_PET - 0.5) * 2

    # move the z axis to the first axis
    # output_CT = np.moveaxis(output_CT, -1, 0)
    # output_PET = np.moveaxis(output_PET, -1, 0)
    print(f"Output CT shape: {output_CT.shape}, Output PET shape: {output_PET.shape}")

    # save the npy data
    # output_CT_path = f"./B100/npy/CTACIVV_{tag}.npy"
    # output_PET_path = f"./B100/npy/PET_TOFNAC_{tag}.npy"
    # np.save(output_CT_path, output_CT)
    # np.save(output_PET_path, output_PET)
    # print(f"Saved to {output_CT_path} and {output_PET_path}")

    # save the nifti data
    output_CT_nii = nib.Nifti1Image(output_CT, PET_file.affine, PET_file.header)
    output_PET_nii = nib.Nifti1Image(output_PET, PET_file.affine, PET_file.header)
    output_CT_path = f"./B100/nifti/CTACIVV_{tag}.nii.gz"
    output_PET_path = f"./B100/nifti/PET_TOFNAC_{tag}.nii.gz"
    nib.save(output_CT_nii, output_CT_path)
    nib.save(output_PET_nii, output_PET_path)
    print(f"Saved to {output_CT_path} and {output_PET_path}")