SLICE_FOLDER = "./B100/f4noattn_step1/"
TOFNAC_FOLDER = "./B100/TOFNAC_resample/"
STEP1_VOLUME_FOLDER = "./B100/f4noattn_step1_volume/"

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MAX_CT = 3976
MIN_CT = -1024
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET


import numpy as np
import nibabel as nib
import glob
import os

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


TOFNAC_list = sorted(glob.glob(TOFNAC_FOLDER+"*.nii.gz"))
print("Found", len(TOFNAC_list), "TOFNAC files")

for idx, TOFNAC_path in enumerate(TOFNAC_list):
    TOFNAC_tag = TOFNAC_path.split('/')[-1].split('.')[0][-5:]
    print(f"Processing [{idx+1}]/[{len(TOFNAC_list)}] {TOFNAC_path} TOFNAC tag is {TOFNAC_tag}")
    TOFNAC_file = nib.load(TOFNAC_path)
    TOFNAC_data = TOFNAC_file.get_fdata()
    len_z = TOFNAC_data.shape[2]
    synCT_data = np.zeros_like(TOFNAC_data)
    
    for idz in range(len_z):
        synCT_path = os.path.join(SLICE_FOLDER, f"STEP2_{TOFNAC_tag}_z{idz}.npy")
        # check if the file exists
        if os.path.exists(synCT_path):
            synCT_slice = np.load(synCT_path)
            synCT_slice = np.clip(synCT_slice, 0, 1)
            synCT_slice = synCT_slice * RANGE_CT + MIN_CT
            synCT_data[:, :, idz] = synCT_slice
        else:
            print(">>> File not found:", synCT_path)
        
    # save the synthetic CT
    synCT_file = nib.Nifti1Image(synCT_data, affine=TOFNAC_file.affine, header=TOFNAC_file.header)
    synCT_path = os.path.join(STEP1_VOLUME_FOLDER, f"STEP2_{TOFNAC_tag}.nii.gz")
    nib.save(synCT_file, synCT_path)
    print(">>> Saved to", synCT_path)