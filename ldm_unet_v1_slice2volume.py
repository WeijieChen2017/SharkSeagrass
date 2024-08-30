SLICE_FOLDER = "./B100/f4noattn_step1/"
TOFNAC_FOLDER = "./B100/TOFNAC_resample/"

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


TOFNAC_list = glob.glob(TOFNAC_FOLDER+"*.nii.gz")
print("Found", len(TOFNAC_list), "TOFNAC files")