import os
import glob
import numpy as np
import nibabel as nib

# Part 5
# ----> here we exclude not well matched cases and do normalization for data
# ----> for PET, we use two segment linear norm 
# ----> for CT, we use linear norm

MAX_CT = 2976
MIN_CT = -1024

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET

exclude_case_list = ["E4063", "E4080", "E4087", "E4097", "E4102", "E4203", "E4204", "E4225"]

TOFNAC_dir = "James_data_v3/TOFNAC_256/"
CTACIVV_dir = "James_data_v3/CTACICC_256"

TOFNAC_norm_dir = "James_data_v3/TOFNAC_256_norm/"
CTACIVV_norm_dir = "James_data_v3/CTACIVV_256_norm/"

os.makedirs(TOFNAC_norm_dir, exist_ok=True)
os.makedirs(CTACIVV_norm_dir, exist_ok=True)

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

TOFNAC_path_lists = sorted(glob.glob(TOFNAC_dir + "*.nii.gz"))
casename_list = []

for TOFNAC_path in TOFNAC_path_lists:
    casename = TOFNAC_path.split("/")[-1].split(".")[0]
    # TOFNAC_E4063_256.nii
    casename = casename.split("_")[1]
    if casename in exclude_case_list:
        print(f"Excluding {casename}")
        continue
    
    casename_list.append(casename)
    CTACIVV_path = CTACIVV_dir + "CTACIVV_" + casename + "_aligned.nii.gz"

    TOFNAC_file = nib.load(TOFNAC_path)
    CTACIVV_file = nib.load(CTACIVV_path)

    TOFNAC_data = TOFNAC_file.get_fdata()
    CTACIVV_data = CTACIVV_file.get_fdata()

    TOFNAC_data = np.clip(TOFNAC_data, MIN_PET, MAX_PET)
    CTACIVV_data = np.clip(CTACIVV_data, MIN_CT, MAX_CT)
    TOFNAC_data_norm = two_segment_scale(TOFNAC_data, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
    CTACIVV_data_norm = (CTACIVV_data - MIN_CT) / RANGE_CT

    TOFNAC_data_norm = np.clip(TOFNAC_data_norm, 0, 1)
    CTACIVV_data_norm = np.clip(CTACIVV_data_norm, 0, 1)

    TOFNAC_data_norm = TOFNAC_data_norm.astype(np.float32)
    CTACIVV_data_norm = CTACIVV_data_norm.astype(np.float32)

    TOFNAC_norm_file = nib.Nifti1Image(TOFNAC_data_norm, TOFNAC_file.affine, TOFNAC_file.header)
    CTACIVV_norm_file = nib.Nifti1Image(CTACIVV_data_norm, CTACIVV_file.affine, CTACIVV_file.header)

    TOFNAC_norm_name = TOFNAC_norm_dir + "TOFNAC_" + casename + "_norm.nii.gz"
    CTACIVV_norm_name = CTACIVV_norm_dir + "CTACIVV_" + casename + "_norm.nii.gz"

    nib.save(TOFNAC_norm_file, TOFNAC_norm_name)
    nib.save(CTACIVV_norm_file, CTACIVV_norm_name)
    print(f"Saving {TOFNAC_norm_name} and {CTACIVV_norm_name}")

print(f"Total {len(casename_list)} cases are processed.")
print(casename_list)
    