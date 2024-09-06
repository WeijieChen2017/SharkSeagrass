# this script is to combine TOFNAC and CTAC images into a single HDF5 file

import glob
import os
import h5py
import random
import nibabel as nib

root_folder = "./B100/TOFNAC_CTAC_hdf5/"

TOFNAC_folder = "B100/TOFNAC_resample/"
CTAC_folder = "B100/CTACIVV_resample/" # this need to cut file from 467, 467, z to 400, 400, z

TOFNAC_list = sorted(glob.glob(f"{TOFNAC_folder}*.nii.gz"))
CTAC_list = sorted(glob.glob(f"{CTAC_folder}*.nii.gz"))

num_TOFNAC = len(TOFNAC_list)
num_CTAC = len(CTAC_list)

n_fold = 5
indices = list(range(num_TOFNAC))
random.shuffle(indices)
hdf5_list = {
    "fold_1": [],
    "fold_2": [],
    "fold_3": [],
    "fold_4": [],
    "fold_5": [],
}

for i in range(num_TOFNAC):
    TOFNAC_path = TOFNAC_list[i]
    CTAC_path = CTAC_list[i]
    print("Processing: ", TOFNAC_path, CTAC_path)

    TOFNAC_data = nib.load(TOFNAC_path).get_fdata()
    CTAC_data = nib.load(CTAC_path).get_fdata()[33:433, 33:433, :]

    print(f"TOFNAC shape: {TOFNAC_data.shape}, CTAC shape: {CTAC_data.shape}")
    current_fold = i % n_fold
    hdf5_list[f"fold_{current_fold + 1}"].append(
        {
            "TOFNAC": TOFNAC_data,
            "CTAC": CTAC_data
        }
    )

# save to hdf5
for fold, data_list in hdf5_list.items():
    print(f"Saving {fold}...")
    with h5py.File(f"{root_folder}{fold}.hdf5", "w") as f:
        for i, data in enumerate(data_list):
            TOFNAC_data = data["TOFNAC"]
            CTAC_data = data["CTAC"]
            f.create_dataset(f"TOFNAC_{i}", data=TOFNAC_data)
            f.create_dataset(f"CTAC_{i}", data=CTAC_data)
    print(f"Saved {fold}")

print("Done")
