# this script is to combine TOFNAC and CTAC images into a single HDF5 file

import glob
import os
import h5py
import random
import nibabel as nib

root_folder = "./B100/TOFNAC_CTAC_hdf5/"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

TOFNAC_folder = "B100/TOFNAC_resample/"
CTAC_folder = "B100/CTACIVV_resample/" # this need to cut file from 467, 467, z to 400, 400, z

TOFNAC_list = sorted(glob.glob(f"{TOFNAC_folder}*.nii.gz"))
CTAC_list = sorted(glob.glob(f"{CTAC_folder}*.nii.gz"))

num_TOFNAC = len(TOFNAC_list)
num_CTAC = len(CTAC_list)

n_fold = 5

# iterate the fold
for i_fold in range(n_fold):

    data_fold = []

    for i_case in range(num_TOFNAC):
        if not i_case % n_fold == i_fold:
            continue

        TOFNAC_path = TOFNAC_list[i_case]
        CTAC_path = CTAC_list[i_case]
        print("Processing: ", TOFNAC_path, CTAC_path)

        TOFNAC_data = nib.load(TOFNAC_path).get_fdata()
        CTAC_data = nib.load(CTAC_path).get_fdata()[33:433, 33:433, :]

        print(f">>> TOFNAC shape: {TOFNAC_data.shape}, CTAC shape: {CTAC_data.shape}")
    
        data_fold.append({
            "TOFNAC": TOFNAC_data,
            "CTAC": CTAC_data
        })
    
    random.shuffle(data_fold)
    # save the fold
    fold_filename = f"{root_folder}fold_{i_fold}.hdf5"
    with h5py.File(fold_filename, "w") as f:
        for i_case, case in enumerate(data_fold):
            f.create_dataset(f"TOFNAC_{i_case}", data=case["TOFNAC"])
            f.create_dataset(f"CTAC_{i_case}", data=case["CTAC"])

    print(f"Fold {i_fold} saved at {fold_filename}")

print("Done")
