# this script is to combine TOFNAC and CTAC images into a single HDF5 file

import glob
import os
import h5py
import nibabel as nib

root_folder = "./B100/TOFNAC_CTAC_hdf5/"

TOFNAC_folder = "B100/TOFNAC_resample/"
CTAC_folder = "B100/CTACIVV_resample/" # this need to cut file from 467, 467, z to 400, 400, z

TOFNAC_list = sorted(glob.glob(f"{TOFNAC_folder}*.nii.gz"))
CTAC_list = sorted(glob.glob(f"{CTAC_folder}*.nii.gz"))

num_TOFNAC = len(TOFNAC_list)
num_CTAC = len(CTAC_list)

for i in range(num_TOFNAC):
    TOFNAC_path = TOFNAC_list[i]
    CTAC_path = CTAC_list[i]
    print("Processing: ", TOFNAC_path, CTAC_path)