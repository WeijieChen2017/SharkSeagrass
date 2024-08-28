import numpy as np
import nibabel as nib
import os
import glob
import json

save_folder = "./B100/f4noattn_step1/"

target_tags = ["E4061", "E4098", "E4114", "E4128", "E4139"]

for tag in target_tags:
    nii_path = "./B100/nifti/CTACIVV_"+tag+".nii.gz"
    nii_file = nib.load(nii_path)
    nii_data = nii_file.get_fdata()
    print("Processing", nii_path)
    print("Data shape:", nii_data.shape)
    for idz in range(nii_data.shape[2]):
        img = nii_data[:, :, idz]
        npy_path = save_folder + "STEP2_"+tag+"_"+str(idz)+".npy"
        np.load(npy_path)
        print("The img shape is", img.shape, "and the npy shape is", np.load(npy_path).shape)