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
    for idz in range(nii_data.shape[2]):
        img = nii_data[:, :, idz]
        npy_path = save_folder + "STEP2_"+tag+"_z"+str(idz)+".npy"
        np_data = np.load(npy_path)
        if img.shape == np_data.shape:
            np.save(npy_path, img)
            print("Processed", npy_path)
        else:
            print("Shape mismatch:", img.shape, np_data.shape)