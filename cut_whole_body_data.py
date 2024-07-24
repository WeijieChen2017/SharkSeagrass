import numpy as np
import nibabel as nib
import os
import glob

PET_file_path = "synCT_PET_James/ori/E4055_PET_re.nii.gz"
CT_file_path = "synCT_PET_James/ori/E4055_CT_re.nii.gz"

PET_file = nib.load(PET_file_path)
CT_file = nib.load(CT_file_path)

PET_data = PET_file.get_fdata()
CT_data = CT_file.get_fdata()

# original_CT is 467*467*730
# this should be cropped to 400*400*730
CT_data_crop = CT_data[33:433, 33:433, :]
print(CT_data_crop.shape)

# save the cropped CT data
CT_data_crop_nii = nib.Nifti1Image(CT_data_crop, CT_file.header, CT_file.affine)
nib.save(CT_data_crop_nii, "synCT_PET_James/ori/E4055_CT_re_crop.nii.gz")
print("Cropped CT data saved.")