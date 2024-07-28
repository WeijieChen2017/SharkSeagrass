import nibabel as nib
import os
import numpy as np

CT_path = "synCT_PET_James/ori/E4063_CT_400.nii.gz"

CT_file = nib.load(CT_path)
CT_data = CT_file.get_fdata()

print("---CT data---")
print(CT_data.shape)
cut_data = CT_data[:, :, 1024:1280]

print(cut_data.shape)

cut_data_nii = nib.Nifti1Image(cut_data, CT_file.affine, CT_file.header)

new_filename_nii = "synCT_PET_James/E4063_CT_thick_256_norm01_s1024e1280.nii.gz"
# nib.save(cut_data_nii, new_filename_nii)
print(f"---Cropped data saved at {new_filename_nii}")

new_filename_npy = "synCT_PET_James/E4063_CT_thick_256_norm01_s1024e1280.npy"
# np.save(new_filename_npy, cut_data)
print(f"---Numpy file saved at {new_filename_npy}")

# double check by loading two files and output the shape
CT_file = nib.load(new_filename_nii)
CT_data = CT_file.get_fdata()
print(f"---{new_filename_nii}---")
print(CT_data.shape)

CT_data = np.load(new_filename_npy)
print(f"---{new_filename_npy}---")
print(CT_data.shape)

