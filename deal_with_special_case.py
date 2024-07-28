import nibabel as nib
import os
import numpy as np

CT_path = "synCT_PET_James/ori/E4063_CT_400.nii.gz"
PET_path = "synCT_PET_James/ori/E4063_PET_re.nii.gz"

CT_file = nib.load(CT_path)
CT_data = CT_file.get_fdata()

PET_file = nib.load(PET_path)
PET_data = PET_file.get_fdata() 

print("---CT data---")
print(CT_data.shape)
cut_data_CT = CT_data[:, :, 1013:1269]
print(cut_data_CT.shape)

print("---PET data---")
print(PET_data.shape)
cut_data_PET = PET_data[:, :, 1013:1269]
print(cut_data_PET.shape)

cut_CT_nii = nib.Nifti1Image(cut_data_CT, PET_file.affine, PET_file.header)
new_CT_nii = "synCT_PET_James/E4063_CT_thick_256_norm01_s1013e1269.nii.gz"
print(f"---Cropped data saved at {new_CT_nii}")
nib.save(cut_CT_nii, new_CT_nii)

cut_PET_nii = nib.Nifti1Image(cut_data_PET, PET_file.affine, PET_file.header)
new_PET_nii = "synCT_PET_James/E4063_PET_thick_256_norm01_s1013e1269.nii.gz"
print(f"---Cropped data saved at {new_PET_nii}")
nib.save(cut_PET_nii, new_PET_nii)