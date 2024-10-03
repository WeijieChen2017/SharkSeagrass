folder = "results/cv1_256/test_mask/zcut3"
file_list = [
    "HNJ120_axial_z140.nii.gz",
    "HNJ120_axial_z280.nii.gz",
    "HNJ120_axial_z420.nii.gz",
    "HNJ120_coronal_z140.nii.gz",
    "HNJ120_coronal_z280.nii.gz",
    "HNJ120_coronal_z420.nii.gz",
    "HNJ120_sagittal_z140.nii.gz",
    "HNJ120_sagittal_z280.nii.gz",
    "HNJ120_sagittal_z420.nii.gz",
]

other_case = ["KWX131", "LBO118", "NAF069", "NIR103", "RSE114"]

# we load every file, add 1024 to the intensity values, and save the result as name_offset1024.nii.gz

import os
import nibabel as nib

for other_case_name in other_case:

    for old_file in file_list:
        file_path = old_file.replace("HNJ120", other_case_name)
        file_path = os.path.join(folder, file_path)
        nii_file = nib.load(file_path)
        nii_data = nii_file.get_fdata()
        nii_data += 1024
        new_save_name = file_path.split(".")[0] + "_offset1024.nii.gz"
        new_save_path = os.path.join(folder, new_save_name)
        new_nii = nib.Nifti1Image(nii_data, nii_file.affine)
        nib.save(new_nii, new_save_path)
        print(f"Saved {new_save_path}")