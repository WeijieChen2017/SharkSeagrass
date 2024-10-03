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

# we want to replace the command in the template with the other_case names
# command:
# 3dZcutup -prefix HNJ120_axial_z140.nii.gz -keep 0 139 HNJ120_CTAC_pred_axial_rescale_cv1.nii.gz
# 3dZcutup -prefix HNJ120_axial_z280.nii.gz -keep 140 279 HNJ120_CTAC_pred_axial_rescale_cv1.nii.gz
# 3dZcutup -prefix HNJ120_axial_z420.nii.gz -keep 280 419 HNJ120_CTAC_pred_axial_rescale_cv1.nii.gz

# 3dZcutup -prefix HNJ120_coronal_z140.nii.gz -keep 0 139 HNJ120_CTAC_pred_coronal_rescale_cv1.nii.gz
# 3dZcutup -prefix HNJ120_coronal_z280.nii.gz -keep 140 279 HNJ120_CTAC_pred_coronal_rescale_cv1.nii.gz
# 3dZcutup -prefix HNJ120_coronal_z420.nii.gz -keep 280 419 HNJ120_CTAC_pred_coronal_rescale_cv1.nii.gz

# 3dZcutup -prefix HNJ120_sagittal_z140.nii.gz -keep 0 139 HNJ120_CTAC_pred_sagittal_rescale_cv1.nii.gz
# 3dZcutup -prefix HNJ120_sagittal_z280.nii.gz -keep 140 279 HNJ120_CTAC_pred_sagittal_rescale_cv1.nii.gz
# 3dZcutup -prefix HNJ120_sagittal_z420.nii.gz -keep 280 419 HNJ120_CTAC_pred_sagittal_rescale_cv1.nii.gz

# 3dZcutup -prefix HNJ120_axial_z140_mask.nii.gz -keep 0 139 HNJ120_CTAC_pred_axial_mask_cv1.nii.gz
# 3dZcutup -prefix HNJ120_axial_z280_mask.nii.gz -keep 140 279 HNJ120_CTAC_pred_axial_mask_cv1.nii.gz
# 3dZcutup -prefix HNJ120_axial_z420_mask.nii.gz -keep 280 419 HNJ120_CTAC_pred_axial_mask_cv1.nii.gz

# 3dZcutup -prefix HNJ120_coronal_z140_mask.nii.gz -keep 0 139 HNJ120_CTAC_pred_coronal_mask_cv1.nii.gz
# 3dZcutup -prefix HNJ120_coronal_z280_mask.nii.gz -keep 140 279 HNJ120_CTAC_pred_coronal_mask_cv1.nii.gz
# 3dZcutup -prefix HNJ120_coronal_z420_mask.nii.gz -keep 280 419 HNJ120_CTAC_pred_coronal_mask_cv1.nii.gz

# 3dZcutup -prefix HNJ120_sagittal_z140_mask.nii.gz -keep 0 139 HNJ120_CTAC_pred_sagittal_mask_cv1.nii.gz
# 3dZcutup -prefix HNJ120_sagittal_z280_mask.nii.gz -keep 140 279 HNJ120_CTAC_pred_sagittal_mask_cv1.nii.gz
# 3dZcutup -prefix HNJ120_sagittal_z420_mask.nii.gz -keep 280 419 HNJ120_CTAC_pred_sagittal_mask_cv1.nii.gz

# commands_3dZcutup = [
#     "3dZcutup -prefix HNJ120_axial_z140.nii.gz -keep 0 139 HNJ120_CTAC_pred_axial_rescale_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_axial_z280.nii.gz -keep 140 279 HNJ120_CTAC_pred_axial_rescale_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_axial_z420.nii.gz -keep 280 419 HNJ120_CTAC_pred_axial_rescale_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_coronal_z140.nii.gz -keep 0 139 HNJ120_CTAC_pred_coronal_rescale_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_coronal_z280.nii.gz -keep 140 279 HNJ120_CTAC_pred_coronal_rescale_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_coronal_z420.nii.gz -keep 280 419 HNJ120_CTAC_pred_coronal_rescale_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_sagittal_z140.nii.gz -keep 0 139 HNJ120_CTAC_pred_sagittal_rescale_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_sagittal_z280.nii.gz -keep 140 279 HNJ120_CTAC_pred_sagittal_rescale_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_sagittal_z420.nii.gz -keep 280 419 HNJ120_CTAC_pred_sagittal_rescale_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_axial_z140_mask.nii.gz -keep 0 139 HNJ120_CTAC_pred_axial_mask_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_axial_z280_mask.nii.gz -keep 140 279 HNJ120_CTAC_pred_axial_mask_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_axial_z420_mask.nii.gz -keep 280 419 HNJ120_CTAC_pred_axial_mask_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_coronal_z140_mask.nii.gz -keep 0 139 HNJ120_CTAC_pred_coronal_mask_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_coronal_z280_mask.nii.gz -keep 140 279 HNJ120_CTAC_pred_coronal_mask_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_coronal_z420_mask.nii.gz -keep 280 419 HNJ120_CTAC_pred_coronal_mask_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_sagittal_z140_mask.nii.gz -keep 0 139 HNJ120_CTAC_pred_sagittal_mask_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_sagittal_z280_mask.nii.gz -keep 140 279 HNJ120_CTAC_pred_sagittal_mask_cv1.nii.gz",
#     "3dZcutup -prefix HNJ120_sagittal_z420_mask.nii.gz -keep 280 419 HNJ120_CTAC_pred_sagittal_mask_cv1.nii.gz",
# ]

# for other_case_name in other_case:
#     print()
#     for command in commands_3dZcutup:
#         new_command = command.replace("HNJ120", other_case_name)
#         print(new_command)
#     print()

# exit()







# we load every file, add 1024 to the intensity values, and save the result as name_offset1024.nii.gz

# import os
# import nibabel as nib

# for other_case_name in other_case:

#     for old_file in file_list:
#         other_file = old_file.replace("HNJ120", other_case_name)
#         file_path = os.path.join(folder, other_file)
#         nii_file = nib.load(file_path)
#         nii_data = nii_file.get_fdata()
#         nii_data += 1024
#         new_save_name = other_file.split(".")[0] + "_offset1024.nii.gz"
#         new_save_path = os.path.join(folder, new_save_name)
#         new_nii = nib.Nifti1Image(nii_data, nii_file.affine)
#         nib.save(new_nii, new_save_path)
#         print(f"Saved {new_save_path}")


# niftymic_reconstruct_volume \
#   --filenames HNJ120_axial_z140.nii.gz HNJ120_coronal_z140.nii.gz HNJ120_sagittal_z140.nii.gz \
#   --filenames-masks HNJ120_axial_z140_mask.nii.gz HNJ120_coronal_z140_mask.nii.gz HNJ120_sagittal_z140_mask.nii.gz \
#   --output HNJ120_axial_z140_niftyMIC.nii.gz

# niftymic_reconstruct_volume \
#   --filenames HNJ120_axial_z280.nii.gz HNJ120_coronal_z280.nii.gz HNJ120_sagittal_z280.nii.gz \
#   --filenames-masks HNJ120_axial_z280_mask.nii.gz HNJ120_coronal_z280_mask.nii.gz HNJ120_sagittal_z280_mask.nii.gz \
#   --output HNJ120_axial_z280_niftyMIC.nii.gz

# niftymic_reconstruct_volume \
#   --filenames HNJ120_axial_z420.nii.gz HNJ120_coronal_z420.nii.gz HNJ120_sagittal_z420.nii.gz \
#   --filenames-masks HNJ120_axial_z420_mask.nii.gz HNJ120_coronal_z420_mask.nii.gz HNJ120_sagittal_z420_mask.nii.gz \
#   --output HNJ120_axial_z420_niftyMIC.nii.gz

z_list = ["140", "280", "420"]
direction_list = ["axial", "coronal", "sagittal"]

import os

# 420 -> GPU-8
# 280 -> GPU-7
# 140 -> GPU-6

for other_case_name in other_case:

    for z in z_list:
        print()
        filenames = []
        filenames_masks = []
        for direction in direction_list:
            filenames.append(f"{other_case_name}_{direction}_z{z}_offset1024.nii.gz")
            filenames_masks.append(f"{other_case_name}_{direction}_z{z}_mask.nii.gz")
        output = f"{other_case_name}_{direction}_z{z}_niftyMIC_offset1024.nii.gz"
        print(f"niftymic_reconstruct_volume --filenames {' '.join(filenames)} --filenames-masks {' '.join(filenames_masks)} --output {output}")

