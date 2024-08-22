tag_list = [
    "E4055", "E4058", "E4061",          "E4066",
    "E4068", "E4069", "E4073", "E4074", "E4077",
    "E4078", "E4079",          "E4081", "E4084",
             "E4091", "E4092", "E4094", "E4096",
             "E4098", "E4099",          "E4103",
    "E4105", "E4106", "E4114", "E4115", "E4118",
    "E4120", "E4124", "E4125", "E4128", "E4129",
    "E4130", "E4131", "E4134", "E4137", "E4138",
    "E4139",
]

import os
import numpy as np
import nibabel as nib

for tag in tag_list:
    print()
    CT_nifti_path = f"./B100/nifti/CTACIVV_{tag}.nii.gz"
    PET_nifti_path = f"./B100/nifti/PET_TOFNAC_{tag}.nii.gz"
    CT_nifti_file = nib.load(CT_nifti_path)
    PET_nifti_file = nib.load(PET_nifti_path)
    print(f"Loaded nifti files {CT_nifti_path} and {PET_nifti_path}")
    for VQ_NAME in ["f4-noattn", "f8-n256"]:
        save_folder = f"./B100/vq_{VQ_NAME}_recon_nifti"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        CT_npy_path = f"./B100/vq_{VQ_NAME}_recon/vq_{VQ_NAME}_{tag}_CTr_recon.npy"
        PET_npy_path = f"./B100/vq_{VQ_NAME}_recon/vq_{VQ_NAME}_{tag}_PET_recon.npy"
        CT_npy = np.load(CT_npy_path)
        PET_npy = np.load(PET_npy_path)
        print(f"Loaded npy files {CT_npy_path} and {PET_npy_path}")
        print(f"CT_npy.shape: {CT_npy.shape}, PET_npy.shape: {PET_npy.shape}")
        CT_npy_data = CT_npy
        PET_npy_data = PET_npy
        # reshape the data from z, x, y to x, y, z
        CT_npy_data = np.transpose(CT_npy_data, (1, 2, 0))
        PET_npy_data = np.transpose(PET_npy_data, (1, 2, 0))
        print(f"CT_npy_data.shape: {CT_npy_data.shape}, PET_npy_data.shape: {PET_npy_data.shape}")
        CT_npy_nifti = nib.Nifti1Image(CT_npy_data, CT_nifti_file.affine, CT_nifti_file.header)
        PET_npy_nifti = nib.Nifti1Image(PET_npy_data, PET_nifti_file.affine, PET_nifti_file.header)
        CT_npy_nifti_path = f"{save_folder}/vq_{VQ_NAME}_{tag}_CTr_recon.nii.gz"
        PET_npy_nifti_path = f"{save_folder}/vq_{VQ_NAME}_{tag}_PET_recon.nii.gz"
        nib.save(CT_npy_nifti, CT_npy_nifti_path)
        nib.save(PET_npy_nifti, PET_npy_nifti_path)
        print(f"Saved nifti files {CT_npy_nifti_path} and {PET_npy_nifti_path}")

