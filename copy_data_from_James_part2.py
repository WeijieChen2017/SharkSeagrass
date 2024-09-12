case_list = [
    "E4143", "E4144", "E4147",
    "E4152", "E4155", "E4157", "E4158",
    "E4162", "E4163", "E4165", "E4166",
    "E4172",
    "E4181", "E4182", "E4183",
]

import os
import json
import glob
import nibabel as nib
import numpy as np

root_folder = "B100/TOFNAC_CTACIVV_part2/"

for case_tag in case_list:
    TOFNAC_path = sorted(glob.glob(f"Duetto_Output_B100_part2_nii_Winston/PET TOFNAC {case_tag} B100/*.nii"))[0]
    CTACIVV_path = f"Duetto_Output_B100_part2_nii_Winston/CTACIVV_{case_tag[1:]}.nii"

    PET_file = nib.load(TOFNAC_path)
    CT_file = nib.load(CTACIVV_path)

    PET_data = PET_file.get_fdata()
    CT_data = CT_file.get_fdata()

    print("<"*50)
    print(f"File: CTACIVV_{case_tag[1:]}.nii.gz")
    print(f"CT shape: {CT_data.shape}, CT_max: {np.max(CT_data)}, CT_min: {np.min(CT_data)}")
    print(f"CT mean: {np.mean(CT_data)}, CT std: {np.std(CT_data)}")
    print(f"CT 95th percentile: {np.percentile(CT_data, 95)}")
    print(f"CT 99th percentile: {np.percentile(CT_data, 99)}")
    print(f"CT 99.9th percentile: {np.percentile(CT_data, 99.9)}")
    print(f"CT 99.99th percentile: {np.percentile(CT_data, 99.99)}")
    print(f"CT physcial spacing: {CT_file.header.get_zooms()}")
    print(f"CT physical range: {CT_file.header.get_zooms() * np.array(CT_data.shape)}")
    print(">"*50)
    print(f"File: PET_TOFNAC_{case_tag}.nii.gz")
    print(f"PET shape: {PET_data.shape}, PET_max: {np.max(PET_data)}, PET_min: {np.min(PET_data)}")
    print(f"PET mean: {np.mean(PET_data)}, PET std: {np.std(PET_data)}")
    print(f"PET 95th percentile: {np.percentile(PET_data, 95)}")
    print(f"PET 99th percentile: {np.percentile(PET_data, 99)}")
    print(f"PET 99.9th percentile: {np.percentile(PET_data, 99.9)}")
    print(f"PET 99.99th percentile: {np.percentile(PET_data, 99.99)}")
    print(f"PET physcial spacing: {PET_file.header.get_zooms()}")
    print(f"PET physical range: {PET_file.header.get_zooms() * np.array(PET_data.shape)}")
    print("<--->")
    PET_save_path = f"{root_folder}TOFNAC_{case_tag}.nii.gz"
    CT_save_path = f"{root_folder}CTACIVV_{case_tag}.nii.gz"
    nib.save(PET_file, PET_save_path)
    nib.save(CT_file, CT_save_path)
    print(f"Save PET at {PET_save_path}")
    print(f"Save CT at {CT_save_path}")
    print("<--->")

