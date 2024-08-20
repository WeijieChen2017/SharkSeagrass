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

# here we load the data
import os
import nibabel as nib
import numpy as np

for tag in tag_list:
    CT_path = f"./B100/CTACIVV_resample/CTACIVV_{tag[1:]}.nii.gz"
    PET_path = f"./B100/TOFNAC_resample/PET_TOFNAC_{tag}.nii.gz"
    CT_file = nib.load(CT_path)
    PET_file = nib.load(PET_path)
    CT_data = CT_file.get_fdata()
    PET_data = PET_file.get_fdata()

    print("<"*50)
    print(f"File: CTACIVV_{tag[1:]}.nii.gz")
    print(f"CT shape: {CT_data.shape}, CT_max: {np.max(CT_data)}, CT_min: {np.min(CT_data)}")
    print(f"CT mean: {np.mean(CT_data):.4f}, CT std: {np.std(CT_data):.4f}")
    print(f"CT 95th, 99th percentile: {np.percentile(CT_data, 95):.4f} {np.percentile(CT_data, 99):.4f}")
    print(f"CT 99.9th, 99.99th percentile: {np.percentile(CT_data, 99.9):.4f}, {np.percentile(CT_data, 99.99):.4f}")
    print(f"CT physcial spacing: {CT_file.header.get_zooms():.4f}, range: {CT_file.header.get_zooms() * np.array(CT_data.shape):.4f}")
    print(">"*50)
    print(f"File: PET_TOFNAC_{tag}.nii.gz")
    print(f"PET shape: {PET_data.shape}, PET_max: {np.max(PET_data)}, PET_min: {np.min(PET_data)}")
    print(f"PET mean: {np.mean(PET_data):.4f}, PET std: {np.std(PET_data):.4f}")
    print(f"PET 95th, 99th percentile: {np.percentile(PET_data, 95):.4f} {np.percentile(PET_data, 99):.4f}")
    print(f"PET 99.9th, 99.99th percentile: {np.percentile(PET_data, 99.9):.4f}, {np.percentile(PET_data, 99.99):.4f}")
    print(f"PET physcial spacing: {PET_file.header.get_zooms():.4f}, range: {PET_file.header.get_zooms() * np.array(PET_data.shape):.4f}")
    print("--"*25)
