import nibabel as nib
import numpy as np
import glob
import os
import json

data_folder = "tsv1_ct/"
data_list = sorted(glob.glob(data_folder + "*.nii.gz"))
print("Total data cases:", len(data_list))
dataset_over128 = []
dataset_over96 = []
dataset_over64 = []

for path in data_list:
    print("<"*18)
    print("Processing", path)
    nii_file = nib.load(path)
    nii_data = nii_file.get_fdata()
    print("Data shape:", nii_data.shape)
    if nii_data.shape[0] > 128:
        dataset_over128.append(path)
    if nii_data.shape[0] > 96:
        dataset_over96.append(path)
    if nii_data.shape[0] > 64:
        dataset_over64.append(path)

print("Over 128:", len(dataset_over128))
print("Over 96:", len(dataset_over96))
print("Over 64:", len(dataset_over64))

json_over128_list = [{"STEP1": path, "STEP2": path,} for path in dataset_over128]
json_over96_list = [{"STEP1": path, "STEP2": path,} for path in dataset_over96]
json_over64_list = [{"STEP1": path, "STEP2": path,} for path in dataset_over64]

with open("tsv1_ct_over128.json", 'w') as file:
    json.dump(json_over128_list, file)
with open("tsv1_ct_over96.json", 'w') as file:
    json.dump(json_over96_list, file)
with open("tsv1_ct_over64.json", 'w') as file:
    json.dump(json_over64_list, file)

print("Saved json files.")