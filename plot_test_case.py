import json
import glob
import os

import nibabel as nib
import matplotlib

data_div_json = "./B100/step1step2_0822_vanila.json"
with open(data_div_json, "r") as f:
    data_div = json.load(f)

train_list = data_div["train"]
val_list = data_div["val"]
test_list = data_div["test"]

num_train = len(train_list)
num_val = len(val_list)
num_test = len(test_list)

print(f"num_train: {num_train}")
print(f"num_val: {num_val}")
print(f"num_test: {num_test}")

# plot the test case
for test_pair in test_list:
    print()
    x_path = test_pair["STEP1"] # "STEP1": "./B100/f4noattn_step1_volume/STEP1_E4078.nii.gz",
    x_path = x_path.replace("f4noattn_step1_volume_vanila", "TOFNAC_resample")
    x_path = x_path.replace("STEP1", "PET_TOFNAC")
    y_path = test_pair["STEP2"] # "STEP2": "./B100/f4noattn_step2_volume/STEP2_E4078.nii.gz",
    z_path = test_pair["STEP1"].replace("STEP1", "STEP3_f364")
    print(f"Processing test case:")
    print(f">>> TOFNAC_path: {x_path}")
    print(f">>> CTAC_path: {y_path}")
    print(f">>> PRED_path: {z_path}")

    x_data = nib.load(x_path).get_fdata()
    y_data = nib.load(y_path).get_fdata()
    z_data = nib.load(z_path).get_fdata()

    print(f">>> TOFNAC_shape: {x_data.shape}, CTAC_shape: {y_data.shape}, PRED_shape: {z_data.shape}")
