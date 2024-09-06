import json
import glob
import os

import nibabel as nib
import matplotlib

data_div_json = "./B100/step1step2_0822_vanila.json"
with open(data_div_json, "r") as f:
    data_div = json.load(f)

save_folder = "./B100/plot_test_case_UNetUNet/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

train_list = data_div["train"]
val_list = data_div["val"]
test_list = data_div["test"]

num_train = len(train_list)
num_val = len(val_list)
num_test = len(test_list)

print(f"num_train: {num_train}")
print(f"num_val: {num_val}")
print(f"num_test: {num_test}")

axial_cut = 8
sagittal_cut = 4
coronal_cut = 4

# plot the test case
for test_pair in test_list:
    print()
    x_path = test_pair["STEP1"] # "STEP1": "./B100/f4noattn_step1_volume/STEP1_E4078.nii.gz",
    x_path = x_path.replace("f4noattn_step1_volume_vanila", "TOFNAC_resample")
    x_path = x_path.replace("STEP1", "PET_TOFNAC")
    y_path = test_pair["STEP2"] # "STEP2": "./B100/f4noattn_step2_volume/STEP2_E4078.nii.gz",
    z_path = test_pair["STEP1"].replace("STEP1", "STEP3_d3f64")
    case_name = y_path[-12:-7]
    print(f"Processing test case: {case_name}")
    print(f">>> TOFNAC_path: {x_path}")
    print(f">>> CTAC_path: {y_path}")
    print(f">>> PRED_path: {z_path}")

    x_data = nib.load(x_path).get_fdata()
    y_data = nib.load(y_path).get_fdata()
    z_data = nib.load(z_path).get_fdata()

    print(f">>> TOFNAC_shape: {x_data.shape}, CTAC_shape: {y_data.shape}, PRED_shape: {z_data.shape}")


    # for axial:
    save_name = f"{save_folder}{case_name}_axial_cut_{axial_cut}.png"
    print(f">>> Saving to {save_name}")

    # for sagittal:
    save_name = f"{save_folder}{case_name}_sagittal_cut_{sagittal_cut}.png"
    print(f">>> Saving to {save_name}")

    # for coronal:
    save_name = f"{save_folder}{case_name}_coronal_cut_{coronal_cut}.png"
    print(f">>> Saving to {save_name}")