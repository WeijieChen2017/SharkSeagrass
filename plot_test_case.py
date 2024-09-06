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
    x_path = test_pair["STEP1"]
    y_path = test_pair["STEP2"]
    z_path = test_pair["STEP1"].replace("STEP1", "STEP3_f364")
    print(f"Processing test case: {z_path}")
    