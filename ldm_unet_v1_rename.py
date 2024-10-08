import numpy as np
import os
import glob
import json

save_folder = "./B100/f4noattn_step1/"
json_path = "./B100/f4noattn_step1_0822_2d3c.json"

os.makedirs(save_folder, exist_ok=True)
# load the json file
with open(json_path, 'r') as file:
    data_division = json.load(file)

train_list = data_division['train']
val_list = data_division['val']
test_list = data_division['test']

print("Train:", len(train_list), "Val:", len(val_list), "Test:", len(test_list))

new_train_list = []
new_val_list = []
new_test_list = []

for pair in train_list:
    step_1_path = pair["STEP1"]
    step_2_path = pair["STEP2"]
    print("Processing", step_2_path)

    new_step_2_path = step_2_path.replace("PET_TOFNAC", "STEP2")
    rename_cmd = "mv "+step_2_path+" "+new_step_2_path
    new_pair = {
        "STEP1": step_1_path,
        "STEP2": new_step_2_path,
    }
    # print(rename_cmd)
    # print(new_pair)
    os.system(rename_cmd)
    new_train_list.append(new_pair)

for pair in val_list:
    step_1_path = pair["STEP1"]
    step_2_path = pair["STEP2"]
    print("Processing", step_2_path)

    new_step_2_path = step_2_path.replace("PET_TOFNAC", "STEP2")
    rename_cmd = "mv "+step_2_path+" "+new_step_2_path
    new_pair = {
        "STEP1": step_1_path,
        "STEP2": new_step_2_path,
    }
    os.system(rename_cmd)
    new_val_list.append(new_pair)

for pair in test_list:
    step_1_path = pair["STEP1"]
    step_2_path = pair["STEP2"]
    print("Processing", step_2_path)

    new_step_2_path = step_2_path.replace("PET_TOFNAC", "STEP2")
    rename_cmd = "mv "+step_2_path+" "+new_step_2_path
    new_pair = {
        "STEP1": step_1_path,
        "STEP2": new_step_2_path,
    }
    os.system(rename_cmd)
    new_test_list.append(new_pair)

new_data_division = {
    "train": new_train_list,
    "val": new_val_list,
    "test": new_test_list,
}

new_json_path = "./B100/f4noattn_step1_0822_2d3c_rename.json"
with open(new_json_path, 'w') as file:
    json.dump(new_data_division, file, indent=4)

print("Done")