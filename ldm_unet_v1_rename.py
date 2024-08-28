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

for pair in train_list:
    step_1_path = pair["STEP1"]
    step_2_path = pair["STEP2"]
    print("Processing", step_2_path)