
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


# randomly shuflle them and set the train:val:test ratio to 70:20:10

import random

random.seed(42)
random.shuffle(tag_list)

train_list = tag_list[:int(len(tag_list) * 0.7)]
val_list = tag_list[int(len(tag_list) * 0.7):int(len(tag_list) * 0.9)]
test_list = tag_list[int(len(tag_list) * 0.9):]

train_pair_list = [
    {"PET": f"./B100/nifti/PET_TOFNAC_{case_tag}.nii.gz", 
     "CT": f"./B100/nifti/CTACIVV_{case_tag}.nii.gz"} for case_tag in train_list
]

val_pair_list = [
    {"PET": f"./B100/nifti/PET_TOFNAC_{case_tag}.nii.gz", 
     "CT": f"./B100/nifti/CTACIVV_{case_tag}.nii.gz"} for case_tag in val_list
]

test_pair_list = [
    {"PET": f"./B100/nifti/PET_TOFNAC_{case_tag}.nii.gz", 
     "CT": f"./B100/nifti/CTACIVV_{case_tag}.nii.gz"} for case_tag in test_list
]

import json

with open("./B100/B100_0822.json", "w") as f:
    json.dump({"train": train_pair_list,
               "val": val_pair_list,
               "test": test_pair_list
               }, f, indent=4)
    
print("Done!")
