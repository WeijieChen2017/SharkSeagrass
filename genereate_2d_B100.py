import os
import json
import nibabel as nib
import numpy as np

json_path = "./B100/B100_0822.json"

with open(json_path, "r") as f:
    data = json.load(f)

train_pair_list = data["train"]
val_pair_list = data["val"]
test_pair_list = data["test"]

print(f"Train: {len(train_pair_list)}")
print(f"Val: {len(val_pair_list)}")
print(f"Test: {len(test_pair_list)}")

data_folder = "./B100/B100_0822_2d3c/"
if not os.path.exists(data_folder):
    os.makedirs(data_folder, exist_ok=True)

new_train_pair_list = []
new_val_pair_list = []
new_test_pair_list = []
pair_list = [
    (train_pair_list, new_train_pair_list),
    (val_pair_list, new_val_pair_list),
    (test_pair_list, new_test_pair_list),
]

for pair_list, new_pair in pair_list:
    for pair in pair_list:
        PET_path = pair["PET"]
        CT_path = pair["CT"]
        PET_name = os.path.basename(PET_path)
        CT_name = os.path.basename(CT_path)
        PET_file = nib.load(PET_path)
        CT_file = nib.load(CT_path)

        PET_data = PET_file.get_fdata()
        CT_data = CT_file.get_fdata()

        for i in range(PET_data.shape[2]):
            if i == 0:
                index_list = [0, 0, i]
            elif i == PET_data.shape[2]:
                index_list = [i-1, i, i]
            else:
                index_list = [i-1, i, i+1]

            PET_slice = PET_data[:, :, index_list]
            CT_slice = CT_data[:, :, i]

            PET_slice_path = os.path.join(data_folder, f"{PET_name[:-7]}_z{i}.npy")
            CT_slice_path = os.path.join(data_folder, f"{CT_name[:-7]}_z{i}.npy")

            np.save(PET_slice_path, PET_slice)
            np.save(CT_slice_path, CT_slice)
            print(f"Saving [{i}/{PET_data.shape[2]}] PET: {PET_slice_path}, CT: {CT_slice_path}")

            new_pair.append({"PET": PET_slice_path, "CT": CT_slice_path})

with open(json_path.replace(".json", "_2d3c.json"), "w") as f:
    json.dump({"train": new_train_pair_list,
               "val": new_val_pair_list,
               "test": new_test_pair_list
               }, f, indent=4)

print("Done!")