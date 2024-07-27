import os
import json
import nibabel as nib
import numpy as np

json_file = "synCT_PET_James/3PET1CT_npy.json"

with open(json_file, "r") as f:
    data = json.load(f)

# save the new npy file in the new json file named "3PET1CT_npy.json"
npy_dict = {}

for chunk_name, sub_tag_dict_list in data.items():
    print(f"---{chunk_name}---")
    npy_dict[chunk_name] = []
    for sub_tag_dict in sub_tag_dict_list:
        chunk_npy_dict = {}
        for modality, file_path in sub_tag_dict.items():
            if os.path.exists(file_path):
                data = np.load(file_path, allow_pickle=True).item()
                print(f"{file_path} file exists and ", f"shape: {data.shape}")
                # data = nib.load(file_path).get_fdata()
                # print(f"{file_path} file exists and ", f"shape: {data.shape}")
                # npy_file_path = file_path.replace(".nii.gz", ".npy")
                # np.save(npy_file_path, data)
                # print(f"---Numpy file saved at {npy_file_path}")
                # chunk_npy_dict[modality] = npy_file_path
            else:
                print(f"{modality} file does not exist.")
                print(file_path)
        npy_dict[chunk_name].append(chunk_npy_dict)
    print("="*40)
    print(npy_dict[chunk_name])
    print("="*40)

# save the npy_dict as a json file
# npy_json_file = json_file.replace(".json", "_npy.json")
# with open(npy_json_file, "w") as json_file:
#     json.dump(npy_dict, json_file, indent=4)
#     print(f"---Numpy files saved at {npy_json_file}")