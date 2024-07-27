import os
import json
import nibabel as nib

json_file = "synCT_PET_James/3PET1CT.json"

with open(json_file, "r") as f:
    data = json.load(f)

for chunk_name, sub_tag_dict_list in data.items():
    print(f"---{chunk_name}---")
    for sub_tag_dict in sub_tag_dict_list:
        for modality, file_path in sub_tag_dict.items():
            if os.path.exists(file_path):
                data = nib.load(file_path).get_fdata()
                print(f"{file_path} file exists and ", f"shape: {data.shape}")
            else:
                print(f"{modality} file does not exist.")
                print(file_path)
    print("="*40)