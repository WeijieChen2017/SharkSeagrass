import os
import json

json_file = "synCT_PET_James/3PET1CT.json"

with open(json_file, "r") as f:
    data = json.load(f)

for chunk_name, sub_tag_dict_list in data.items():
    print(f"---{chunk_name}---")
    for sub_tag_dict in sub_tag_dict_list:
        for modality, file_path in sub_tag_dict.items():
            if os.path.exists(file_path):
                print(f"{modality} file exists.")
            else:
                print(f"{modality} file does not exist.")
                print(file_path)
    print("="*40)