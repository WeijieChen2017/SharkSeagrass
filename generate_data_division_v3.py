import json
import os
import glob
import random

tag_list = ["E4055", "E4058", "E4061",          "E4066",
            "E4068", "E4069", "E4073", "E4074", "E4077",
            "E4078", "E4079",          "E4081", "E4084",
                     "E4091", "E4092", "E4094"]

target_dir = f"crop/"
json_filename = f"data_division.json"

# randomly shuffle the tags
random.shuffle(tag_list)

num_chunks = 4

# divide the tags into chunks
chunk_size = len(tag_list) // num_chunks
chunks = [tag_list[i:i+chunk_size] for i in range(0, len(tag_list), chunk_size)]
print(chunks)

def generate_dataset_from_tag(tag, target_dir):
    data_sample_dict = {}
    ct_path = f"{target_dir}{tag}_CT_crop_th04.nii.gz"
    pet_path = f"{target_dir}{tag}_PET_crop_th04.nii.gz"
    pet_mask_path = f"{target_dir}{tag}_PET_mask_crop_th04.nii.gz"
    pet_smooth_path = f"{target_dir}{tag}_PET_GauKer3_crop_th04.nii.gz"
    pet_grad_path = f"{target_dir}{tag}_PET_GradMag_crop_th04.nii.gz"
    data_sample_dict["CT"] = ct_path
    data_sample_dict["PET_raw"] = pet_path
    # data_sample_dict["PET_mask"] = pet_mask_path
    data_sample_dict["PET_blr"] = pet_smooth_path
    data_sample_dict["PET_grd"] = pet_grad_path
    return data_sample_dict

output_json = {
    "chunk_0": [],
    "chunk_1": [],
    "chunk_2": [],
    "chunk_3": [],
}

for idx, chunk in enumerate(chunks):
    for tag in chunk:
        data_sample_dict = generate_dataset_from_tag(tag, target_dir)
        output_json[f"chunk_{idx}"].append(data_sample_dict)

for key in output_json.keys():
    print(key)
    for key2 in output_json[key]:
        print(key2)

with open(target_dir+json_filename, 'w') as f:
    json.dump(output_json, f, indent=4)

print(f"Data division saved to {target_dir+json_filename}")