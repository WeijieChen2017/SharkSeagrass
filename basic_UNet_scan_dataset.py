import json

dataset_path = "synCT_PET_James/3PET1CT.json"
with open(dataset_path, "r") as json_file:
    dataset = json.load(json_file)

print(dataset)
for chunk_name, sub_tag_dict_list in dataset.items():
    print(f"---{chunk_name}---")
    for sub_tag_dict in sub_tag_dict_list:
        print(sub_tag_dict)
        for image_modality, image_path in sub_tag_dict.items():
            print(f"---{image_modality}---")
            print(image_path)
    print("="*40)