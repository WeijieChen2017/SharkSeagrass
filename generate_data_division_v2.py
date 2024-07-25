import os
import glob
import random

tags_list = sorted(glob.glob("synCT_PET_James/E*.nii.gz"))
tags_list = [os.path.basename(tag)[:5] for tag in tags_list]
print(tags_list)
print("="*40)

# filter out same tags in the list
tags_list = sorted(list(set(tags_list)))

print(tags_list)
print("="*40)

# compute how many files for each tag
n_files = {tag: len(glob.glob(f"synCT_PET_James/{tag}*")) for tag in tags_list}
for tag, n in n_files.items():
    print(f"{tag}: {n} files")

# E4055: 16 files
# E4058: 16 files
# E4061: 16 files
# E4063: 28 files
# E4066: 16 files
# E4068: 16 files
# E4069: 16 files
# E4073: 24 files
# E4074: 16 files
# E4077: 16 files
# E4078: 16 files
# E4079: 16 files
# E4080: 28 files
# E4081: 16 files
# E4084: 16 files
# E4087: 28 files
# E4091: 24 files
# E4092: 16 files
# E4094: 16 files

# Divide the files into 5 chunks, where each chunk contains the similar number of files
n_chunks = 5
n_try = 100
best_equalness = 10000
chunk_division = []

# randomly shuffle the tags and compute the number of files for each tag
for _ in range(n_try):
    random.shuffle(tags_list)
    for i in range(n_chunks):
        chunk_tags = tags_list[i::n_chunks]
        chunk_n_files = sum([n_files[tag] for tag in chunk_tags])
        chunk_division.append((chunk_tags, chunk_n_files))
    # compute the equalness of the number of files in each chunk
    # equalness is the std of the number of files in each chunk
    mean_files = sum([chunk_n_files for _, chunk_n_files in chunk_division]) / n_chunks
    equalness = sum([(chunk_n_files - mean_files)**2 for _, chunk_n_files in chunk_division]) / n_chunks
    if equalness < best_equalness:
        best_equalness = equalness
        best_chunk_division = chunk_division
        print(f"Best equalness: {best_equalness}")
        print(best_chunk_division)
    
# [(['E4091', 'E4068', 'E4058', 'E4092'], 72), (['E4055', 'E4078', 'E4079', 'E4080'], 76), (['E4069', 'E4077', 'E4061', 'E4063'], 76), (['E4073', 'E4066', 'E4081', 'E4094'], 72), (['E
# 4084', 'E4087', 'E4074'], 60)]

chucks_dict = {
    "chunk_0": ['E4091', 'E4068', 'E4058', 'E4092'],
    "chunk_1": ['E4055', 'E4078', 'E4079', 'E4080'],
    "chunk_2": ['E4069', 'E4077', 'E4061', 'E4063'],
    "chunk_3": ['E4073', 'E4066', 'E4081', 'E4094'],
    "chunk_4": ['E4084', 'E4087', 'E4074'],
}

modality_list = ["CT", "PET_raw", "PET_blr", "PET_grd"]

output_dict = {
    "chunk_0": [],
    "chunk_1": [],
    "chunk_2": [],
    "chunk_3": [],
    "chunk_4": [],
}

for chunk_name in chucks_dict.keys():
    for tag in chucks_dict[chunk_name]:
        print(f"---{tag}---")
        file_path_list = sorted(glob.glob(f"synCT_PET_James/{tag}*_s*"))
        for file_path in file_path_list:
            print(file_path)

        sub_tags = []
        for sub_tag_path in file_path_list:
            sub_tag = sub_tag_path.split("1_s")[1].split("e")[0]
            sub_tags.append(sub_tag)
        sub_tags = sorted(list(set(sub_tags)))
        print(sub_tags)
        print("="*40)
        
        for sub_tag in sub_tags:
            loc_s = int(sub_tag)
            loc_e = int(sub_tag) + 256
            loc_e = str(loc_e)
            sub_tag_dict = {
                "CT": f'synCT_PET_James/{tag}_CT_thick_256_norm01_s{sub_tag}e{loc_e}.nii.gz',
                "PET_raw": f'synCT_PET_James/{tag}_PET_thick_256_norm01_s{sub_tag}e{loc_e}.nii.gz',
                "PET_blr": f'synCT_PET_James/{tag}_PET_GauKer_3_norm01_s{sub_tag}e{loc_e}.nii.gz',
                "PET_grd": f'synCT_PET_James/{tag}_PET_GradMag_norm01_s{sub_tag}e{loc_e}.nii.gz',
            }

            # check whether the files exist
            for modality in modality_list:
                if os.path.exists(sub_tag_dict[modality]):
                    print(f"{modality} file exists.")
                else:
                    print(f"{modality} file does not exist.")
                    print(sub_tag_dict[modality])
            print("="*40)
            output_dict[chunk_name].append(sub_tag_dict)
            

# show the output_dict
for chunk_name, sub_tag_dict_list in output_dict.items():
    print(f"---{chunk_name}---")
    for sub_tag_dict in sub_tag_dict_list:
        print(sub_tag_dict)
    print("="*40)
