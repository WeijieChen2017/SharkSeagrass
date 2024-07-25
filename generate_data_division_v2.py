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
best_equalness = 100
chunk_division = []

# randomly shuffle the tags and compute the number of files for each tag
for _ in range(n_try):
    print(f"Try {_}")
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
    