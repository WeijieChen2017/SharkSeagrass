import os
import glob

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
n_files_per_chunk = sum(n_files.values()) // n_chunks
print(f"Each chunk will contain {n_files_per_chunk} files")

# Divide the files into chunks
chunks = []
chunk = []
chunk_size = 0
for tag, n in n_files.items():
    if chunk_size + n <= n_files_per_chunk:
        chunk.append(tag)
        chunk_size += n
    else:
        chunks.append(chunk)
        chunk = [tag]
        chunk_size = n
chunks.append(chunk)

# Print the chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}", "number of files:", sum([n_files[tag] for tag in chunk]))

