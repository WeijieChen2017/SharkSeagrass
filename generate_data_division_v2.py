import os
import glob

tags_list = sorted(glob.glob("synCT_PET_James/ori/E*.nii.gz"))
tags_list = [os.path.basename(tag)[:5] for tag in tags_list]
print(tags_list)
print("="*40)

# filter out same tags in the list
tags_list = sorted(list(set(tags_list)))

print(tags_list)
print("="*40)