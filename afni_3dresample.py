target_folder = "B100/TOFNAC_CTACIVV_part2/"
import os
import json
import glob

data_to_convert = sorted(glob.glob("B100/TOFNAC_CTACIVV_part2/*.nii.gz"))
for path in data_to_convert:
    if not "_400" in path:
        dst_path = path.replace(".nii.gz", "_256.nii.gz")
        print(f"3dresample -dxyz 2.344 2.344 2.344 -prefix {dst_path} -inset {path}")