# z = 252
import nibabel as nib
import numpy as np
import glob
import os

data_folder = "B100/TOFNAC_CTACIVV_part2/TC256_part2/"
case_list = sorted(glob.glob(data_folder + "CTACIVV_*_256.nii.gz"))
for case in case_list:
    case_tag = os.path.basename(case).split("_")[1]
    print(case_tag)