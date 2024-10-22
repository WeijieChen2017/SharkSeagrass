import os
import glob

TOFNAC_dir = "James_data_v3/Duetto_Output_B100_nii/"
CTACIVV_dir = "James_data_v3/part3/"

TOFNAC_dir_list =sorted(glob.glob(TOFNAC_dir + "*/*.nii"))
for TOFNAC_path in TOFNAC_dir_list:
    case_name = TOFNAC_path.split("/")[-2].split(".")[0]
    print(case_name)