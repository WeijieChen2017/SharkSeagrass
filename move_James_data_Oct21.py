import os
import glob

TOFNAC_dir = "James_data_v3/Duetto_Output_B100_nii/"
CTACIVV_dir = "James_data_v3/part3/"

TOFNAC_dir_list =sorted(glob.glob(TOFNAC_dir + "*/*.nii"))
for TOFNAC_path in TOFNAC_dir_list:
    case_name = TOFNAC_path.split("/")[-2].split(".")[0]
    # PET TOFNAC E4237 B100
    case_name = case_name.split(" ")[-2]
    CTACIVV_path = f"{CTACIVV_dir}CTACIVV_{case_name[1:]}.nii.gz"
    print(case_name)
    copy_cmd_TOFNAC = f"cp {TOFNAC_path} James_data_v3/TOFNAC_{case_name}.nii"
    copy_cmd_CTACIVV = f"cp {CTACIVV_path} James_data_v3/CTACIVV_{case_name}.nii"
    print(copy_cmd_TOFNAC)
    print(copy_cmd_CTACIVV)
    os.system(copy_cmd_TOFNAC)
    os.system(copy_cmd_CTACIVV)