import os
import glob

TOFNAC_dir = "James_data_v3/Duetto_Output_B100_nii/"
CTACIVV_dir = "James_data_v3/part3/"

TOFNAC_dir_list =sorted(glob.glob(TOFNAC_dir + "*/*.nii"))
for TOFNAC_path in TOFNAC_dir_list:
    print()
    case_name = TOFNAC_path.split("/")[-2].split(".")[0]
    # PET TOFNAC E4237 B100
    case_name = case_name.split(" ")[-2]
    CTACIVV_path = f"{CTACIVV_dir}CTACIVV_{case_name[1:]}.nii"
    print(case_name)
    move_cmd_TOFNAC = f"mv {TOFNAC_path} James_data_v3/TOFNAC_{case_name}.nii"
    move_cmd_CTACIVV = f"mv {CTACIVV_path} James_data_v3/CTACIVV_{case_name}.nii"
    print(move_cmd_TOFNAC)
    print(move_cmd_CTACIVV)
    os.system(move_cmd_TOFNAC)
    os.system(move_cmd_CTACIVV)