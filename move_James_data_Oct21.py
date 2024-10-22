import os
import glob

# Part 1
# ----> rename all the files in the James_data_v3 directory

# TOFNAC_dir = "James_data_v3/_nii_Winston/"
# CTACIVV_dir = "James_data_v3/part3/"

# TOFNAC_dir_list =sorted(glob.glob(TOFNAC_dir + "*/*.nii"))
# for TOFNAC_path in TOFNAC_dir_list:
#     print()
#     case_name = TOFNAC_path.split("/")[-2].split(".")[0]
#     # PET TOFNAC E4237 B100
#     case_name = case_name.split(" ")[-2]
#     CTACIVV_path = f"{CTACIVV_dir}CTACIVV_{case_name[1:]}.nii"
#     print(case_name)
#     move_cmd_TOFNAC = f"mv \"{TOFNAC_path}\" James_data_v3/TOFNAC_{case_name}.nii"
#     move_cmd_CTACIVV = f"mv {CTACIVV_path} James_data_v3/CTACIVV_{case_name}.nii"
#     print(move_cmd_TOFNAC)
#     print(move_cmd_CTACIVV)
#     os.system(move_cmd_TOFNAC)
#     os.system(move_cmd_CTACIVV)

# Part 2
# ----> 3d resample to 2.344 mm isotropic

TOFNAC_dir = "James_data_v3/TOFNAC/"
CTACIVV_dir = "James_data_v3/CTACIVV/"

TOFNAC_dir_list =sorted(glob.glob(TOFNAC_dir + "*.nii"))
CTACIVV_dir_list =sorted(glob.glob(CTACIVV_dir + "*.nii"))

for TOFNAC_path in TOFNAC_dir_list:
    dst_path = TOFNAC_path.replace(".nii", "_256.nii")
    print(f"3dresample -dxyz 2.344 2.344 2.344 -prefix {dst_path} -inset {TOFNAC_path}")

for CTACIVV_path in CTACIVV_dir_list:
    dst_path = CTACIVV_path.replace(".nii", "_256.nii")
    print(f"3dresample -dxyz 2.344 2.344 2.344 -prefix {dst_path} -inset {CTACIVV_path}")

    