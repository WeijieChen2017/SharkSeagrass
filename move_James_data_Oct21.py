import os
import glob
import numpy as np
import nibabel as nib

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

# TOFNAC_dir = "James_data_v3/TOFNAC/"
# CTACIVV_dir = "James_data_v3/CTACIVV/"

# TOFNAC_dir_list =sorted(glob.glob(TOFNAC_dir + "*.nii"))
# CTACIVV_dir_list =sorted(glob.glob(CTACIVV_dir + "*.nii"))

# for TOFNAC_path in TOFNAC_dir_list:
#     dst_path = TOFNAC_path.replace(".nii", "_256.nii")
#     print(f"3dresample -dxyz 2.344 2.344 2.344 -prefix {dst_path} -inset {TOFNAC_path}")

# for CTACIVV_path in CTACIVV_dir_list:
#     dst_path = CTACIVV_path.replace(".nii", "_256.nii")
#     print(f"3dresample -dxyz 2.344 2.344 2.344 -prefix {dst_path} -inset {CTACIVV_path}")

# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4055_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4055.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4058_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4058.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4061_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4061.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4063_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4063.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4066_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4066.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4068_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4068.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4069_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4069.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4073_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4073.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4074_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4074.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4077_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4077.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4078_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4078.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4079_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4079.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4080_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4080.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4081_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4081.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4084_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4084.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4087_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4087.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4091_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4091.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4092_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4092.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4094_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4094.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4096_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4096.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4097_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4097.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4098_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4098.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4099_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4099.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4102_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4102.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4103_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4103.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4105_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4105.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4106_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4106.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4114_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4114.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4115_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4115.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4118_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4118.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4120_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4120.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4124_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4124.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4125_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4125.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4128_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4128.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4129_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4129.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4130_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4130.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4131_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4131.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4134_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4134.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4137_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4137.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4138_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4138.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4139_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4139.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4143_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4143.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4144_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4144.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4147_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4147.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4152_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4152.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4155_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4155.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4157_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4157.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4158_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4158.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4162_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4162.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4163_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4163.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4165_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4165.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4166_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4166.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4172_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4172.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4181_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4181.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4182_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4182.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4183_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4183.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4185_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4185.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4187_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4187.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4189_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4189.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4193_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4193.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4197_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4197.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4198_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4198.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4203_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4203.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4204_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4204.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4207_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4207.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4208_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4208.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4216_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4216.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4217_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4217.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4219_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4219.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4220_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4220.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4225_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4225.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4232_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4232.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4237_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4237.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4238_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4238.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4239_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4239.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/TOFNAC/TOFNAC_E4241_256.nii -inset James_data_v3/TOFNAC/TOFNAC_E4241.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4055_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4055.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4058_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4058.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4061_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4061.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4063_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4063.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4066_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4066.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4068_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4068.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4069_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4069.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4073_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4073.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4074_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4074.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4077_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4077.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4078_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4078.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4079_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4079.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4080_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4080.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4081_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4081.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4084_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4084.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4087_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4087.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4091_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4091.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4092_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4092.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4094_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4094.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4096_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4096.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4097_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4097.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4098_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4098.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4099_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4099.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4102_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4102.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4103_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4103.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4105_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4105.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4106_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4106.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4114_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4114.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4115_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4115.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4118_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4118.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4120_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4120.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4124_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4124.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4125_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4125.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4128_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4128.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4129_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4129.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4130_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4130.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4131_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4131.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4134_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4134.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4137_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4137.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4138_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4138.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4139_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4139.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4143_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4143.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4144_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4144.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4147_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4147.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4152_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4152.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4155_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4155.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4157_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4157.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4158_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4158.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4162_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4162.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4163_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4163.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4165_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4165.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4166_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4166.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4172_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4172.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4181_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4181.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4182_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4182.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4183_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4183.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4185_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4185.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4187_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4187.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4189_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4189.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4193_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4193.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4197_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4197.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4198_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4198.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4203_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4203.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4204_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4204.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4207_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4207.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4208_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4208.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4216_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4216.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4217_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4217.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4219_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4219.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4220_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4220.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4225_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4225.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4232_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4232.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4237_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4237.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4238_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4238.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4239_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4239.nii
# 3dresample -dxyz 2.344 2.344 2.344 -prefix James_data_v3/CTACIVV/CTACIVV_E4241_256.nii -inset James_data_v3/CTACIVV/CTACIVV_E4241.nii

# Part 3
# ----> check the data for dim, min and max

TOFNAC_dir = "James_data_v3/TOFNAC_256/"
CTACIVV_dir = "James_data_v3/CTACIVV_256/"

TOFNAC_path_lists = sorted(glob.glob(TOFNAC_dir + "*.nii"))
casename_list = []

for TOFNAC_path in TOFNAC_path_lists:
    casename = TOFNAC_path.split("/")[-1].split(".")[0]
    # TOFNAC_E4063_256.nii
    casename = casename.split("_")[1]
    casename_list.append(casename)
    CTACIVV_path = CTACIVV_dir + "CTACIVV_" + casename + "_256.nii"

    TOFNAC_file = nib.load(TOFNAC_path)
    CTACIVV_file = nib.load(CTACIVV_path)

    TOFNAC_data = TOFNAC_file.get_fdata()
    CTACIVV_data = CTACIVV_file.get_fdata()

    print(f"case: {casename}, DIM TOFNAC: {TOFNAC_data.shape}, CTACIVV: {CTACIVV_data.shape}")
    print(f"case: {casename}, MIN TOFNAC: {TOFNAC_data.min()}, CTACIVV: {CTACIVV_data.min()}")
    print(f"case: {casename}, MAX TOFNAC: {TOFNAC_data.max()}, CTACIVV: {CTACIVV_data.max()}")

    TOFNAC_file_new = nib.Nifti1Image(TOFNAC_data, TOFNAC_file.affine, TOFNAC_file.header)
    CTACIVV_file_new = nib.Nifti1Image(CTACIVV_data, CTACIVV_file.affine, CTACIVV_file.header)

    nib.save(TOFNAC_file_new, TOFNAC_dir + "TOFNAC_" + casename + "_256.nii.gz")
    nib.save(CTACIVV_file_new, CTACIVV_dir + "CTACIVV_" + casename + "_256.nii.gz")

    print(f"case: {casename}, saved!")