import nibabel as nib
import glob
import os

tag_list = ['E4091', 'E4068', 'E4058', 'E4092', 'E4055', 
            'E4078', 'E4079', 'E4080', 'E4069', 'E4077',
            'E4061', 'E4063', 'E4073', 'E4066', 'E4081',
            'E4094', 'E4084', 'E4087', 'E4074']

tag_list = sorted(tag_list)

PET_list = [f"synCT_PET_James/ori/{tag}_PET.nii.gz" for tag in tag_list]
CT_list = [f"synCT_PET_James/ori/{tag}_CT.nii.gz" for tag in tag_list]

# for PET_path, CT_path in zip(PET_list, CT_list):
#     PET_file = nib.load(PET_path)
#     PET_data = PET_file.get_fdata()

#     CT_file = nib.load(CT_path)
#     CT_data = CT_file.get_fdata()

#     print("-"*40)
#     print(PET_path)
#     print("PET data shape: ", PET_data.shape)
#     pix_dim = PET_file.header.get_zooms()
#     print("Pixel dimension: ", pix_dim)
#     # show the physical dimension
#     print("Physical dimension: ", [pix_dim[i]*PET_data.shape[i] for i in range(3)])

#     print(CT_path)
#     print("CT data shape: ", CT_data.shape)
#     pix_dim = CT_file.header.get_zooms()
#     print("Pixel dimension: ", pix_dim)
#     # show the physical dimension
#     print("Physical dimension: ", [pix_dim[i]*CT_data.shape[i] for i in range(3)])
#     print("-"*40)

for PET_path, CT_path in zip(PET_list, CT_list):
    cmd_PET = f"cp {PET_path} synCT_PET_James/ori/raw/"
    cmd_CT = f"cp {CT_path} synCT_PET_James/ori/raw/"

    os.system(cmd_PET)
    os.system(cmd_CT)
    print(f"---{PET_path} and {CT_path} copied to synCT_PET_James/ori/raw/")



# root@aee6fdcd2fb2:/Ammongus/SharkSeagrass# python inspect_data.py
# ----------------------------------------
# synCT_PET_James/ori/E4055_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4055_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4058_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4058_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4061_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4061_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4063_PET.nii.gz
# PET data shape:  (256, 256, 587)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1919.4899888038635]
# synCT_PET_James/ori/E4063_CT.nii.gz
# CT data shape:  (512, 512, 582)
# Pixel dimension:  (1.367188, 1.367188, 3.2699966)
# Physical dimension:  [700.000244140625, 700.000244140625, 1903.1380462646484]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4066_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4066_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4068_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4068_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4069_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4069_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4073_PET.nii.gz
# PET data shape:  (256, 256, 515)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1684.0499901771545]
# synCT_PET_James/ori/E4073_CT.nii.gz
# CT data shape:  (512, 512, 515)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1684.0499901771545]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4074_PET.nii.gz
# PET data shape:  (256, 256, 299)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 977.7299942970276]
# synCT_PET_James/ori/E4074_CT.nii.gz
# CT data shape:  (512, 512, 299)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 977.7299942970276]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4077_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4077_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4078_PET.nii.gz
# PET data shape:  (256, 256, 299)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 977.7299942970276]
# synCT_PET_James/ori/E4078_CT.nii.gz
# CT data shape:  (512, 512, 299)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 977.7299942970276]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4079_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4079_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4080_PET.nii.gz
# PET data shape:  (256, 256, 623)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 2037.209988117218]
# synCT_PET_James/ori/E4080_CT.nii.gz
# CT data shape:  (512, 512, 606)
# Pixel dimension:  (1.367188, 1.367188, 5.0)
# Physical dimension:  [700.000244140625, 700.000244140625, 3030.0]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4081_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4081_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4084_PET.nii.gz
# PET data shape:  (256, 256, 299)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 977.7299942970276]
# synCT_PET_James/ori/E4084_CT.nii.gz
# CT data shape:  (512, 512, 299)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 977.7299942970276]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4087_PET.nii.gz
# PET data shape:  (256, 256, 587)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1919.4899888038635]
# synCT_PET_James/ori/E4087_CT.nii.gz
# CT data shape:  (512, 512, 582)
# Pixel dimension:  (1.367188, 1.367188, 5.0)
# Physical dimension:  [700.000244140625, 700.000244140625, 2910.0]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4091_PET.nii.gz
# PET data shape:  (256, 256, 515)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1684.0499901771545]
# synCT_PET_James/ori/E4091_CT.nii.gz
# CT data shape:  (512, 512, 515)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1684.0499901771545]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4092_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4092_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# ----------------------------------------
# synCT_PET_James/ori/E4094_PET.nii.gz
# PET data shape:  (256, 256, 335)
# Pixel dimension:  (2.34375, 2.34375, 3.27)
# Physical dimension:  [600.0, 600.0, 1095.449993610382]
# synCT_PET_James/ori/E4094_CT.nii.gz
# CT data shape:  (512, 512, 335)
# Pixel dimension:  (1.367188, 1.367188, 3.27)
# Physical dimension:  [700.000244140625, 700.000244140625, 1095.449993610382]
# ----------------------------------------
# root@aee6fdcd2fb2:/Ammongus/SharkSeagrass# 
