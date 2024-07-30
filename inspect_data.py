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