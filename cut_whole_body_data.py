import numpy as np
import nibabel as nib
import os
import glob

tags_list = sorted(glob.glob("synCT_PET_James/ori/*_re.nii.gz"))
tags_list = [os.path.basename(tag)[:5] for tag in tags_list]
print(tags_list)

# filter out same tags in the list
tags_list = list(set(tags_list))

print(tags_list)

for tag in tags_list:

    PET_file_path = f"synCT_PET_James/ori/{tag}_PET_re.nii.gz"
    CT_file_path = f"synCT_PET_James/ori/{tag}_CT_re.nii.gz"

    PET_file = nib.load(PET_file_path)
    CT_file = nib.load(CT_file_path)

    PET_data = PET_file.get_fdata()
    CT_data = CT_file.get_fdata()

    print(tag)
    print("PET data shape: ", PET_data.shape, "CT data shape: ", CT_data.shape)
    print("PET data max: ", np.max(PET_data), "PET data min: ", np.min(PET_data))
    print("CT data max: ", np.max(CT_data), "CT data min: ", np.min(CT_data))
    print("="*40)

    # # original_CT is 467*467*730
    # # this should be cropped to 400*400*730
    # CT_data_crop = CT_data[33:433, 33:433, :]
    # print(CT_data_crop.shape)

    # # save the cropped CT data
    # CT_data_crop_nii = nib.Nifti1Image(CT_data_crop, PET_file.affine, PET_file.header)
    # nib.save(CT_data_crop_nii, "synCT_PET_James/ori/E4055_CT_re_crop_PET_meta.nii.gz")
    # print("Cropped CT data saved.")