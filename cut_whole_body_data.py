import numpy as np
import nibabel as nib
import os
import glob

tags_list = sorted(glob.glob("synCT_PET_James/ori/*_re.nii.gz"))
tags_list = [os.path.basename(tag)[:5] for tag in tags_list]
print(tags_list)
print("="*40)

# filter out same tags in the list
tags_list = list(set(tags_list))

print(tags_list)
print("="*40)

Q_list_PET = [3000, 4000, 5000, 6000]
Q_list_CT = [1000, 2000, 3000, 4000]

def calculate_percentiles(data, q_list):
    # Calculate the percentile values for each value in q_list
    # percentiles = [np.percentile(data, (np.sum(data <= q) / data.size) * 100) for q in q_list]
    percentiles = [(np.sum(data < q) / data.size) * 100 for q in q_list]
    return percentiles

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
    # output 99%, 99.9% , 99.99% percentile for PET and CT
    print("99% percentile PET: ", np.percentile(PET_data, 99), "99% percentile CT: ", np.percentile(CT_data, 99))
    print("99.9% percentile PET: ", np.percentile(PET_data, 99.9), "99.9% percentile CT: ", np.percentile(CT_data, 99.9))
    print("99.99% percentile PET: ", np.percentile(PET_data, 99.99), "99.99% percentile CT: ", np.percentile(CT_data, 99.99))
    # output the percentile Q_list_PET and Q_list_CT
    # it is where the value in Q_list_PET is the percentage of the PET_data
    # for example, if Q_list_PET = 3000, then the value is the 75% percentile of PET_data
    for idx_Q in range(len(Q_list_PET)):
        pQ_PET = calculate_percentiles(PET_data, Q_list_PET)[idx_Q]
        pQ_CT = calculate_percentiles(CT_data, Q_list_CT)[idx_Q]
        print(f"{pQ_PET:4f}% of PET: {Q_list_PET[idx_Q]}, {pQ_CT:4f}% of CT: {Q_list_CT[idx_Q]}")

    print("="*40)

    # # original_CT is 467*467*730
    # # this should be cropped to 400*400*730
    CT_data_crop = CT_data[33:433, 33:433, :]
    print(CT_data_crop.shape)

    # # save the cropped CT data
    CT_data_crop_nii = nib.Nifti1Image(CT_data_crop, PET_file.affine, PET_file.header)
    savename = f"synCT_PET_James/ori/{tag}_CT_400.nii.gz"
    nib.save(CT_data_crop_nii, savename)
    print("Cropped CT data saved at ", savename)