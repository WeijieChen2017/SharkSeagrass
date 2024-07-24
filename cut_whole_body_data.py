import numpy as np
import nibabel as nib
import os
import glob

tags_list = sorted(glob.glob("synCT_PET_James/ori/*_re.nii.gz"))
tags_list = [os.path.basename(tag)[:5] for tag in tags_list]
print(tags_list)
print("="*40)

# filter out same tags in the list
tags_list = sorted(list(set(tags_list)))

print(tags_list)
print("="*40)

Q_list = [90, 99, 99.9, 99.99]
Q_list_PET = [3000, 4000, 5000, 6000]
Q_list_CT = [1000, 2000, 3000, 4000]

thickness_pixel = 256
overlap = 64

def calculate_percentiles(data, q_list):
    # Calculate the percentile values for each value in q_list
    # percentiles = [np.percentile(data, (np.sum(data <= q) / data.size) * 100) for q in q_list]
    flatten_data = data.flatten()
    percentiles = [(np.sum(flatten_data < q) / flatten_data.size) * 100 for q in q_list]
    return percentiles

def save_data_at_z1_to_z2(data_1, data_2, loc_z, thickness_pixel, tag, affine_1, header_1, affine_2, header_2):

    z1 = loc_z
    z2 = loc_z + thickness_pixel

    data_1_z = data_1[:, :, z1:z2]
    data_2_z = data_2[:, :, z1:z2]

    print("Cropped data from ", z1, " to ", z2)

    data_1_z_nii = nib.Nifti1Image(data_1_z, affine_1, header_1)
    data_2_z_nii = nib.Nifti1Image(data_2_z, affine_2, header_2)

    savename_1 = f"synCT_PET_James/crop/{tag}_PET_thick_256_s{z1:03d}e{z2:03d}.nii.gz"
    savename_2 = f"synCT_PET_James/crop/{tag}_CT_thick_256_s{z1:03d}e{z2:03d}.nii.gz"

    nib.save(data_1_z_nii, savename_1)
    nib.save(data_2_z_nii, savename_2)

    print(f"---Cropped data saved at {savename_1}")
    print(f"---Cropped data saved at {savename_2}")

    # PET_data_crop = PET_data_clip[:, :, loc_z:loc_z+thickness_pixel]
    # CT_data_crop = CT_data_clip[:, :, loc_z:loc_z+thickness_pixel]
    # print("Cropped PET data from ", loc_z, " to ", loc_z+thickness_pixel)
    # print("Cropped CT data from ", loc_z, " to ", loc_z+thickness_pixel)

    # PET_data_crop_nii = nib.Nifti1Image(PET_data_crop, PET_file.affine, PET_file.header)
    # CT_data_crop_nii = nib.Nifti1Image(CT_data_crop, CT_file.affine, CT_file.header)

    # savename_PET = f"synCT_PET_James/crop/{tag}_PET_{loc_z}.nii.gz"
    # savename_CT = f"synCT_PET_James/crop/{tag}_CT_{loc_z}.nii.gz"

    # nib.save(PET_data_crop_nii, savename_PET)
    # nib.save(CT_data_crop_nii, savename_CT)

    # print(f"---Cropped PET data saved at {savename_PET}")
    # print(f"---Cropped CT data saved at {savename_CT}")


for tag in tags_list[:2]:

    PET_file_path = f"synCT_PET_James/ori/{tag}_PET_re.nii.gz"
    CT_file_path = f"synCT_PET_James/ori/{tag}_CT_400.nii.gz"

    PET_file = nib.load(PET_file_path)
    CT_file = nib.load(CT_file_path)

    PET_data = PET_file.get_fdata()
    CT_data = CT_file.get_fdata()

    len_z = PET_data.shape[2]

    print("["*6, tag, "]"*6)
    # print("PET data shape: ", PET_data.shape, "CT data shape: ", CT_data.shape)
    # print(f"PET data max: {np.max(PET_data):.4f}, PET data min: {np.min(PET_data):.4f}")
    # print(f"CT data max: {np.max(CT_data):.4f}, CT data min: {np.min(CT_data):.4f}")
    # # output 99%, 99.9% , 99.99% percentile for PET and CT
    # for q in Q_list:
    #     p_PET = np.percentile(PET_data.flatten(), q)
    #     p_CT = np.percentile(CT_data.flatten(), q)
    #     print(f">{q:.2f}% percentile of PET: {p_PET:.4f}, {q:.2f}% percentile of CT: {p_CT:.4f}")
    # # output the percentile Q_list_PET and Q_list_CT
    # # it is where the value in Q_list_PET is the percentage of the PET_data
    # # for example, if Q_list_PET = 3000, then the value is the 75% percentile of PET_data
    # pQ_PET = calculate_percentiles(PET_data, Q_list_PET)
    # pQ_CT = calculate_percentiles(CT_data, Q_list_CT)
    # for idx in range(len(Q_list_PET)):
    #     print(f"<{pQ_PET[idx]:.4f}% of PET: {Q_list_PET[idx]}, {pQ_CT[idx]:.4f}% of CT: {Q_list_CT[idx]}")

    

    # # original_CT is 467*467*730
    # # this should be cropped to 400*400*730
    # CT_data_crop = CT_data[33:433, 33:433, :]
    # print("New CT data shape: ", CT_data_crop.shape)

    # # # save the cropped CT data
    # CT_data_crop_nii = nib.Nifti1Image(CT_data_crop, PET_file.affine, PET_file.header)
    # savename = f"synCT_PET_James/ori/{tag}_CT_400.nii.gz"
    # nib.save(CT_data_crop_nii, savename)
    # print("---Cropped CT data saved at ", savename)

    PET_data_clip = np.clip(PET_data, 0, 4000)
    CT_data_clip = np.clip(CT_data, -1024, 2976)

    # start with 0 to thickness_pixel, then move thickness_pixel - overlap
    # for example, 0 to 200, then 100 to 300, then 200 to 400
    loc_z = 0
    while loc_z + thickness_pixel <= len_z:
        save_data_at_z1_to_z2(PET_data_clip, 
                              CT_data_clip, 
                              loc_z, 
                              thickness_pixel, 
                              tag, 
                              PET_file.affine, 
                              PET_file.header, 
                              CT_file.affine, 
                              CT_file.header)

        loc_z += thickness_pixel - overlap

    # last one is from len_z - thickness_pixel to len_z
    save_data_at_z1_to_z2(PET_data_clip, 
                          CT_data_clip, 
                          len_z - thickness_pixel, 
                          thickness_pixel, 
                          tag, 
                          PET_file.affine, 
                          PET_file.header, 
                          CT_file.affine, 
                          CT_file.header)

    print("="*40)