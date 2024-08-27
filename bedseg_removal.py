# we will find the segmentation map from B100/BedSeg_z366/*.nii.gz
# then extract the z=366 slice from the segmentation map = 1
# and set every slices using the map to 0

import os
import glob
import numpy as np
import nibabel as nib

idz = 366 # this is in the ITK-SNAP coordinate system
bedseg_path = './B100/BedSeg_z366/'
bedseg_list = sorted(glob.glob(bedseg_path + '*.nii.gz'))
CT_raw_folder = './B100/nifti/raw/'
os.makedirs(CT_raw_folder, exist_ok=True)

for bedseg_path in bedseg_list:

    case_tag = bedseg_path.split('/')[-1].split('.')[0][-5:]
    CT_path = "./B100/nifti/CTACIVV_"+case_tag+".nii.gz"
    print("Processing", bedseg_path, "and", CT_path)
    
    bedseg_file = nib.load(bedseg_path)
    CT_file = nib.load(CT_path)

    bedseg_data = bedseg_file.get_fdata()[:, :, idz-1]
    CT_data = CT_file.get_fdata()
    print("CT shape:", CT_data.shape)
    print("BedSeg shape:", mask.shape)

    # extract the mask
    mask = bedseg_data == 1
    # show the mask size
    print("Mask size:", np.sum(mask))

    for i in range(CT_data.shape[2]):
        img = CT_data[:, :, i]
        img[mask] = 0
        CT_data[:, :, i] = img
    
    new_CT_file = nib.Nifti1Image(CT_data, affine=CT_file.affine, header=CT_file.header)
    # mv the original CT file to raw folder
    os.system("mv "+CT_path+" "+CT_raw_folder)
    nib.save(new_CT_file, CT_path)
    print("Processed", CT_path)
