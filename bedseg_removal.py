# we will find the segmentation map from B100/BedSeg_z366/*.nii.gz
# then extract the z=366 slice from the segmentation map = 1
# and set every slices using the map to 0

import os
import glob
import nibabel as nib

idz = 366
bedseg_path = './B100/BedSeg_z366/'
bedseg_list = sorted(glob.glob(bedseg_path + '*.nii.gz'))

for bedseg_path in bedseg_list:
    print("Processing", bedseg_path)
    case_tag = bedseg_path.split('/')[-1].split('.')[0]
    print("Case tag:", case_tag)