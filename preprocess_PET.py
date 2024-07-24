import os
import glob
import nibabel as nib
from scipy.ndimage import gaussian_filter

resampled_PET_list = sorted(glob.glob("synCT_PET_James/*_PET_thick_256_norm01_s*.nii.gz"))

for path in resampled_PET_list:
    print(path)