import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

resampled_PET_list = sorted(glob.glob("synCT_PET_James/*_PET_thick_256_norm01_s*.nii.gz"))

for path in resampled_PET_list:
    print(path)

    PET_file = nib.load(path)
    PET_data = PET_file.get_fdata()

    PET_data_smooth = gaussian_filter(PET_data, sigma=3)
    PET_data_smooth_filename = path.replace("thick_256", "GauKer_3")
    PET_data_smooth_nii = nib.Nifti1Image(PET_data_smooth, PET_file.affine, PET_file.header)
    nib.save(PET_data_smooth_nii, PET_data_smooth_filename)
    print(f"---Smoothed data saved at {PET_data_smooth_filename}")

    # bounary detection
    # Compute the gradient along each axis
    img_grad = np.gradient(PET_data_smooth)

    # Compute the magnitude of the gradient
    img_grad_abs = np.sqrt(img_grad[0]**2 + img_grad[1]**2 + img_grad[2]**2)
    img_grad_abs_filename = path.replace("thick_256", "GradMag")
    img_grad_abs_nii = nib.Nifti1Image(img_grad_abs, PET_file.affine, PET_file.header)
    nib.save(img_grad_abs_nii, img_grad_abs_filename)
    print(f"---Gradient magnitude data saved at {img_grad_abs_filename}")

