import nibabel as nib
import glob
import os

import numpy as np
import scipy.ndimage as nd
from scipy.ndimage import gaussian_filter

th_list = [0.05, 0.04, 0.06]

def generate_mask(data, threshold=0.05):
    mask_data = np.zeros_like(data)
    mask_data[data >= threshold] = 1

    radius_dilation = 15  # Example radius
    # Create a spherical structuring element
    struct = nd.generate_binary_structure(3, 1) 
    struct = nd.iterate_structure(struct, radius_dilation)

    # # Perform 3D dilation
    mask_data = nd.binary_dilation(mask_data, structure=struct)

    # Create a structuring element for x-y plane erosion with radius 7
    radius_erosion = 15
    struct_xy = np.zeros((radius_erosion*2+1, radius_erosion*2+1, 3))  # Initialize a larger array to accommodate the radius
    struct_xy[:, :, 1] = nd.iterate_structure(nd.generate_binary_structure(2, 1), radius_erosion)  # Set the middle z-slice to a 2D structuring element with radius 7

    # Perform x-y plane erosion
    mask_data = nd.binary_erosion(mask_data, structure=struct_xy)

    return mask_data

PET_list = sorted(glob.glob("synCT_PET_James/ori/E4079_PET_re.nii.gz"))

for th_value in th_list:
    for PET_path in PET_list:
        
        print("Processing ", PET_path, "at threshold ", th_value)
        PET_file = nib.load(PET_path)
        PET_data = PET_file.get_fdata()

        PET_data = np.clip(PET_data, 0, 4000)
        PET_data = PET_data / 4000
        PET_data_smooth = gaussian_filter(PET_data, sigma=3)
        PET_data_smooth_gradient = np.gradient(PET_data_smooth)
        PET_data_smooth_gradient_magnitude = np.sqrt(PET_data_smooth_gradient[0]**2 + PET_data_smooth_gradient[1]**2 + PET_data_smooth_gradient[2]**2)
        
        Ker3_filename = PET_path.replace("PET_re", f"PET_GauKer3")
        Ker3_nii = nib.Nifti1Image(PET_data_smooth, PET_file.affine, PET_file.header)
        nib.save(Ker3_nii, Ker3_filename)
        print(f"---Smoothed data saved at {Ker3_filename}")

        GradMag_filename = PET_path.replace("PET_re", f"PET_GradMag")
        GradMag_nii = nib.Nifti1Image(PET_data_smooth_gradient_magnitude, PET_file.affine, PET_file.header)
        nib.save(GradMag_nii, GradMag_filename)
        print(f"---Gradient magnitude data saved at {GradMag_filename}")


        PET_mask = generate_mask(PET_data_smooth, threshold=th_value)

        PET_mask_nii = nib.Nifti1Image(PET_mask, PET_file.affine, PET_file.header)
        PET_mask_filename = PET_path.replace("PET_re", f"PET_mask_{th_value*100:02d}")
        nib.save(PET_mask_nii, PET_mask_filename)
        print(f"---Mask saved at {PET_mask_filename}")
