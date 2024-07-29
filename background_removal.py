import nibabel as nib
import glob
import os

import numpy as np
import scipy.ndimage as nd
from scipy.ndimage import gaussian_filter

def generate_mask(data, threshold=0.0001):
    mask_data = np.zeros_like(data)
    mask_data[data >= threshold] = 1

    radius_dilation = 15  # Example radius
    # Create a spherical structuring element
    struct = nd.generate_binary_structure(3, 1) 
    struct = nd.iterate_structure(struct, radius_dilation)

    # # Perform 3D dilation
    mask_data = nd.binary_dilation(mask_data, structure=struct)

    # Create a structuring element for x-y plane erosion with radius 7
    radius_erosion = 7
    struct_xy = np.zeros((radius_erosion*2+1, radius_erosion*2+1, 3))  # Initialize a larger array to accommodate the radius
    struct_xy[:, :, 1] = nd.iterate_structure(nd.generate_binary_structure(2, 1), radius_erosion)  # Set the middle z-slice to a 2D structuring element with radius 7

    # Perform x-y plane erosion
    mask_data = nd.binary_erosion(mask_data, structure=struct_xy)

    return mask_data



PET_file = "synCT_PET_James/ori/E4079_CT_400.nii.gz"
PET_file = nib.load(PET_file)
PET_data = PET_file.get_fdata()

PET_data = np.clip(PET_data, 0, 4000)
PET_data = PET_data / 4000
PET_data_smooth = gaussian_filter(PET_data, sigma=3)

Ker3_filename = "synCT_PET_James/ori/E4079_CT_400_GauKer_3.nii.gz"
Ker3_nii = nib.Nifti1Image(PET_data_smooth, PET_file.affine, PET_file.header)
nib.save(Ker3_nii, Ker3_filename)
print(f"---Smoothed data saved at {Ker3_filename}")

PET_mask = generate_mask(PET_data_smooth)

PET_mask_nii = nib.Nifti1Image(PET_mask, PET_file.affine, PET_file.header)
nib.save(PET_mask_nii, "synCT_PET_James/ori/E4079_CT_400_mask.nii.gz")
print(f"---Mask saved at synCT_PET_James/ori/E4079_CT_400_mask.nii.gz")



