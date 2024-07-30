import nibabel as nib
import glob
import os

import numpy as np
import scipy.ndimage as nd
from scipy.ndimage import gaussian_filter

th_list = [0.04]

def generate_mask(data, threshold=0.04):
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

def find_bounding_box(mask):
    # Find non-zero indices in the mask
    non_zero_indices = np.argwhere(mask)
    
    # Determine the bounding box for non-zero elements
    x_min, y_min, _ = non_zero_indices.min(axis=0)
    x_max, y_max, _ = non_zero_indices.max(axis=0)

    print("Bounding box: ", x_min, x_max, y_min, y_max)
    
    return x_min, x_max, y_min, y_max




PET_list = sorted(glob.glob("synCT_PET_James/ori/*_PET_re.nii.gz"))
num_files = len(PET_list)

for idx_PET, PET_path in enumerate(PET_list):
        
    print(f"[{idx_PET+1}/{num_files}] Processing {PET_path}")

    PET_file = nib.load(PET_path)
    PET_data = PET_file.get_fdata()

    PET_data = np.clip(PET_data, 0, 4000)
    PET_data = PET_data / 4000
    PET_data_smooth = gaussian_filter(PET_data, sigma=3)
    PET_data_smooth_gradient = np.gradient(PET_data_smooth)
    PET_data_smooth_gradient_magnitude = np.sqrt(PET_data_smooth_gradient[0]**2 + PET_data_smooth_gradient[1]**2 + PET_data_smooth_gradient[2]**2)
    
    # Ker3_filename = PET_path.replace("PET_re", f"PET_GauKer3")
    # Ker3_nii = nib.Nifti1Image(PET_data_smooth, PET_file.affine, PET_file.header)
    # nib.save(Ker3_nii, Ker3_filename)
    # print(f"---Smoothed data saved at {Ker3_filename}")

    # GradMag_filename = PET_path.replace("PET_re", f"PET_GradMag")
    # GradMag_nii = nib.Nifti1Image(PET_data_smooth_gradient_magnitude, PET_file.affine, PET_file.header)
    # nib.save(GradMag_nii, GradMag_filename)
    # print(f"---Gradient magnitude data saved at {GradMag_filename}")
    
    for th_value in th_list:
        
        print("Processing ", PET_path, "at threshold ", th_value)
        PET_mask = generate_mask(PET_data_smooth, threshold=th_value)

        PET_mask_nii = nib.Nifti1Image(PET_mask, PET_file.affine, PET_file.header)
        PET_mask_filename = PET_path.replace("PET_re", f"PET_mask_{int(th_value*100):02d}")
        nib.save(PET_mask_nii, PET_mask_filename)
        print(f"---Mask saved at {PET_mask_filename}")

    CT_path = PET_path.replace("PET_re", "CT_400")
    CT_file = nib.load(CT_path)
    CT_data = CT_file.get_fdata()

    CT_data = np.clip(CT_data, -1024, 2976)
    CT_data = (CT_data + 1024) / 4000

    # find the bounding box of the mask
    x_min, x_max, y_min, y_max = find_bounding_box(PET_mask)
    z_max = PET_data.shape[2]

    # crop the data
    CT_data_crop = CT_data[x_min:x_max, y_min:y_max, :z_max]
    PET_data_crop = PET_data[x_min:x_max, y_min:y_max, :z_max]
    PET_mask_crop = PET_mask[x_min:x_max, y_min:y_max, :z_max]
    PET_data_smooth_crop = PET_data_smooth[x_min:x_max, y_min:y_max, :z_max]
    PET_data_smooth_gradient_magnitude_crop = PET_data_smooth_gradient_magnitude[x_min:x_max, y_min:y_max, :z_max]

    # apply the mask to the cropped data
    CT_data_crop = CT_data_crop * PET_mask_crop
    PET_data_crop = PET_data_crop * PET_mask_crop
    PET_data_smooth_crop = PET_data_smooth_crop * PET_mask_crop
    PET_data_smooth_gradient_magnitude_crop = PET_data_smooth_gradient_magnitude_crop * PET_mask_crop

    # create nifti files
    CT_data_crop_nii = nib.Nifti1Image(CT_data_crop, PET_file.affine, PET_file.header)
    PET_data_crop_nii = nib.Nifti1Image(PET_data_crop, PET_file.affine, PET_file.header)
    PET_mask_crop_nii = nib.Nifti1Image(PET_mask_crop, PET_file.affine, PET_file.header)
    PET_data_smooth_crop_nii = nib.Nifti1Image(PET_data_smooth_crop, PET_file.affine, PET_file.header)
    PET_data_smooth_gradient_magnitude_crop_nii = nib.Nifti1Image(PET_data_smooth_gradient_magnitude_crop, PET_file.affine, PET_file.header)

    # modify the filename
    CT_data_crop_filename = PET_path.replace("PET_re", f"CT_crop_th{int(th_value*100):02d}")
    PET_data_crop_filename = PET_path.replace("PET_re", f"PET_crop_th{int(th_value*100):02d}")
    PET_mask_crop_filename = PET_path.replace("PET_re", f"PET_mask_crop_th{int(th_value*100):02d}")
    PET_data_smooth_crop_filename = PET_path.replace("PET_re", f"PET_GauKer3_crop_th{int(th_value*100):02d}")
    PET_data_smooth_gradient_magnitude_crop_filename = PET_path.replace("PET_re", f"PET_GradMag_crop_th{int(th_value*100):02d}")

    # save the cropped data
    nib.save(CT_data_crop_nii, CT_data_crop_filename)
    nib.save(PET_data_crop_nii, PET_data_crop_filename)
    nib.save(PET_mask_crop_nii, PET_mask_crop_filename)
    nib.save(PET_data_smooth_crop_nii, PET_data_smooth_crop_filename)
    nib.save(PET_data_smooth_gradient_magnitude_crop_nii, PET_data_smooth_gradient_magnitude_crop_filename)

    # print the filename
    print(f"---Cropped CT data saved at {CT_data_crop_filename}")
    print(f"---Cropped PET data saved at {PET_data_crop_filename}")
    print(f"---Cropped PET mask saved at {PET_mask_crop_filename}")
    print(f"---Cropped smoothed PET data saved at {PET_data_smooth_crop_filename}")
    print(f"---Cropped smoothed gradient magnitude PET data saved at {PET_data_smooth_gradient_magnitude_crop_filename}")

