import os
import nibabel as nib
from scipy.ndimage import binary_fill_holes

tag_list = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 'E4143', 'E4144', 'E4147', 'E4152',
    'E4155', 'E4157', 'E4158', 'E4162', 'E4163',
    'E4165', 'E4166', 'E4172', 'E4181', 'E4182',
    'E4183', 'E4185', 'E4187', 'E4189', 'E4193',
    'E4197', 'E4198', 'E4207', 'E4208', 'E4216',
    'E4217', 'E4219', 'E4220', 'E4232', 'E4237',
    'E4238', 'E4239', 'E4241',
]

HU_boundary_valid_air = -450
HU_boundary_soft = [-450, 150]
HU_boundary_valid_bone = 150

MAX_CT = 2976
MIN_CT = -1024

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET

norm_Boundary_valid_air = (HU_boundary_valid_air - MIN_CT) / RANGE_CT
norm_Boundary_soft = [(HU_boundary_soft[0] - MIN_CT) / RANGE_CT, (HU_boundary_soft[1] - MIN_CT) / RANGE_CT]
norm_Boundary_valid_bone = (HU_boundary_valid_bone - MIN_CT) / RANGE_CT

save_folder = "James_data_v3/mask"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for case_name in tag_list:
    CT_path = f"James_data_v3/CTACIVV_256_norm/CTACIVV_{case_name}_norm.nii.gz"
    CT_file = nib.load(CT_path)
    CT_data = CT_file.get_fdata()

    mask_body_contour = CT_data > norm_Boundary_valid_air
    for i in range(CT_data.shape[0]):
        mask_body_contour[i] = binary_fill_holes(mask_body_contour[i])

    mask_body_contour_nii = nib.Nifti1Image(mask_body_contour, CT_file.affine, CT_file.header)
    nib.save(mask_body_contour_nii, os.path.join(save_folder, f"mask_body_contour_{case_name}.nii.gz"))
    print(f"mask_body_contour_{case_name}.nii.gz saved")

    mask_air = CT_data < norm_Boundary_valid_air
    mask_soft = (CT_data >= norm_Boundary_soft[0]) & (CT_data <= norm_Boundary_soft[1])
    mask_bone = CT_data > norm_Boundary_valid_bone

    # intersection of mask_body_contour and mask_air
    mask_body_air = mask_body_contour & mask_air
    mask_body_air_nii = nib.Nifti1Image(mask_body_air, CT_file.affine, CT_file.header)
    nib.save(mask_body_air_nii, os.path.join(save_folder, f"mask_body_air_{case_name}.nii.gz"))
    print(f"mask_body_air_{case_name}.nii.gz saved")

    # intersection of mask_body_contour and mask_soft
    mask_body_soft = mask_body_contour & mask_soft
    mask_body_soft_nii = nib.Nifti1Image(mask_body_soft, CT_file.affine, CT_file.header)
    nib.save(mask_body_soft_nii, os.path.join(save_folder, f"mask_body_soft_{case_name}.nii.gz"))
    print(f"mask_body_soft_{case_name}.nii.gz saved")

    # intersection of mask_body_contour and mask_bone
    mask_body_bone = mask_body_contour & mask_bone
    mask_body_bone_nii = nib.Nifti1Image(mask_body_bone, CT_file.affine, CT_file.header)
    nib.save(mask_body_bone_nii, os.path.join(save_folder, f"mask_body_bone_{case_name}.nii.gz"))
    print(f"mask_body_bone_{case_name}.nii.gz saved")