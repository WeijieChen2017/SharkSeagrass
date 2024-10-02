# niftymic_reconstruct_volume \
#   --filenames HNJ120_axial_z140.nii.gz HNJ120_coronal_z140.nii.gz HNJ120_sagittal_z140.nii.gz \
#   --filenames-masks HNJ120_axial_z140_mask.nii.gz HNJ120_coronal_z140_mask.nii.gz HNJ120_sagittal_z140_mask.nii.gz \
#   --output HNJ120_axial_z140_niftyMIC.nii.gz

# niftymic_reconstruct_volume \
#   --filenames HNJ120_axial_z280.nii.gz HNJ120_coronal_z280.nii.gz HNJ120_sagittal_z280.nii.gz \
#   --filenames-masks HNJ120_axial_z280_mask.nii.gz HNJ120_coronal_z280_mask.nii.gz HNJ120_sagittal_z280_mask.nii.gz \
#   --output HNJ120_axial_z280_niftyMIC.nii.gz

# niftymic_reconstruct_volume \
#   --filenames HNJ120_axial_z420.nii.gz HNJ120_coronal_z420.nii.gz HNJ120_sagittal_z420.nii.gz \
#   --filenames-masks HNJ120_axial_z420_mask.nii.gz HNJ120_coronal_z420_mask.nii.gz HNJ120_sagittal_z420_mask.nii.gz \
#   --output HNJ120_axial_z420_niftyMIC.nii.gz

z_list = ["140", "280", "420"]
direction_list = ["axial", "coronal", "sagittal"]

import os

for z in z_list:
    print()
    filenames = []
    filenames_masks = []
    for direction in direction_list:
        filenames.append(f"HNJ120_{direction}_z{z}_offset1024.nii.gz")
        filenames_masks.append(f"HNJ120_{direction}_z{z}_mask.nii.gz")
    output = f"HNJ120_{direction}_z{z}_niftyMIC_offset1024.nii.gz"
    print(f"niftymic_reconstruct_volume --filenames {' '.join(filenames)} --filenames-masks {' '.join(filenames_masks)} --output {output}")

