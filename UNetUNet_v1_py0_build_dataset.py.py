# this script is to combine TOFNAC and CTAC images into a single HDF5 file
import glob
import os
# import h5py
import numpy as np
import random
import nibabel as nib

import itertools

# Define the original list of names
names = ['4055', '4069', '4079', '4094', '4105', '4120', '4130', '4139', '4058', '4073', '4081', '4096', '4106', '4124', '4131', '4061', '4074', '4084', '4098', '4114', '4125', '4134', '4066', '4077', '4091', '4099', '4115', '4128', '4137', '4068', '4078', '4092', '4103', '4118', '4129', '4138']

# Create a hashmap (dictionary)
name_map = {}

# Generate three-letter codes (AAA, AAB, AAC, etc.)
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
three_letter_codes = [''.join(i) for i in itertools.product(letters, repeat=3)]

# Shuffle the three-letter codes
import random
random.shuffle(three_letter_codes)

# Iterate through the names and assign codes
for i, name in enumerate(names):
    three_letter_code = three_letter_codes[i]  # Get the next three-letter code
    name_map[name] = f"{three_letter_code}{name[-3:]}"

# Print the hashmap
print("<>"*20)
for name in names:
    print(f"{name} : {name_map[name]}")
print("<>"*20)

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MAX_CT = 2976
MIN_CT = -1024
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET


def two_segment_scale(arr, MIN, MID, MAX, MIQ):
    # Create an empty array to hold the scaled results
    scaled_arr = np.zeros_like(arr, dtype=np.float32)

    # First segment: where arr <= MID
    mask1 = arr <= MID
    scaled_arr[mask1] = (arr[mask1] - MIN) / (MID - MIN) * MIQ

    # Second segment: where arr > MID
    mask2 = arr > MID
    scaled_arr[mask2] = MIQ + (arr[mask2] - MID) / (MAX - MID) * (1 - MIQ)
    
    return scaled_arr

root_folder = "./B100/TOFNAC_CTAC_hash/"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

TOFNAC_folder = "B100/TOFNAC_resample/"
CTAC_folder = "B100/CTACIVV_resample/" # this need to cut file from 467, 467, z to 400, 400, z

TOFNAC_list = sorted(glob.glob(f"{TOFNAC_folder}*.nii.gz"))
CTAC_list = sorted(glob.glob(f"{CTAC_folder}*.nii.gz"))

num_TOFNAC = len(TOFNAC_list)
num_CTAC = len(CTAC_list)

for i_case in range(num_TOFNAC):

    TOFNAC_path = TOFNAC_list[i_case]
    CTAC_path = CTAC_list[i_case]
    print("Processing: ", TOFNAC_path, CTAC_path)

    TOFNAC_file = nib.load(TOFNAC_path)
    CTAC_file = nib.load(CTAC_path)
    TOFNAC_data = TOFNAC_file.get_fdata()
    CTAC_data = CTAC_file.get_fdata()[33:433, 33:433, :]

    # print(f">>> TOFNAC shape: {TOFNAC_data.shape}, CTAC shape: {CTAC_data.shape}")
    # normalize the data
    TOFNAC_data = two_segment_scale(TOFNAC_data, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
    CTAC_data = np.clip(CTAC_data, MIN_CT, MAX_CT)
    CTAC_data = (CTAC_data - MIN_CT) / RANGE_CT

    print(">>>After normalization")
    print(f">>>TOFNAC min: {TOFNAC_data.min():.4f}, TOFNAC max: {TOFNAC_data.max():.4f}")
    print(f">>>CTAC min: {CTAC_data.min():.4f}, CTAC max: {CTAC_data.max():.4f}")
    print(f">>>TOFNAC mean: {TOFNAC_data.mean():.4f}, TOFNAC std: {TOFNAC_data.std():.4f}")
    print(f">>>CTAC mean: {CTAC_data.mean():.4f}, CTAC std: {CTAC_data.std():.4f}")

    # find the hash name
    hash_name = None
    for name in names:
        if name in TOFNAC_path:
            hash_name = name_map[name]
            break
    if hash_name is None:
        raise ValueError(f"Cannot find the hash name for {TOFNAC_path}")
        
    
    # save the data
    save_filename_TOFNAC = f"{root_folder}{hash_name}_TOFNAC.nii.gz"
    save_filename_CTAC = f"{root_folder}{hash_name}_CTAC.nii.gz"

    nii_file_TOFNAC = nib.Nifti1Image(TOFNAC_data, TOFNAC_file.affine, TOFNAC_file.header)
    nii_file_CTAC = nib.Nifti1Image(CTAC_data, CTAC_file.affine, CTAC_file.header)

    nib.save(nii_file_TOFNAC, save_filename_TOFNAC)
    nib.save(nii_file_CTAC, save_filename_CTAC)

    print(f"Saved {hash_name} at {save_filename_TOFNAC} and {save_filename_CTAC}")
    



# fold_0 = ["4055", "4069", "4079", "4094", "4105", "4120", "4130", "4139"]
# fold_1 = ["4058", "4073", "4081", "4096", "4106", "4124", "4131"]
# fold_2 = ["4061", "4074", "4084", "4098", "4114", "4125", "4134"]
# fold_3 = ["4066", "4077", "4091", "4099", "4115", "4128", "4137"]
# fold_4 = ["4068", "4078", "4092", "4103", "4118", "4129", "4138"]

# total_fold = fold_0 + fold_1 + fold_2 + fold_3 + fold_4
# print(len(total_fold))
# print(total_fold)




# n_fold = 5

# # iterate the fold
# for i_fold in range(n_fold):

#     data_fold = []

#     for i_case in range(num_TOFNAC):
#         if not i_case % n_fold == i_fold:
#             continue

#         TOFNAC_path = TOFNAC_list[i_case]
#         CTAC_path = CTAC_list[i_case]
#         print("Processing: ", TOFNAC_path, CTAC_path)

#         TOFNAC_data = nib.load(TOFNAC_path).get_fdata()
#         CTAC_data = nib.load(CTAC_path).get_fdata()[33:433, 33:433, :]

#         print(f">>> TOFNAC shape: {TOFNAC_data.shape}, CTAC shape: {CTAC_data.shape}")
    
#         data_fold.append({
#             "TOFNAC": TOFNAC_data,
#             "CTAC": CTAC_data
#         })
    
#     random.shuffle(data_fold)
#     # save the fold
#     fold_filename = f"{root_folder}fold_{i_fold}.hdf5"
#     with h5py.File(fold_filename, "w") as f:
#         for i_case, case in enumerate(data_fold):
#             f.create_dataset(f"TOFNAC_{i_case}", data=case["TOFNAC"])
#             f.create_dataset(f"CTAC_{i_case}", data=case["CTAC"])

#     print(f"Fold {i_fold} saved at {fold_filename}")

# print("Done")

# fold_0 = ["4055", "4069", "4079", "4094", "4105", "4120", "4130", "4139"]
# fold_1 = ["4058", "4073", "4081", "4096", "4106", "4124", "4131"]
# fold_2 = ["4061", "4074", "4084", "4098", "4114", "4125", "4134"]
# fold_3 = ["4066", "4077", "4091", "4099", "4115", "4128", "4137"]
# fold_4 = ["4068", "4078", "4092", "4103", "4118", "4129", "4138"]

# root@bacf066e60b8:/SharkSeagrass# python build_hdf5_dataset.py
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4055.nii.gz B100/CTACIVV_resample/CTACIVV_4055.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4069.nii.gz B100/CTACIVV_resample/CTACIVV_4069.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4079.nii.gz B100/CTACIVV_resample/CTACIVV_4079.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4094.nii.gz B100/CTACIVV_resample/CTACIVV_4094.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4105.nii.gz B100/CTACIVV_resample/CTACIVV_4105.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4120.nii.gz B100/CTACIVV_resample/CTACIVV_4120.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4130.nii.gz B100/CTACIVV_resample/CTACIVV_4130.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4139.nii.gz B100/CTACIVV_resample/CTACIVV_4139.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Fold 0 saved at ./B100/TOFNAC_CTAC_hdf5/fold_0.hdf5
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4058.nii.gz B100/CTACIVV_resample/CTACIVV_4058.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4073.nii.gz B100/CTACIVV_resample/CTACIVV_4073.nii.gz
# >>> TOFNAC shape: (400, 400, 1123), CTAC shape: (400, 400, 1123)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4081.nii.gz B100/CTACIVV_resample/CTACIVV_4081.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4096.nii.gz B100/CTACIVV_resample/CTACIVV_4096.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4106.nii.gz B100/CTACIVV_resample/CTACIVV_4106.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4124.nii.gz B100/CTACIVV_resample/CTACIVV_4124.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4131.nii.gz B100/CTACIVV_resample/CTACIVV_4131.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Fold 1 saved at ./B100/TOFNAC_CTAC_hdf5/fold_1.hdf5
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4061.nii.gz B100/CTACIVV_resample/CTACIVV_4061.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4074.nii.gz B100/CTACIVV_resample/CTACIVV_4074.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4084.nii.gz B100/CTACIVV_resample/CTACIVV_4084.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4098.nii.gz B100/CTACIVV_resample/CTACIVV_4098.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4114.nii.gz B100/CTACIVV_resample/CTACIVV_4114.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4125.nii.gz B100/CTACIVV_resample/CTACIVV_4125.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4134.nii.gz B100/CTACIVV_resample/CTACIVV_4134.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Fold 2 saved at ./B100/TOFNAC_CTAC_hdf5/fold_2.hdf5
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4066.nii.gz B100/CTACIVV_resample/CTACIVV_4066.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4077.nii.gz B100/CTACIVV_resample/CTACIVV_4077.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4091.nii.gz B100/CTACIVV_resample/CTACIVV_4091.nii.gz
# >>> TOFNAC shape: (400, 400, 1123), CTAC shape: (400, 400, 1123)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4099.nii.gz B100/CTACIVV_resample/CTACIVV_4099.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4115.nii.gz B100/CTACIVV_resample/CTACIVV_4115.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4128.nii.gz B100/CTACIVV_resample/CTACIVV_4128.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4137.nii.gz B100/CTACIVV_resample/CTACIVV_4137.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Fold 3 saved at ./B100/TOFNAC_CTAC_hdf5/fold_3.hdf5
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4068.nii.gz B100/CTACIVV_resample/CTACIVV_4068.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4078.nii.gz B100/CTACIVV_resample/CTACIVV_4078.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4092.nii.gz B100/CTACIVV_resample/CTACIVV_4092.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4103.nii.gz B100/CTACIVV_resample/CTACIVV_4103.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4118.nii.gz B100/CTACIVV_resample/CTACIVV_4118.nii.gz
# >>> TOFNAC shape: (400, 400, 652), CTAC shape: (400, 400, 652)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4129.nii.gz B100/CTACIVV_resample/CTACIVV_4129.nii.gz
# >>> TOFNAC shape: (400, 400, 1201), CTAC shape: (400, 400, 1201)
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4138.nii.gz B100/CTACIVV_resample/CTACIVV_4138.nii.gz
# >>> TOFNAC shape: (400, 400, 730), CTAC shape: (400, 400, 730)
# Fold 4 saved at ./B100/TOFNAC_CTAC_hdf5/fold_4.hdf5
