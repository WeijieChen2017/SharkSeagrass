# z = 252

# "E4143", 
# "E4144", 
# "E4147", 
# "E4152", 
# "E4155", 
# "E4157", 
# "E4158", 
# "E4162", 
# "E4163", 
# "E4165", 
# "E4166", 
# "E4172", 
# "E4181", 
# "E4182", 
# "E4183", 

import nibabel as nib
import numpy as np
import glob
import os
import random
import string

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MAX_CT = 1976
MIN_CT = -1024
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET

def get_random_name():
    # we need to use combination of letters from A-Z and return a 3 letter string
    return ''.join(random.choices(string.ascii_uppercase, k=3))


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

save_folder = "B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# this is the bed mask for 0/1 masking

data_folder = "B100/TOFNAC_CTACIVV_part2/TC256_part2/"
case_list = sorted(glob.glob(data_folder + "CTACIVV_*_256.nii.gz"))
for case in case_list:
    case_tag = os.path.basename(case).split("_")[1]
    # print(f"\"{case_tag}\", ")
    print("> " * 50)
    TOFNAC_path = "B100/TOFNAC_CTACIVV_part2/TC256_part2/TOFNAC_" + case_tag + "_256.nii.gz"
    CTACIVV_path = case

    TOFNAC_file = nib.load(TOFNAC_path)
    CTACIVV_file = nib.load(CTACIVV_path)

    TOFNAC_data = TOFNAC_file.get_fdata()
    CTACIVV_data = CTACIVV_file.get_fdata()

    TOFNAC_data = TOFNAC_data.astype(np.float32)
    CTACIVV_data = CTACIVV_data.astype(np.float32)# from 299 to 256
    CTACIVV_data = CTACIVV_data[21:277, 21:277, :]
    print(f"TOFNAC: {TOFNAC_data.shape}, CTACIVV: {CTACIVV_data.shape}")
    print(f"TOFNAC MAX: {TOFNAC_data.max()}, MIN: {TOFNAC_data.min()}")
    print(f"CTACIVV MAX: {CTACIVV_data.max()}, MIN: {CTACIVV_data.min()}")
    print(f"TOFNAC MEAN: {TOFNAC_data.mean()}, STD: {TOFNAC_data.std()}")
    print(f"CTACIVV MEAN: {CTACIVV_data.mean()}, STD: {CTACIVV_data.std()}")

    # clip
    TOFNAC_data = np.clip(TOFNAC_data, MIN_PET, MAX_PET)
    CTACIVV_data = np.clip(CTACIVV_data, MIN_CT, MAX_CT)

    TOFNAC_data = two_segment_scale(TOFNAC_data, MIN_CT, MID_PET, MAX_CT, MIQ_PET)
    CTACIVV_data = (CTACIVV_data - MIN_CT) / RANGE_CT

    # masking, the mask part should be set to 0, while the mask is 1
    if case_tag in ["E4143", "E4162", "E4172"]:
        if case_tag == "E4143":
            itksnap_z = 208
        elif case_tag == "E4162":
            itksnap_z = 275
        elif case_tag == "E4172":
            itksnap_z = 250
        mask = nib.load(f"B100/TOFNAC_CTACIVV_part2/bed_{case_tag}_{itksnap_z}.nii.gz")
        mask_data = mask.get_fdata()[21:277, 21:277, itksnap_z-1]
        mask_data = 1 - mask_data
        len_z = TOFNAC_data.shape[2]
        for z in range(len_z):
            CTACIVV_data[:, :, z] = CTACIVV_data[:, :, z] * mask_data

    new_name = get_random_name() + case_tag[2:]
    new_path_TOFNAC = os.path.join(save_folder, f"{new_name}_TOFNAC_256.nii.gz")
    new_path_CTACIVV = os.path.join(save_folder, f"{new_name}_CTAC_256.nii.gz")

    new_file_TOFNAC = nib.Nifti1Image(TOFNAC_data, TOFNAC_file.affine, TOFNAC_file.header)
    new_file_CTACIVV = nib.Nifti1Image(CTACIVV_data, CTACIVV_file.affine, CTACIVV_file.header)

    nib.save(new_file_TOFNAC, new_path_TOFNAC)
    nib.save(new_file_CTACIVV, new_path_CTACIVV)
    print(f"Saved: {new_path_TOFNAC}")
    print(f"Saved: {new_path_CTACIVV}")

    
    





# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 417), CTACIVV: (256, 256, 417)
# TOFNAC MAX: 13942.994140625, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 185.1471710205078, STD: 384.6344909667969
# CTACIVV MEAN: -992.024658203125, STD: 665.6013793945312
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/WCL143_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/WCL143_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 417), CTACIVV: (256, 256, 417)
# TOFNAC MAX: 36275.8359375, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 183.66880798339844, STD: 441.09088134765625
# CTACIVV MEAN: -1017.9144897460938, STD: 643.8919067382812
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/OHC144_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/OHC144_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 417), CTACIVV: (256, 256, 417)
# TOFNAC MAX: 1028378.625, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 226.4475860595703, STD: 1836.38720703125
# CTACIVV MEAN: -971.5745849609375, STD: 675.7304077148438
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/IVZ147_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/IVZ147_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 467), CTACIVV: (256, 256, 467)
# TOFNAC MAX: 23150.88671875, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 246.68521118164062, STD: 426.392822265625
# CTACIVV MEAN: -887.78271484375, STD: 723.3220825195312
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/URO152_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/URO152_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 417), CTACIVV: (256, 256, 417)
# TOFNAC MAX: 254470.875, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 206.53915405273438, STD: 726.3346557617188
# CTACIVV MEAN: -1042.3302001953125, STD: 626.5883178710938
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/LTL155_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/LTL155_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 417), CTACIVV: (256, 256, 417)
# TOFNAC MAX: 10895.5732421875, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 169.56387329101562, STD: 336.9682312011719
# CTACIVV MEAN: -1018.8206176757812, STD: 641.2237548828125
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/LQJ157_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/LQJ157_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 417), CTACIVV: (256, 256, 417)
# TOFNAC MAX: 31409.890625, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 181.64068603515625, STD: 349.7256774902344
# CTACIVV MEAN: -956.3104858398438, STD: 680.5831909179688
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/AFS158_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/AFS158_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 467), CTACIVV: (256, 256, 467)
# TOFNAC MAX: 15498.7919921875, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 210.48785400390625, STD: 417.7614440917969
# CTACIVV MEAN: -934.9859619140625, STD: 700.9740600585938
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/BGZ162_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/BGZ162_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 467), CTACIVV: (256, 256, 467)
# TOFNAC MAX: 21715.28515625, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 203.38047790527344, STD: 408.42193603515625
# CTACIVV MEAN: -942.3536987304688, STD: 694.714599609375
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/YMU163_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/YMU163_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 467), CTACIVV: (256, 256, 467)
# TOFNAC MAX: 23878.109375, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 247.3577423095703, STD: 398.5054016113281
# CTACIVV MEAN: -879.1939086914062, STD: 727.2513427734375
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/FUN165_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/FUN165_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 467), CTACIVV: (256, 256, 467)
# TOFNAC MAX: 54488.55859375, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 181.0684356689453, STD: 515.8857421875
# CTACIVV MEAN: -1003.945556640625, STD: 655.1267700195312
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/UTZ166_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/UTZ166_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 467), CTACIVV: (256, 256, 467)
# TOFNAC MAX: 20981.203125, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 202.47108459472656, STD: 422.7602233886719
# CTACIVV MEAN: -927.8794555664062, STD: 704.1044921875
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/MRL172_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/MRL172_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 467), CTACIVV: (256, 256, 467)
# TOFNAC MAX: 14682.8701171875, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 218.2843780517578, STD: 312.4371337890625
# CTACIVV MEAN: -919.674560546875, STD: 706.789306640625
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/YEG181_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/YEG181_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 467), CTACIVV: (256, 256, 467)
# TOFNAC MAX: 15786.0361328125, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 180.29718017578125, STD: 404.3666687011719
# CTACIVV MEAN: -924.2464599609375, STD: 705.3038940429688
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/PFY182_TOFNAC_256.nii.gz
# Saved: B100/TOFNAC_CTACIVV_part2/TC256_part2_norm/PFY182_CTAC_256.nii.gz
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# TOFNAC: (256, 256, 467), CTACIVV: (256, 256, 467)
# TOFNAC MAX: 18435.865234375, MIN: 0.0
# CTACIVV MAX: 3071.0, MIN: -3024.0
# TOFNAC MEAN: 183.37393188476562, STD: 409.7044372558594
# CTACIVV MEAN: -1002.3021850585938, STD: 657.7523193359375
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
