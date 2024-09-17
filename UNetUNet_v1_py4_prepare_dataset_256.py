import nibabel as nib
import numpy as np
import argparse
import h5py
import json
import os

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MAX_CT = 1976
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

def printlog(log_file, message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

def main():
    argparser = argparse.ArgumentParser(description='Prepare dataset for training')
    argparser.add_argument('-c', '--cross_validation', type=int, default=0, help='Index of the cross validation')
    args = argparser.parse_args()
    root_folder = "B100/UNetUnet_best/"

    data_div_json = "UNetUNet_v1_data_split.json"
    with open(data_div_json, "r") as f:
        data_div = json.load(f)
    
    for cv in [0,1,2,3,4]:

        # cv = args.cross_validation
        train_list = data_div[f"cv_{cv}"]["train"]
        val_list = data_div[f"cv_{cv}"]["val"]
        test_list = data_div[f"cv_{cv}"]["test"]
        log_file = f"{root_folder}UNetUNet_v1_cv{cv}_log.txt"

        # num_train = len(train_list)
        # num_val = len(val_list)
        # num_test = len(test_list)

        str_train_list = ", ".join(train_list)
        str_val_list = ", ".join(val_list)
        str_test_list = ", ".join(test_list)

        # construct the data path list
        train_path_list = []
        val_path_list = []
        test_path_list = []

        for hashname in train_list:
            train_path_list.append({
                # BPO124_CTAC_pred_cv0.nii.gz
                # "STEP1": f"B100/UNetUnet_best/cv{cv}/train/{hashname}_CTAC_pred_cv{cv}.nii.gz",
                "STEP1": f"{root_folder}cv{cv}_256/train/{hashname}_CTAC_pred_cv{cv}.nii.gz",
                # BPO124_CTAC.nii.gz
                "STEP2": f"{root_folder}CTAC/{hashname}_CTAC.nii.gz",
            })

        for hashname in val_list:
            val_path_list.append({
                "STEP1": f"{root_folder}cv{cv}_256/val/{hashname}_CTAC_pred_cv{cv}.nii.gz",
                "STEP2": f"{root_folder}CTAC/{hashname}_CTAC.nii.gz",
            })

        for hashname in test_list:
            test_path_list.append({
                "STEP1": f"{root_folder}cv{cv}_256/test/{hashname}_CTAC_pred_cv{cv}.nii.gz",
                "STEP2": f"{root_folder}CTAC/{hashname}_CTAC.nii.gz",
            })


        # train
        for idx_list, path_list in enumerate([train_path_list, val_path_list, test_path_list]):
            
            printlog(log_file, f"Processing cv{cv} data in split {idx_list}")


            for pair in path_list:
                
                printlog(log_file, "<" * 50)

                step1_path = pair["STEP1"]
                step2_path = pair["STEP2"]

                step1_file = nib.load(step1_path)
                step2_file = nib.load(step2_path)

                step1_data = step1_file.get_fdata()
                step2_data = step2_file.get_fdata()

                compress_step1_data = np.clip(step1_data, 0, 1).astype(np.float32)
                compress_step1_path = step1_path.replace(".nii.gz", "_clip.nii.gz")
                compress_step1_path = step1_path.replace("_256", "_256_clip")
                compress_step1_nii = nib.Nifti1Image(compress_step1_data, step1_file.affine, step1_file.header)
                nib.save(compress_step1_nii, compress_step1_path)
                printlog(log_file, f"Compressed data saved at {compress_step1_path}")
                
                printlog(log_file, f"Processing:  {step1_path} {step2_path}")
                printlog(log_file, f"s1 shape: {step1_data.shape}, s2 shape: {step2_data.shape} s3 shape: {compress_step1_data.shape}")
                printlog(log_file, f"s1 mean: {np.mean(step1_data):.4f}, s1 std: {np.std(step1_data):.4f}")
                printlog(log_file, f"s2 mean: {np.mean(step2_data):.4f}, s2 std: {np.std(step2_data):.4f}")
                printlog(log_file, f"s3 mean: {np.mean(compress_step1_data):.4f}, s3 std: {np.std(compress_step1_data):.4f}")
                printlog(log_file, f"s1 min: {np.min(step1_data):.4f}, s1 max: {np.max(step1_data):.4f}")
                printlog(log_file, f"s2 min: {np.min(step2_data):.4f}, s2 max: {np.max(step2_data):.4f}")
                printlog(log_file, f"s3 min: {np.min(compress_step1_data):.4f}, s3 max: {np.max(compress_step1_data):.4f}")
                printlog(log_file, f"s1 dtype: {step1_data.dtype}, s2 dtype: {step2_data.dtype}, s3 dtype: {compress_step1_data.dtype}")

                printlog(log_file, ">" * 50)



if __name__ == "__main__":
    main()



# tar -czvf TOFNAC_CTAC_hash.tar.gz TOFNAC_CTAC_hash