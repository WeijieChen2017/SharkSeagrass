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


def main():
    argparser = argparse.ArgumentParser(description='Prepare dataset for training')
    argparser.add_argument('--train_fold', type=str, default="0,1,2", help='Path to the training fold')
    argparser.add_argument('--val_fold', type=str, default="3", help='Path to the validation fold')
    argparser.add_argument('--test_fold', type=str, default="4", help='Path to the testing fold')

    args = argparser.parse_args()

    train_fold = args.train_fold
    val_fold = args.val_fold
    test_fold = args.test_fold

    # conver to list
    train_fold_list = list(map(int, train_fold.split(",")))
    val_fold_list = list(map(int, val_fold.split(",")))
    test_fold_list = list(map(int, test_fold.split(",")))

    print(f"train_fold:[{train_fold}], val_fold:[{val_fold}], test_fold:[{test_fold}]")

    # the following is the way we build the dataset
    # fold_filename = f"{root_folder}fold_{i_fold}.hdf5"
    # with h5py.File(fold_filename, "w") as f:
    #     for i_case, case in enumerate(data_fold):
    #         f.create_dataset(f"TOFNAC_{i_case}", data=case["TOFNAC"])
    #         f.create_dataset(f"CTAC_{i_case}", data=case["CTAC"])

    train_folder = "dataset/train/"
    val_folder = "dataset/val/"
    test_folder = "dataset/test/"

    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    data_div_json = "data_div.json"
    data_div_dict = {
        "train": [],
        "val": [],
        "test": [],
    }

    data_mode_list = [
        [train_fold_list, "train", train_folder],
        [val_fold_list, "val", val_folder], 
        [test_fold_list, "test", test_folder],
    ]

    # for training
    for data_mode in data_mode_list:
        fold_list = data_mode[0]
        mode = data_mode[1]
        save_folder = data_mode[2]
        for fold in fold_list:
            print()
            print(f"Processing {mode} fold {fold}")
            fold_filename = f"fold_{fold}.hdf5"
            with h5py.File(fold_filename, "r") as f:
                len_file = len(f) // 2
                print(f">>>Number of cases: {len_file}")
                for i_case in range(len_file):
                    TOFNAC_data = f[f"TOFNAC_{i_case}"][()]
                    CTAC_data = f[f"CTAC_{i_case}"][()]
                    print(f">>>TOFNAC shape: {TOFNAC_data.shape}, CTAC shape: {CTAC_data.shape}")
                    print(f">>>TOFNAC min: {TOFNAC_data.min():.4f}, TOFNAC max: {TOFNAC_data.max():.4f}")
                    print(f">>>CTAC min: {CTAC_data.min():.4f}, CTAC max: {CTAC_data.max():.4f}")
                    print(f">>>TOFNAC mean: {TOFNAC_data.mean():.4f}, TOFNAC std: {TOFNAC_data.std():.4f}")
                    print(f">>>CTAC mean: {CTAC_data.mean():.4f}, CTAC std: {CTAC_data.std():.4f}")

                    # normalize the data
                    TOFNAC_data = two_segment_scale(TOFNAC_data, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
                    CTAC_data = np.clip(CTAC_data, MIN_CT, MAX_CT)
                    CTAC_data = (CTAC_data - MIN_CT) / RANGE_CT

                    print(">>>After normalization")
                    print(f">>>TOFNAC min: {TOFNAC_data.min():.4f}, TOFNAC max: {TOFNAC_data.max():.4f}")
                    print(f">>>CTAC min: {CTAC_data.min():.4f}, CTAC max: {CTAC_data.max():.4f}")
                    print(f">>>TOFNAC mean: {TOFNAC_data.mean():.4f}, TOFNAC std: {TOFNAC_data.std():.4f}")
                    print(f">>>CTAC mean: {CTAC_data.mean():.4f}, CTAC std: {CTAC_data.std():.4f}")

                    save_filename_TOFNAC = f"{save_folder}fold_{fold}_case_{i_case}_TOFNAC.npy"
                    save_filename_CTAC = f"{save_folder}fold_{fold}_case_{i_case}_CTAC.npy"

                    np.save(save_filename_TOFNAC, TOFNAC_data)
                    np.save(save_filename_CTAC, CTAC_data)

                    data_div_dict[mode].append({
                        "TOFNAC": save_filename_TOFNAC,
                        "CTAC": save_filename_CTAC
                    })

                    print(f">>>Fold {fold} case {i_case} saved at {save_filename_TOFNAC} and {save_filename_CTAC}")

    # save the data_div.json
    with open(data_div_json, "w") as f:
        json.dump(data_div_dict, f)
        
    print("Done")

if __name__ == "__main__":
    main()
