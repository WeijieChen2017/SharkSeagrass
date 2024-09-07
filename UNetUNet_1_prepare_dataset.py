import numpy as np
import argparse
import h5py
import json
import os

def main():
    argparser = argparse.ArgumentParser(description='Prepare dataset for training')
    argparser.add_argument('--train_folder', type=str, default="0,1,2", help='Path to the training fold')
    argparser.add_argument('--val_folder', type=str, default="3", help='Path to the validation fold')
    argparser.add_argument('--test_folder', type=str, default="4", help='Path to the testing fold')

    args = argparser.parse_args()

    train_folder = args.train_folder
    val_folder = args.val_folder
    test_folder = args.test_folder

    # conver to list
    train_fold_list = list(map(int, train_folder.split(",")))
    val_fold_list = list(map(int, val_folder.split(",")))
    test_fold_list = list(map(int, test_folder.split(",")))

    print(f"train_fold:[{train_folder}], val_fold:[{val_folder}], test_fold:[{test_folder}]")

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

    # for training
    for fold in train_fold_list:
        print()
        print(f"Processing training fold {fold}")
        fold_filename = f"fold_{fold}.hdf5"
        with h5py.File(fold_filename, "r") as f:
            len_file = len(f) // 2
            print(f">>>Number of cases: {len_file}")
            for i_case in range(len_file):
                TOFNAC = f[f"TOFNAC_{i_case}"]
                CTAC = f[f"CTAC_{i_case}"]
                print(f">>>TOFNAC shape: {TOFNAC.shape}, CTAC shape: {CTAC.shape}")

                # save the slice
                len_z = TOFNAC.shape[2]
                for i_z in range(len_z):
                    if i_z == 0:
                        index_list = [0, 0, 1]
                    elif i_z == len_z - 1:
                        index_list = [i_z - 1, i_z, i_z]
                    else:
                        index_list = [i_z - 1, i_z, i_z + 1]
                    
                    print(index_list)
                    slice_TOFNAC = TOFNAC[:, :, index_list]
                    slice_CTAC = CTAC[:, :, index_list]

                    save_filename_TOFNAC = f"{train_folder}fold_{fold}_case_{i_case}_slice_{i_z}_TOFNAC.npy"
                    save_filename_CTAC = f"{train_folder}fold_{fold}_case_{i_case}_slice_{i_z}_CTAC.npy"

                    np.save(save_filename_TOFNAC, slice_TOFNAC)
                    np.save(save_filename_CTAC, slice_CTAC)

                    data_div_dict["train"].append({
                        "TOFNAC": save_filename_TOFNAC,
                        "CTAC": save_filename_CTAC
                    })

                    print(f">>>[{i_z+1}]/[{len_z}]Fold {fold} case {i_case} slice {i_z} saved at {save_filename_TOFNAC} and {save_filename_CTAC}")

    # for validation
    for fold in val_fold_list:
        print()
        print(f">>>Processing validation fold {fold}")
        fold_filename = f"fold_{fold}.hdf5"
        with h5py.File(fold_filename, "r") as f:
            len_file = len(f) // 2
            print(f">>>Number of cases: {len_file}")
            for i_case in range(len_file):
                TOFNAC = f[f"TOFNAC_{i_case}"]
                CTAC = f[f"CTAC_{i_case}"]
                print(f">>>TOFNAC shape: {TOFNAC.shape}, CTAC shape: {CTAC.shape}")

                # save the slice
                len_z = TOFNAC.shape[2]
                for i_z in range(len_z):
                    if i_z == 0:
                        index_list = [0, 0, 1]
                    elif i_z == len_z - 1:
                        index_list = [i_z - 1, i_z, i_z]
                    else:
                        index_list = [i_z - 1, i_z, i_z + 1]
                    
                    slice_TOFNAC = TOFNAC[:, :, index_list]
                    slice_CTAC = CTAC[:, :, index_list]

                    save_filename_TOFNAC = f"{val_folder}fold_{fold}_case_{i_case}_slice_{i_z}_TOFNAC.npy"
                    save_filename_CTAC = f"{val_folder}fold_{fold}_case_{i_case}_slice_{i_z}_CTAC.npy"

                    np.save(save_filename_TOFNAC, slice_TOFNAC)
                    np.save(save_filename_CTAC, slice_CTAC)

                    data_div_dict["val"].append({
                        "TOFNAC": save_filename_TOFNAC,
                        "CTAC": save_filename_CTAC
                    })

                    print(f">>>[{i_z+1}]/[{len_z}]Fold {fold} case {i_case} slice {i_z} saved at {save_filename_TOFNAC} and {save_filename_CTAC}")

    # for testing
    for fold in test_fold_list:
        print()
        print(f">>>Processing testing fold {fold}")
        fold_filename = f"fold_{fold}.hdf5"
        with h5py.File(fold_filename, "r") as f:
            len_file = len(f) // 2
            print(f">>>Number of cases: {len_file}")
            for i_case in range(len_file):
                TOFNAC = f[f"TOFNAC_{i_case}"]
                CTAC = f[f"CTAC_{i_case}"]
                print(f">>>TOFNAC shape: {TOFNAC.shape}, CTAC shape: {CTAC.shape}")

                # save the slice
                len_z = TOFNAC.shape[2]
                for i_z in range(len_z):
                    if i_z == 0:
                        index_list = [0, 0, 1]
                    elif i_z == len_z - 1:
                        index_list = [i_z - 1, i_z, i_z]
                    else:
                        index_list = [i_z - 1, i_z, i_z + 1]
                    
                    slice_TOFNAC = TOFNAC[:, :, index_list]
                    slice_CTAC = CTAC[:, :, index_list]

                    save_filename_TOFNAC = f"{test_folder}fold_{fold}_case_{i_case}_slice_{i_z}_TOFNAC.npy"
                    save_filename_CTAC = f"{test_folder}fold_{fold}_case_{i_case}_slice_{i_z}_CTAC.npy"

                    np.save(save_filename_TOFNAC, slice_TOFNAC)
                    np.save(save_filename_CTAC, slice_CTAC)

                    data_div_dict["test"].append({
                        "TOFNAC": save_filename_TOFNAC,
                        "CTAC": save_filename_CTAC
                    })

                    print(f">>>[{i_z+1}]/[{len_z}]Fold {fold} case {i_case} slice {i_z} saved at {save_filename_TOFNAC} and {save_filename_CTAC}")

    # save the data_div.json
    with open(data_div_json, "w") as f:
        json.dump(data_div_dict, f)
        
    print("Done")

if __name__ == "__main__":
    main()
