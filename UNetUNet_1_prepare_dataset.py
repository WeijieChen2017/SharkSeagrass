import argparse
import h5py

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
    train_folder = list(map(int, train_folder.split(",")))
    val_folder = list(map(int, val_folder.split(",")))
    test_folder = list(map(int, test_folder.split(",")))

    print(f"train_fold:[{train_folder}], val_fold:[{val_folder}], test_fold:[{test_folder}]")

    # the following is the way we build the dataset
    # fold_filename = f"{root_folder}fold_{i_fold}.hdf5"
    # with h5py.File(fold_filename, "w") as f:
    #     for i_case, case in enumerate(data_fold):
    #         f.create_dataset(f"TOFNAC_{i_case}", data=case["TOFNAC"])
    #         f.create_dataset(f"CTAC_{i_case}", data=case["CTAC"])

    for fold in train_folder:
        print(f"Processing training fold {fold}")
        fold_filename = f"fold_{fold}.hdf5"
        TOFNAC_data = []
        CTAC_data = []
        with h5py.File(fold_filename, "r") as f:
            len_file = len(f) // 2
            print(f"Number of cases: {len_file}")
            for i_case in range(len_file):
                TOFNAC = f[f"TOFNAC_{i_case}"]
                CTAC = f[f"CTAC_{i_case}"]
                print(f"TOFNAC shape: {TOFNAC.shape}, CTAC shape: {CTAC.shape}")
                TOFNAC_data.append(TOFNAC)
                CTAC_data.append(CTAC)
        print("Done")

if __name__ == "__main__":
    main()
