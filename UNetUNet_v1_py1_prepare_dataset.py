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



# tar -czvf TOFNAC_CTAC_hash.tar.gz TOFNAC_CTAC_hash
# <><><><><><><><><><><><><><><><><><><><>
# data_dict = {
#     4055: "PAW055",
#     4069: "NAF069",
#     4079: "SAM079",
#     4094: "KQA094",
#     4105: "TVA105",
#     4120: "HNJ120",
#     4130: "JQR130",
#     4139: "RFK139",
#     4058: "EIA058",
#     4073: "YKY073",
#     4081: "TTE081",
#     4096: "BII096",
#     4106: "GSB106",
#     4124: "BPO124",
#     4131: "KWX131",
#     4061: "JLB061",
#     4074: "SPT074",
#     4084: "KZF084",
#     4098: "XZG098",
#     4114: "RSE114",
#     4125: "OOP125",
#     4134: "ONC134",
#     4066: "EGS066",
#     4077: "MLU077",
#     4091: "NKQ091",
#     4099: "DZS099",
#     4115: "WVX115",
#     4128: "LCQ128",
#     4137: "FNG137",
#     4068: "SCH068",
#     4078: "FGX078",
#     4092: "ZTS092",
#     4103: "NIR103",
#     4118: "LBO118",
#     4129: "SNF129",
#     4138: "WLX138"
# }
# <><><><><><><><><><><><><><><><><><><><>
# 4055 : PAW055
# 4069 : NAF069
# 4079 : SAM079
# 4094 : KQA094
# 4105 : TVA105
# 4120 : HNJ120
# 4130 : JQR130
# 4139 : RFK139
# 4058 : EIA058
# 4073 : YKY073
# 4081 : TTE081
# 4096 : BII096
# 4106 : GSB106
# 4124 : BPO124
# 4131 : KWX131
# 4061 : JLB061
# 4074 : SPT074
# 4084 : KZF084
# 4098 : XZG098
# 4114 : RSE114
# 4125 : OOP125
# 4134 : ONC134
# 4066 : EGS066
# 4077 : MLU077
# 4091 : NKQ091
# 4099 : DZS099
# 4115 : WVX115
# 4128 : LCQ128
# 4137 : FNG137
# 4068 : SCH068
# 4078 : FGX078
# 4092 : ZTS092
# 4103 : NIR103
# 4118 : LBO118
# 4129 : SNF129
# 4138 : WLX138
# <><><><><><><><><><><><><><><><><><><><>
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4055.nii.gz B100/CTACIVV_resample/CTACIVV_4055.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9851
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0326, TOFNAC std: 0.0669
# >>>CTAC mean: 0.0443, CTAC std: 0.0987
# Saved PAW055 at ./B100/TOFNAC_CTAC_hash/PAW055_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/PAW055_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4058.nii.gz B100/CTACIVV_resample/CTACIVV_4058.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9608
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0361, TOFNAC std: 0.0679
# >>>CTAC mean: 0.0432, CTAC std: 0.0978
# Saved EIA058 at ./B100/TOFNAC_CTAC_hash/EIA058_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/EIA058_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4061.nii.gz B100/CTACIVV_resample/CTACIVV_4061.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9657
# >>>CTAC min: 0.0000, CTAC max: 0.8950
# >>>TOFNAC mean: 0.0369, TOFNAC std: 0.0593
# >>>CTAC mean: 0.0602, CTAC std: 0.1066
# Saved JLB061 at ./B100/TOFNAC_CTAC_hash/JLB061_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/JLB061_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4066.nii.gz B100/CTACIVV_resample/CTACIVV_4066.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.9184
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0402, TOFNAC std: 0.0657
# >>>CTAC mean: 0.0585, CTAC std: 0.1068
# Saved EGS066 at ./B100/TOFNAC_CTAC_hash/EGS066_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/EGS066_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4068.nii.gz B100/CTACIVV_resample/CTACIVV_4068.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0854
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0282, TOFNAC std: 0.0613
# >>>CTAC mean: 0.0411, CTAC std: 0.0950
# Saved SCH068 at ./B100/TOFNAC_CTAC_hash/SCH068_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/SCH068_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4069.nii.gz B100/CTACIVV_resample/CTACIVV_4069.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.1513
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0325, TOFNAC std: 0.0651
# >>>CTAC mean: 0.0374, CTAC std: 0.0941
# Saved NAF069 at ./B100/TOFNAC_CTAC_hash/NAF069_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/NAF069_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4073.nii.gz B100/CTACIVV_resample/CTACIVV_4073.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9403
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0214, TOFNAC std: 0.0511
# >>>CTAC mean: 0.0286, CTAC std: 0.0818
# Saved YKY073 at ./B100/TOFNAC_CTAC_hash/YKY073_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/YKY073_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4074.nii.gz B100/CTACIVV_resample/CTACIVV_4074.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0151
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0287, TOFNAC std: 0.0528
# >>>CTAC mean: 0.0418, CTAC std: 0.0936
# Saved SPT074 at ./B100/TOFNAC_CTAC_hash/SPT074_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/SPT074_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4077.nii.gz B100/CTACIVV_resample/CTACIVV_4077.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.1959
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0400, TOFNAC std: 0.0693
# >>>CTAC mean: 0.0479, CTAC std: 0.1008
# Saved MLU077 at ./B100/TOFNAC_CTAC_hash/MLU077_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/MLU077_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4078.nii.gz B100/CTACIVV_resample/CTACIVV_4078.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0282
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0327, TOFNAC std: 0.0702
# >>>CTAC mean: 0.0412, CTAC std: 0.0939
# Saved FGX078 at ./B100/TOFNAC_CTAC_hash/FGX078_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/FGX078_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4079.nii.gz B100/CTACIVV_resample/CTACIVV_4079.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.1022
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0391, TOFNAC std: 0.0674
# >>>CTAC mean: 0.0557, CTAC std: 0.1055
# Saved SAM079 at ./B100/TOFNAC_CTAC_hash/SAM079_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/SAM079_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4081.nii.gz B100/CTACIVV_resample/CTACIVV_4081.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0464
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0356, TOFNAC std: 0.0682
# >>>CTAC mean: 0.0416, CTAC std: 0.0954
# Saved TTE081 at ./B100/TOFNAC_CTAC_hash/TTE081_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/TTE081_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4084.nii.gz B100/CTACIVV_resample/CTACIVV_4084.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0772
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0377, TOFNAC std: 0.0785
# >>>CTAC mean: 0.0329, CTAC std: 0.0866
# Saved KZF084 at ./B100/TOFNAC_CTAC_hash/KZF084_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/KZF084_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4091.nii.gz B100/CTACIVV_resample/CTACIVV_4091.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.1130
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0255, TOFNAC std: 0.0574
# >>>CTAC mean: 0.0341, CTAC std: 0.0863
# Saved NKQ091 at ./B100/TOFNAC_CTAC_hash/NKQ091_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/NKQ091_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4092.nii.gz B100/CTACIVV_resample/CTACIVV_4092.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.1462
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0386, TOFNAC std: 0.0728
# >>>CTAC mean: 0.0310, CTAC std: 0.0869
# Saved ZTS092 at ./B100/TOFNAC_CTAC_hash/ZTS092_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/ZTS092_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4094.nii.gz B100/CTACIVV_resample/CTACIVV_4094.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9815
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0323, TOFNAC std: 0.0724
# >>>CTAC mean: 0.0339, CTAC std: 0.0890
# Saved KQA094 at ./B100/TOFNAC_CTAC_hash/KQA094_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/KQA094_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4096.nii.gz B100/CTACIVV_resample/CTACIVV_4096.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9782
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0399, TOFNAC std: 0.0703
# >>>CTAC mean: 0.0643, CTAC std: 0.1075
# Saved BII096 at ./B100/TOFNAC_CTAC_hash/BII096_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/BII096_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4098.nii.gz B100/CTACIVV_resample/CTACIVV_4098.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.4360
# >>>CTAC min: 0.0000, CTAC max: 0.9647
# >>>TOFNAC mean: 0.0412, TOFNAC std: 0.0719
# >>>CTAC mean: 0.0577, CTAC std: 0.1068
# Saved XZG098 at ./B100/TOFNAC_CTAC_hash/XZG098_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/XZG098_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4099.nii.gz B100/CTACIVV_resample/CTACIVV_4099.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0673
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0364, TOFNAC std: 0.0799
# >>>CTAC mean: 0.0309, CTAC std: 0.0845
# Saved DZS099 at ./B100/TOFNAC_CTAC_hash/DZS099_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/DZS099_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4103.nii.gz B100/CTACIVV_resample/CTACIVV_4103.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9846
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0365, TOFNAC std: 0.0659
# >>>CTAC mean: 0.0548, CTAC std: 0.1050
# Saved NIR103 at ./B100/TOFNAC_CTAC_hash/NIR103_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/NIR103_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4105.nii.gz B100/CTACIVV_resample/CTACIVV_4105.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.8213
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0404, TOFNAC std: 0.0841
# >>>CTAC mean: 0.0307, CTAC std: 0.0840
# Saved TVA105 at ./B100/TOFNAC_CTAC_hash/TVA105_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/TVA105_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4106.nii.gz B100/CTACIVV_resample/CTACIVV_4106.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0139
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0308, TOFNAC std: 0.0651
# >>>CTAC mean: 0.0369, CTAC std: 0.0903
# Saved GSB106 at ./B100/TOFNAC_CTAC_hash/GSB106_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/GSB106_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4114.nii.gz B100/CTACIVV_resample/CTACIVV_4114.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0113
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0354, TOFNAC std: 0.0715
# >>>CTAC mean: 0.0527, CTAC std: 0.1011
# Saved RSE114 at ./B100/TOFNAC_CTAC_hash/RSE114_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/RSE114_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4115.nii.gz B100/CTACIVV_resample/CTACIVV_4115.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9778
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0474, TOFNAC std: 0.0741
# >>>CTAC mean: 0.0476, CTAC std: 0.0983
# Saved WVX115 at ./B100/TOFNAC_CTAC_hash/WVX115_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/WVX115_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4118.nii.gz B100/CTACIVV_resample/CTACIVV_4118.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9967
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0371, TOFNAC std: 0.0675
# >>>CTAC mean: 0.0403, CTAC std: 0.0943
# Saved LBO118 at ./B100/TOFNAC_CTAC_hash/LBO118_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/LBO118_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4120.nii.gz B100/CTACIVV_resample/CTACIVV_4120.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.1992
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0406, TOFNAC std: 0.0845
# >>>CTAC mean: 0.0294, CTAC std: 0.0840
# Saved HNJ120 at ./B100/TOFNAC_CTAC_hash/HNJ120_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/HNJ120_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4124.nii.gz B100/CTACIVV_resample/CTACIVV_4124.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.2251
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0295, TOFNAC std: 0.0587
# >>>CTAC mean: 0.0419, CTAC std: 0.0961
# Saved BPO124 at ./B100/TOFNAC_CTAC_hash/BPO124_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/BPO124_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4125.nii.gz B100/CTACIVV_resample/CTACIVV_4125.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.1894
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0387, TOFNAC std: 0.0720
# >>>CTAC mean: 0.0519, CTAC std: 0.1036
# Saved OOP125 at ./B100/TOFNAC_CTAC_hash/OOP125_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/OOP125_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4128.nii.gz B100/CTACIVV_resample/CTACIVV_4128.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9533
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0389, TOFNAC std: 0.0605
# >>>CTAC mean: 0.0648, CTAC std: 0.1103
# Saved LCQ128 at ./B100/TOFNAC_CTAC_hash/LCQ128_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/LCQ128_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4129.nii.gz B100/CTACIVV_resample/CTACIVV_4129.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0091
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0226, TOFNAC std: 0.0627
# >>>CTAC mean: 0.0251, CTAC std: 0.0760
# Saved SNF129 at ./B100/TOFNAC_CTAC_hash/SNF129_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/SNF129_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4130.nii.gz B100/CTACIVV_resample/CTACIVV_4130.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9782
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0260, TOFNAC std: 0.0564
# >>>CTAC mean: 0.0396, CTAC std: 0.0938
# Saved JQR130 at ./B100/TOFNAC_CTAC_hash/JQR130_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/JQR130_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4131.nii.gz B100/CTACIVV_resample/CTACIVV_4131.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 1.0056
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0260, TOFNAC std: 0.0622
# >>>CTAC mean: 0.0425, CTAC std: 0.0949
# Saved KWX131 at ./B100/TOFNAC_CTAC_hash/KWX131_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/KWX131_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4134.nii.gz B100/CTACIVV_resample/CTACIVV_4134.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9827
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0330, TOFNAC std: 0.0630
# >>>CTAC mean: 0.0350, CTAC std: 0.0892
# Saved ONC134 at ./B100/TOFNAC_CTAC_hash/ONC134_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/ONC134_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4137.nii.gz B100/CTACIVV_resample/CTACIVV_4137.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9848
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0401, TOFNAC std: 0.0693
# >>>CTAC mean: 0.0494, CTAC std: 0.1000
# Saved FNG137 at ./B100/TOFNAC_CTAC_hash/FNG137_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/FNG137_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4138.nii.gz B100/CTACIVV_resample/CTACIVV_4138.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 7.1722
# >>>CTAC min: 0.0000, CTAC max: 0.8680
# >>>TOFNAC mean: 0.0339, TOFNAC std: 0.0728
# >>>CTAC mean: 0.0360, CTAC std: 0.0904
# Saved WLX138 at ./B100/TOFNAC_CTAC_hash/WLX138_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/WLX138_CTAC.nii.gz
# Processing:  B100/TOFNAC_resample/PET_TOFNAC_E4139.nii.gz B100/CTACIVV_resample/CTACIVV_4139.nii.gz
# >>>After normalization
# >>>TOFNAC min: 0.0000, TOFNAC max: 0.9978
# >>>CTAC min: 0.0000, CTAC max: 1.0000
# >>>TOFNAC mean: 0.0317, TOFNAC std: 0.0591
# >>>CTAC mean: 0.0420, CTAC std: 0.0950
# Saved RFK139 at ./B100/TOFNAC_CTAC_hash/RFK139_TOFNAC.nii.gz and ./B100/TOFNAC_CTAC_hash/RFK139_CTAC.nii.gz
