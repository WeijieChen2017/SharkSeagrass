import os
import time
import json
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
# import pytorch_lightning as pl

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,
    RandSpatialCropd,
)
from monai.data import CacheDataset, DataLoader



def prepare_dataset(data_div_json, global_config):
    
    with open(data_div_json, "r") as f:
        data_div = json.load(f)
    
    cv = global_config["cross_validation"]

    train_list = data_div[f"cv_{cv}"]["train"]
    val_list = data_div[f"cv_{cv}"]["val"]
    test_list = data_div[f"cv_{cv}"]["test"]

    # num_train = len(train_list)
    # num_val = len(val_list)
    # num_test = len(test_list)

    str_train_list = ", ".join(train_list)
    str_val_list = ", ".join(val_list)
    str_test_list = ", ".join(test_list)

    global_config["logger"].log(0, "data_split_train", str_train_list)
    global_config["logger"].log(0, "data_split_val", str_val_list)
    global_config["logger"].log(0, "data_split_test", str_test_list)

    # construct the data path list
    train_path_list = []
    val_path_list = []
    test_path_list = []

    for hashname in train_list:
        train_path_list.append({
            # BPO124_CTAC_pred_cv0.nii.gz have been normalized and clipped to [0, 1]
            "STEP1": f"cv{cv}_256_clip/{hashname}_CTAC_pred_cv{cv}.nii.gz",
            # BPO124_CTAC.nii.gz have been normalized and clipped to [0, 1]
            "STEP2": f"TC256/{hashname}_CTAC_256.nii.gz",
        })

    for hashname in val_list:
        val_path_list.append({
            "STEP1": f"cv{cv}_256_clip/{hashname}_CTAC_pred_cv{cv}.nii.gz",
            "STEP2": f"TC256/{hashname}_CTAC_256.nii.gz",
        })

    for hashname in test_list:
        test_path_list.append({
            "STEP1": f"cv{cv}_256_clip/{hashname}_CTAC_pred_cv{cv}.nii.gz",
            "STEP2": f"TC256/{hashname}_CTAC_256.nii.gz",
        })

    # save the data division file
    root_folder = global_config["root_folder"]
    data_division_file = os.path.join(root_folder, "data_division.json")
    data_division_dict = {
        "train": train_path_list,
        "val": val_path_list,
        "test": test_path_list,
    }
    for key in data_division_dict.keys():
        print(key)
        for key2 in data_division_dict[key]:
            print(key2)

    with open(data_division_file, "w") as f:
        json.dump(data_division_dict, f, indent=4)

    input_modality = global_config["model_step2_params"]["input_modality"]
    # input_modality_dict = {
    #     "x": input_modality[0],
    #     "y": input_modality[1],
    # }
    cube_size = global_config["model_step2_params"]["cube_size"]
    # in_channel = global_config["model_step1_params"]["ddconfig"]["in_channels"]
    # out_channel = global_config["model_step1_params"]["ddconfig"]["out_ch"]

    # set the data transform
    train_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
            RandSpatialCropd(
                keys=input_modality,
                roi_size=(cube_size, cube_size, cube_size), 
                random_size=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["y"],
            #     roi_size=(img_size, img_size, out_channel),
            #     random_size=False),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["x"],
            #     channel_dim=-1),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["y"],
            #     channel_dim="none" if out_channel == 1 else -1),

        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
            RandSpatialCropd(
                keys=input_modality,
                roi_size=(cube_size, cube_size, cube_size), 
                random_size=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["y"],
            #     roi_size=(img_size, img_size, out_channel),
            #     random_size=False),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["x"],
            #     channel_dim=-1),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["y"],
            #     channel_dim="none" if out_channel == 1 else -1),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
            RandSpatialCropd(
                keys=input_modality,
                roi_size=(cube_size, cube_size, cube_size), 
                random_size=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["y"],
            #     roi_size=(img_size, img_size, out_channel),
            #     random_size=False),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["x"],
            #     channel_dim=-1),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["y"],
            #     channel_dim="none" if out_channel == 1 else -1),
        ]
    )

    

    train_ds = CacheDataset(
        data=train_path_list,
        transform=train_transforms,
        # cache_num=num_train_files,
        cache_rate=global_config["data_loader_params"]["train"]["cache_rate"],
        num_workers=global_config["data_loader_params"]["train"]["num_workers_cache"],
    )

    val_ds = CacheDataset(
        data=val_path_list,
        transform=val_transforms, 
        # cache_num=num_val_files,
        cache_rate=global_config["data_loader_params"]["val"]["cache_rate"],
        num_workers=global_config["data_loader_params"]["val"]["num_workers_cache"],
    )

    test_ds = CacheDataset(
        data=test_path_list,
        transform=test_transforms,
        # cache_num=num_test_files,
        cache_rate=global_config["data_loader_params"]["test"]["cache_rate"],
        num_workers=global_config["data_loader_params"]["test"]["num_workers_cache"],
    )

    train_loader = DataLoader(train_ds, 
                            batch_size=global_config["data_loader_params"]["train"]["batch_size"],
                            shuffle=global_config["data_loader_params"]["train"]["shuffle"],
                            num_workers=global_config["data_loader_params"]["train"]["num_workers_loader"],

    )
    val_loader = DataLoader(val_ds, 
                            batch_size=global_config["data_loader_params"]["val"]["batch_size"],
                            shuffle=global_config["data_loader_params"]["val"]["shuffle"],
                            num_workers=global_config["data_loader_params"]["val"]["num_workers_loader"],
    )

    test_loader = DataLoader(test_ds,
                            batch_size=global_config["data_loader_params"]["test"]["batch_size"],
                            shuffle=global_config["data_loader_params"]["test"]["shuffle"],
                            num_workers=global_config["data_loader_params"]["test"]["num_workers_loader"],
    )

    return train_loader, val_loader, test_loader


class simple_logger():
    def __init__(self, log_file_path, global_config):
        self.log_file_path = log_file_path
        self.log_dict = dict()
        self.IS_LOGGER_WANDB = global_config["IS_LOGGER_WANDB"]
        self.wandb_run = global_config["wandb_run"]
    
    def log(self, global_epoch, key, msg):
        if key not in self.log_dict.keys():
            self.log_dict[key] = dict()
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dict[key] = {
            "time": current_time,
            "global_epoch": global_epoch,
            "msg": msg
        }
        log_str = f"{current_time} Global epoch: {global_epoch}, {key}, {msg}\n"
        with open(self.log_file_path, "a") as f:
            f.write(log_str)
        print(log_str)

        # log to wandb if msg is number
        if self.IS_LOGGER_WANDB and isinstance(msg, (int, float)):
            self.wandb_run.log({key: msg})




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

