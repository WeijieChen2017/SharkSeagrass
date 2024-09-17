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


import matplotlib.pyplot as plt

def inputs_labels_outputs_to_imgs(inputs, labels, outputs, cube_size, cut_index, i):

    if cut_index == "z":
        img_PET = np.rot90(inputs[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1)) # -1 to 1
       
        img_CT = np.rot90(labels[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1)) # -1 to 1
        
        img_pred = np.rot90(outputs[i, 0, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1)) # -1 to 1

    elif cut_index == "y":
        img_PET = np.rot90(inputs[i, :, :, cube_size // 2, :].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1))

        img_CT = np.rot90(labels[i, :, :, cube_size // 2, :].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1))

        img_pred = np.rot90(outputs[i, 0, :, :, cube_size // 2, :].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1))

    elif cut_index == "x":

        img_PET = np.rot90(inputs[i, :, cube_size // 2, :, :].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1))

        img_CT = np.rot90(labels[i, :, cube_size // 2, :, :].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1))

        img_pred = np.rot90(outputs[i, 0, :, cube_size // 2, :, :].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1))
    
    else:
        raise ValueError("cut_index must be either x, y, or z")

    img_5 = img_pred + img_PET # -1 to 1
    img_5 = (img_5 + 1) / 2 # 0 to 1
    img_1 = (img_PET + 1) / 2 # 0 to 1
    img_2 = (img_CT + 1) / 2 # 0 to 1
    img_3 = (img_pred + 1) / 2 # 0 to 1
    img_4 = img_2 - img_1 # -1 to 1
    img_4 = (img_4 + 1) / 2 # 0 to 1
    
    return img_1, img_2, img_3, img_4, img_5

def plot_results(inputs, labels, outputs, idx_epoch, root_folder, cube_size, global_config):
    # plot the results

    n_block = 1
    if inputs.shape[0] < n_block:
        n_block = inputs.shape[0]
    fig = plt.figure(figsize=(12, n_block*3.6), dpi=300)

    n_row = n_block * 3
    n_col = 10

    # compute mean for img_PET
    img_PET_mean = np.mean(inputs.detach().cpu().numpy(), axis=(1, 2, 3, 4))
    img_PET_mean = (img_PET_mean + 1) / 2
    img_PET_mean = img_PET_mean * CT_NORM + CT_MIN
    fig.suptitle(f"Epoch {idx_epoch}, mean PET: {img_PET_mean}", fontsize=16)

    # for axial view

    for i_asc in range(3): # for axial, sagittal, coronal

        if i_asc == 0:
            cut_index = "z"
        elif i_asc == 1:
            cut_index = "y"
        elif i_asc == 2:
            cut_index = "x"
        
        for i in range(n_block):

            img_1, img_2, img_3, img_4, img_5 = inputs_labels_outputs_to_imgs(inputs, labels, outputs, cube_size, cut_index, i)
            
            # first three and hist
            plt.subplot(n_row, n_col, i * n_col + 1 + i_asc * 10)
            plt.imshow(img_1, cmap="gray", vmin=0, vmax=0.5) # x
            # plt.title("input PET")
            if i == 0:
                plt.title("input STEP1")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 2 + i_asc * 10)
            plt.imshow(img_2, cmap="gray", vmin=0, vmax=0.5) # y
            # plt.title("label CT")
            if i == 0:
                plt.title("input STEP2")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 3 + i_asc * 10)
            # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
            plt.imshow(img_3, cmap="bwr", vmin=0.45, vmax=0.55) # yhat = f(x) + x, img_pred = f(x) = yhat - x
            # plt.title("output CT")
            plt.colorbar()
            if i == 0:
                plt.title("f(x)=yhat-x")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 4 + i_asc * 10)
            # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
            plt.imshow(img_4, cmap="bwr", vmin=0.45, vmax=0.55) # y = x + (y - x), (y - x) = y - x
            plt.colorbar()
            if i == 0:
                plt.title("gt=y-x")
            # plt.title("output CT")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 5 + i_asc * 10)
            # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
            plt.imshow(img_5, cmap="gray", vmin=0, vmax=0.5) # yhat
            if i == 0:
                plt.title("yhat")
            # plt.title("output CT")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 6 + i_asc * 10)
            # img_PET = np.clip(img_PET, 0, 1)
            plt.hist(img_1.flatten(), bins=100)
            # plt.title("input PET")
            if i == 0:
                plt.title("input STEP1")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(0, 1)

            plt.subplot(n_row, n_col, i * n_col + 7 + i_asc * 10)
            # img_CT = np.clip(img_CT, 0, 1)
            plt.hist(img_2.flatten(), bins=100)
            # plt.title("label CT")
            if i == 0:
                plt.title("input STEP2")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(0, 1)

            plt.subplot(n_row, n_col, i * n_col + 8 + i_asc * 10)
            # img_pred = np.clip(img_pred, 0, 1)
            plt.hist((img_5 - img_1).flatten(), bins=100)
            # plt.title("output CT")
            if i == 0:
                plt.title("f(x)=yhat-x")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(-1, 1)

            plt.subplot(n_row, n_col, i * n_col + 9 + i_asc * 10)
            # img_pred = np.clip(img_pred, 0, 1)
            plt.hist((img_2 - img_1).flatten(), bins=100)
            # plt.title("output CT")
            if i == 0:
                plt.title("gt=y-x")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(-1, 1)

            plt.subplot(n_row, n_col, i * n_col + 10 + i_asc * 10)
            # img_pred = np.clip(img_pred, 0, 1)
            plt.hist((img_5).flatten(), bins=100)
            # plt.title("output CT")
            if i == 0:
                plt.title("yhat")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(0, 1)


    plt.tight_layout()
    plt.savefig(os.path.join(root_folder, f"epoch_{idx_epoch}.png"))
    plt.close()






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


