import os

# # Define the base cache directory
# base_cache_dir = './cache'

# # Define and create necessary subdirectories within the base cache directory
# cache_dirs = {
#     'WANDB_DIR': os.path.join(base_cache_dir, 'wandb'),
#     'WANDB_CACHE_DIR': os.path.join(base_cache_dir, 'wandb_cache'),
#     'WANDB_CONFIG_DIR': os.path.join(base_cache_dir, 'config'),
#     'WANDB_DATA_DIR': os.path.join(base_cache_dir, 'data'),
#     'TRANSFORMERS_CACHE': os.path.join(base_cache_dir, 'transformers'),
#     'MPLCONFIGDIR': os.path.join(base_cache_dir, 'mplconfig')
# }

# # Create the base cache directory if it doesn't exist
# os.makedirs(base_cache_dir, exist_ok=True)

# # Create the necessary subdirectories and set the environment variables
# for key, path in cache_dirs.items():
#     os.makedirs(path, exist_ok=True)
#     os.environ[key] = path

# # set the environment variable to use the GPU if available
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is: ", device)

# import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import time
import glob
import yaml
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    CacheDataset,
)

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    RandRotated,
)

def plot_and_save_x_y_z(x, y, z, num_per_direction=1, savename=None):
    numpy_x = x[0, :, :, :, :].cpu().numpy().squeeze()
    numpy_y = y[0, :, :, :, :].cpu().numpy().squeeze()
    numpy_z = z[0, :, :, :, :].cpu().numpy().squeeze()
    x_clip = np.clip(numpy_x, 0, 1)
    y_clip = np.clip(numpy_y, 0, 1)
    z_clip = np.clip(numpy_z, 0, 1)
    fig_width = num_per_direction * 3
    fig_height = 4
    fig, axs = plt.subplots(3, fig_width, figsize=(fig_width, fig_height), dpi=100)
    # for axial
    for i in range(num_per_direction):
        img_x = x_clip[x_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        img_y = y_clip[y_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        img_z = z_clip[z_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        axs[0, 3*i].imshow(img_x, cmap="gray")
        axs[0, 3*i].set_title(f"A x {x_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i].axis("off")
        axs[1, 3*i].imshow(img_y, cmap="gray")
        axs[1, 3*i].set_title(f"A y {y_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i].axis("off")
        axs[2, 3*i].imshow(img_z, cmap="gray")
        axs[2, 3*i].set_title(f"A z {z_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i].axis("off")
    # for sagittal
    for i in range(num_per_direction):
        img_x = x_clip[:, :, x_clip.shape[2]//(num_per_direction+1)*(i+1)]
        img_y = y_clip[:, :, y_clip.shape[2]//(num_per_direction+1)*(i+1)]
        img_z = z_clip[:, :, z_clip.shape[2]//(num_per_direction+1)*(i+1)]
        axs[0, 3*i+1].imshow(img_x, cmap="gray")
        axs[0, 3*i+1].set_title(f"S x {x_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i+1].axis("off")
        axs[1, 3*i+1].imshow(img_y, cmap="gray")
        axs[1, 3*i+1].set_title(f"S y {y_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i+1].axis("off")
        axs[2, 3*i+1].imshow(img_z, cmap="gray")
        axs[2, 3*i+1].set_title(f"S z {z_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i+1].axis("off")

    # for coronal
    for i in range(num_per_direction):
        img_x = x_clip[:, x_clip.shape[1]//(num_per_direction+1)*(i+1), :]
        img_y = y_clip[:, y_clip.shape[1]//(num_per_direction+1)*(i+1), :]
        img_z = z_clip[:, z_clip.shape[1]//(num_per_direction+1)*(i+1), :]
        axs[0, 3*i+2].imshow(img_x, cmap="gray")
        axs[0, 3*i+2].set_title(f"C x {x_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i+2].axis("off")
        axs[1, 3*i+2].imshow(img_y, cmap="gray")
        axs[1, 3*i+2].set_title(f"C y {y_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i+2].axis("off")
        axs[2, 3*i+2].imshow(img_z, cmap="bwr")
        axs[2, 3*i+2].set_title(f"C z {z_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i+2].axis("off")

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()
    print(f"Save the plot to {savename}")

class local_logger():
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
    
    def log(self, epoch, key, value):
        self.log_file = open(self.log_file_path, 'w')
        self.log_file.write(f"{epoch}, {key}, {value}\n")
    
    def close(self):
        self.log_file.close()

def parse_yaml_arguments():
    parser = argparse.ArgumentParser(description='Train a basic UNet model for PET to synthetic CT.')
    parser.add_argument('--config_file_path', type=str, default="basic_UNet.yaml")
    return parser.parse_args()

def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def main():
    config_file_path = parse_yaml_arguments().config_file_path
    global_config = load_yaml_config(config_file_path)
    os.makedirs(global_config['save_folder'], exist_ok=True)

    # show the global_config
    print("The global_config is: ", global_config)
    gap_sign = global_config["gap_sign"]

    # set the random seed
    random.seed(global_config['random_seed'])
    np.random.seed(global_config['random_seed'])
    torch.manual_seed(global_config['random_seed'])
    print("The random seed is: ", global_config['random_seed'])

    # set the logger
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file_path = f"train_log_{current_time}.txt"
    logger = local_logger(log_file_path)
    print("The log file path is: ", log_file_path)

    # set the input modality
    input_modality = global_config["input_modality"]
    in_channels = len(input_modality)
    print("The input modality is: ", input_modality)

    # set the model
    model = UNet(
        spatial_dims = global_config["spatial_dims"],
        in_channels = in_channels,
        out_channels = global_config["out_channels"],
        channels = global_config["channels"],
        strides = global_config["strides"],
        num_res_units = global_config["num_res_units"],
    ).to(device)
    print(f"The model {model.__class__.__name__} is built successfully.")
    print(gap_sign*50)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=global_config["optimizer_lr"],
                                 weight_decay=global_config["optimizer_weight_decay"])
    
    pix_dim = global_config["pix_dim"]
    volume_size = global_config["volume_size"]

    # set the data transform
    train_transforms = Compose(
        [
            LoadImaged(keys=[input_modality]),
            EnsureChannelFirstd(keys=[input_modality]),
            Orientationd(keys=[input_modality], axcodes="RAS"),
            # Spacingd(
            #     keys=["PET", "CT"],
            #     pixdim=(pix_dim, pix_dim, pix_dim),
            #     mode=("bilinear"),
            # ),
            # ScaleIntensityRanged(
            #     keys=["PET"],
            #     a_min=0,
            #     a_max=4000,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            # ScaleIntensityRanged(
            #     keys=["CT"],
            #     a_min=-1024,
            #     a_max=2976,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            RandSpatialCropd(keys=[input_modality], roi_size=(volume_size, volume_size, volume_size), random_center=True, random_size=False),
            RandFlipd(keys=[input_modality], prob=0.5, spatial_axis=0),
            RandFlipd(keys=[input_modality], prob=0.5, spatial_axis=1),
            RandFlipd(keys=[input_modality], prob=0.5, spatial_axis=2),
            RandRotated(keys=[input_modality], prob=0.5, range_x=15, range_y=15, range_z=15),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=[input_modality]),
            EnsureChannelFirstd(keys=[input_modality]),
            Orientationd(keys=[input_modality], axcodes="RAS"),
            # Spacingd(
            #     keys=["PET", "CT"],
            #     pixdim=(pix_dim, pix_dim, pix_dim),
            #     mode=("bilinear"),
            # ),
            # ScaleIntensityRanged(
            #     keys=["PET"],
            #     a_min=0,
            #     a_max=4000,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            # ScaleIntensityRanged(
            #     keys=["CT"],
            #     a_min=-1024,
            #     a_max=2976,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            RandSpatialCropd(keys=[input_modality], roi_size=(volume_size, volume_size, volume_size), random_center=True, random_size=False),
        ]
    )

    data_division_file = global_config["data_division"]
    # load data_chunks.json and specif chunk_0 to chunk_4 for training, chunk_5 to chunk_7 for validation, chunk_8 and chunk_9 for testing
    with open(data_division_file, "r") as f:
        data_chunk = json.load(f)

    train_files = []
    val_files = []
    test_files = []

    for i in range(3):
        train_files.extend(data_chunk[f"chunk_{i}"])
    for i in range(3, 4):
        val_files.extend(data_chunk[f"chunk_{i}"])
    for i in range(4, 5):
        test_files.extend(data_chunk[f"chunk_{i}"])

    num_train_files = len(train_files)
    num_val_files = len(val_files)
    num_test_files = len(test_files)
    
    print("The number of train files is: ", num_train_files)
    print("The number of val files is: ", num_val_files)
    print("The number of test files is: ", num_test_files)
    print(gap_sign*50)

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=num_train_files,
        cache_rate=global_config["cache_ratio_train"],
        num_workers=global_config["num_workers_train_cache_dataset"],
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms, 
        cache_num=num_val_files,
        cache_rate=global_config["cache_ratio_val"],
        num_workers=global_config["num_workers_val_cache_dataset"],
    )

    train_loader = DataLoader(train_ds, 
                              batch_size=global_config["batch_size_train"],
                              shuffle=True, 
                              num_workers=global_config["num_workers_train_dataloader"])
    val_loader = DataLoader(val_ds, 
                            batch_size=global_config["batch_size_val"], 
                            shuffle=False, 
                            num_workers=global_config["num_workers_val_dataloader"])
    
    print("The data loaders are built successfully.")
    print(gap_sign*50)

    print("Start training...")

    val_per_epoch = global_config["val_per_epoch"]
    save_per_epoch = global_config["save_per_epoch"]
    plot_per_epoch = global_config["plot_per_epoch"]
    best_val_loss = 1e6

    for idx_epoch in range(global_config["num_epoch"]):
        
        # training
        model.train()

        epoch_loss_train = {
            "reconL1": [],
        }
        for idx_batch, batch in enumerate(train_loader):
            y = batch["CT"].to(device)
            x = batch["PET_raw"].to(device)
            # if there are other modalities, concatenate them at the channel dimension
            for modality in input_modality:
                if modality != "PET_raw":
                    x = torch.cat((x, batch[modality].to(device)), dim=1)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = F.l1_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss_train["reconL1"].append(loss.item())
            print(f"<{idx_epoch}> [{idx_batch}/{num_train_files}] Total loss: {loss.item()}")
        
        for key in epoch_loss_train.keys():
            epoch_loss_train[key] = np.asanyarray(epoch_loss_train[key])
            logger.log(idx_epoch, f"train_{key}_mean", epoch_loss_train[key].mean())
        
        # validation
        if idx_epoch % val_per_epoch == 0:
            model.eval()
            epoch_loss_val = {
                "reconL1": [],
            }
            with torch.no_grad():
                for idx_batch, batch in enumerate(val_loader):
                    y = batch["CT"].to(device)
                    x = batch["PET_raw"].to(device)
                    # if there are other modalities, concatenate them at the channel dimension
                    for modality in input_modality:
                        if modality != "PET_raw":
                            x = torch.cat((x, batch[modality].to(device)), dim=1)
                    y_pred = model(x)
                    loss = F.l1_loss(y_pred, y)
                    epoch_loss_val["reconL1"].append(loss.item())
                    print(f"<{idx_epoch}> [{idx_batch}/{num_val_files}] Total loss: {loss.item()}")
            
            for key in epoch_loss_val.keys():
                epoch_loss_val[key] = np.asanyarray(epoch_loss_val[key])
                logger.log(idx_epoch, f"val_{key}_mean", epoch_loss_val[key].mean())
            
            current_val_loss = epoch_loss_val["reconL1"].mean()
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                model_save_name = global_config["save_folder"]+f"model_best_{idx_epoch}_state_dict.pth"
                optimizer_save_name = global_config["save_folder"]+f"optimizer_best_{idx_epoch}_state_dict.pth"
                torch.save(model.state_dict(), model_save_name)
                torch.save(optimizer.state_dict(), optimizer_save_name)
                logger.log(idx_epoch, "best_val_loss", best_val_loss)
            
        # save the model every save_per_epoch
        if idx_epoch % save_per_epoch == 0:
            # delete previous model
            for f in glob.glob(global_config["save_folder"]+"latest_*"):
                os.remove(f)
            model_save_name = global_config["save_folder"]+f"latest_model_{idx_epoch}_state_dict.pth"
            optimizer_save_name = global_config["save_folder"]+f"latest_optimizer_{idx_epoch}_state_dict.pth"
            torch.save(model.state_dict(), model_save_name)
            torch.save(optimizer.state_dict(), optimizer_save_name)
            logger.log(idx_epoch, "model_saved", f"model_{idx_epoch}_state_dict.pth")
        
        # plot the PET and CT every plot_per_epoch
        if idx_epoch % plot_per_epoch == 0:
            plot_and_save_x_y_z(x=x[:, 0, :, :, :],
                                y=y, 
                                z=y_pred, 
                                num_per_direction=3, 
                                savename=None)

            
if __name__ == "__main__":
    main()