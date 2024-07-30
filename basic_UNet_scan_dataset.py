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

import os
import time
import torch
# set the device to be GPU:1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    CacheDataset,
)

from torch.utils.data import Dataset

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotated,
)

from monai.transforms import LoadImage, Compose, EnsureChannelFirst

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export

# from collections import Sequence
from typing import Sequence, Union
import warnings

class monai_UNet(nn.Module):
# class UNet(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[tuple, str] = Act.PRELU,
        norm: Union[tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)
        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if self.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)
                if self.bias:
                    nn.init.constant_(m.bias, 0)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


def plot_and_save_x_y_z(x, y, z, num_per_direction=1, savename=None):
    numpy_x = x[0, 0, :, :, :].detach().cpu().numpy().squeeze()
    numpy_y = y[0, :, :, :, :].detach().cpu().numpy().squeeze()
    numpy_z = z[0, :, :, :, :].detach().cpu().numpy().squeeze()
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
        axs[2, 3*i+2].imshow(img_z, cmap="gray")
        axs[2, 3*i+2].set_title(f"C z {z_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i+2].axis("off")

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()
    print(f"Save the plot to {savename}")

# def custom_collate_fn(batch):
#     collated_batch = {}
#     keys = batch[0].keys()
    
#     for key in keys:
#         if isinstance(batch[0][key], torch.Tensor):
#             collated_batch[key] = torch.stack([item[key] for item in batch])
#         elif isinstance(batch[0][key], dict):
#             collated_batch[key] = [{subkey: item[key][subkey] for subkey in item[key]} for item in batch]
#         else:
#             collated_batch[key] = [item[key] for item in batch]
    
#     return collated_batch

def collate_fn(batch, pet_valid_th=0.01):
    # Flatten the list of lists into a single list of samples
    batch = [item for sublist in batch for item in sublist]
    valid_samples = [sample for sample in batch if sample["PET"].mean() > pet_valid_th]
    if not valid_samples:
        return None
    valid_data = {key: torch.stack([sample[key] for sample in valid_samples]) for key in valid_samples[0]}
    return valid_data

class local_logger():
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
    
    def log(self, epoch, key, value):
        self.log_file = open(self.log_file_path, 'w')
        self.log_file.write(f"{epoch}, {key}, {value}\n")
        print(f"{epoch}, {key}, {value}")
    
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
    save_folder = global_config["save_folder"]
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    global_config["save_folder"] = f"{save_folder}/{time_stamp}"
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
    log_file_path = global_config["save_folder"]+f"train_log_{current_time}.txt"
    logger = local_logger(log_file_path)
    print("The log file path is: ", log_file_path)

    # set the input modality
    input_modality = global_config["input_modality"]
    in_channels = len(input_modality) - 1
    print("The input modality is: ", input_modality)

    # set the model
    model = monai_UNet(
        spatial_dims = global_config["spatial_dims"],
        in_channels = in_channels,
        out_channels = global_config["out_channels"],
        channels = global_config["channels"],
        strides = global_config["strides"],
        num_res_units = global_config["num_res_units"],
    ).to(device)

    # initialize the model


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
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality),
            # Orientationd(keys=input_modality, axcodes="RAS"),
            # RandSpatialCropd(keys=input_modality, 
            #                  roi_size=(volume_size, volume_size, volume_size), 
            #                  random_center=True, random_size=False,
            #                  num_samples=global_config["batches_from_each_nii"]),
            RandSpatialCropSamplesd(keys=input_modality,
                                    roi_size=(volume_size, volume_size, volume_size),
                                    num_samples=global_config["batches_from_each_nii"]),
            RandFlipd(keys=input_modality, prob=0.5, spatial_axis=0),
            RandFlipd(keys=input_modality, prob=0.5, spatial_axis=1),
            RandFlipd(keys=input_modality, prob=0.5, spatial_axis=2),
            RandRotated(keys=input_modality, prob=0.5, range_x=15, range_y=15, range_z=15),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=input_modality),
            EnsureChannelFirstd(keys=input_modality),
            # Orientationd(keys=input_modality, axcodes="RAS"),
            # RandSpatialCropd(keys=input_modality, 
            #                  roi_size=(volume_size, volume_size, volume_size), 
            #                  random_center=True, random_size=False,
            #                  num_samples=global_config["batches_from_each_nii"]),
            RandSpatialCropSamplesd(keys=input_modality,
                        roi_size=(volume_size, volume_size, volume_size),
                        num_samples=global_config["batches_from_each_nii"]),
        ]
    )

    data_division_file = global_config["data_division"]
    # load data_chunks.json and specif chunk_0 to chunk_4 for training, chunk_5 to chunk_7 for validation, chunk_8 and chunk_9 for testing
    with open(data_division_file, "r") as f:
        data_chunk = json.load(f)

    train_files = []
    val_files = []
    test_files = []

    chunk_train = global_config["chunk_train"]
    chunk_val = global_config["chunk_val"]
    chunk_test = global_config["chunk_test"]
    # if chunk is int, convert it to list
    if isinstance(chunk_train, int):
        chunk_train = [chunk_train]
    if isinstance(chunk_val, int):
        chunk_val = [chunk_val]
    if isinstance(chunk_test, int):
        chunk_test = [chunk_test]

    for i in chunk_train:
        train_files.extend(data_chunk[f"chunk_{i}"])
    for i in chunk_val:
        val_files.extend(data_chunk[f"chunk_{i}"])
    for i in chunk_test:
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
                              num_workers=global_config["num_workers_train_dataloader"],
                              collate_fn=collate_fn
                              )
    val_loader = DataLoader(val_ds, 
                            batch_size=global_config["batch_size_val"], 
                            shuffle=False, 
                            num_workers=global_config["num_workers_val_dataloader"],
                            collate_fn=collate_fn
                            )
    
    print("The data loaders are built successfully.")
    print(gap_sign*50)

    print("Start training...")

    val_per_epoch = global_config["val_per_epoch"]
    save_per_epoch = global_config["save_per_epoch"]
    plot_per_epoch = global_config["plot_per_epoch"]
    best_val_loss = 1e6
    PET_valid_th = global_config["PET_valid_th"]

    for idx_epoch in range(global_config["num_epoch"]):
        
        # training
        model.train()

        epoch_loss_train = {
            "reconL1": [],
        }
        for idx_batch, batch in enumerate(train_loader):

            # skip the batch if it is None
            if batch is None:
                continue

            # print("Currently loading the batch named: ", batch["filename"])
            y = batch["CT"].to(device)
            x = batch["PET_raw"].to(device)
            
            # if there are other modalities, concatenate them at the channel dimension
            for modality in input_modality:
                if modality != "PET_raw" and modality != "CT":
                    x = torch.cat((x, batch[modality].to(device)), dim=1)
            # for modality in input_modality:
            #     print(f"{modality}_path:", batch[modality+"_meta_dict"]["filename_or_obj"])
            
            # print(f"Train <{idx_epoch}> [{idx_batch}] x: {x.shape} at device {x.device}, y: {y.shape} at device {y.device}")
            optimizer.zero_grad()
            y_pred = model(x)
            loss = F.l1_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss_train["reconL1"].append(loss.item())
            print(f"Train <{idx_epoch}> [{idx_batch}] batch loss: {loss.item()} with {len(y)} valid samples.")
        
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
                    # skip the batch if it is None
                    if batch is None:
                        continue
                    y = batch["CT"].to(device)
                    x = batch["PET_raw"].to(device)
                    # if there are other modalities, concatenate them at the channel dimension
                    for modality in input_modality:
                        if modality != "PET_raw" and modality != "CT":
                            x = torch.cat((x, batch[modality].to(device)), dim=1)
                    y_pred = model(x)
                    loss = F.l1_loss(y_pred, y)
                    epoch_loss_val["reconL1"].append(loss.item())
                    print(f"Eval <{idx_epoch}> [{idx_batch}] Total loss: {loss.item()}")
            
            for key in epoch_loss_val.keys():
                epoch_loss_val[key] = np.asanyarray(epoch_loss_val[key])
                logger.log(idx_epoch, f"val_{key}_mean", epoch_loss_val[key].mean())
            
            current_val_loss = epoch_loss_val["reconL1"].mean()
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                model_save_name = global_config["save_folder"]+f"/model_best_{idx_epoch}_state_dict.pth"
                optimizer_save_name = global_config["save_folder"]+f"/optimizer_best_{idx_epoch}_state_dict.pth"
                torch.save(model.state_dict(), model_save_name)
                torch.save(optimizer.state_dict(), optimizer_save_name)
                logger.log(idx_epoch, "best_val_loss", best_val_loss)
            
        # save the model every save_per_epoch
        if idx_epoch % save_per_epoch == 0:
            # delete previous model
            for f in glob.glob(global_config["save_folder"]+"/latest_*"):
                os.remove(f)
            model_save_name = global_config["save_folder"]+f"/latest_model_{idx_epoch}_state_dict.pth"
            optimizer_save_name = global_config["save_folder"]+f"/latest_optimizer_{idx_epoch}_state_dict.pth"
            torch.save(model.state_dict(), model_save_name)
            torch.save(optimizer.state_dict(), optimizer_save_name)
            logger.log(idx_epoch, "model_saved", f"model_{idx_epoch}_state_dict.pth")
        
        # plot the PET and CT every plot_per_epoch
        # print(x.shape, y.shape, y_pred.shape)
        if idx_epoch % plot_per_epoch == 0:
            plot_and_save_x_y_z(x=x,
                                y=y, 
                                z=y_pred, 
                                num_per_direction=3, 
                                savename=f"{global_config['save_folder']}/plot_{idx_epoch:04}.png")

            
if __name__ == "__main__":
    main()