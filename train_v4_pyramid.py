"""
This is changed from the orginal code and modified by ChatGPT from
https://github.com/thuanz123/enhancing-transformers/blob/1778fc497ea11ed2cef134404f99d4d6b921cda9/enhancing/modules/stage1/layers.py
"""

# This is for conv layers, using the pyramidal structure
# 


# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os

# Define the base cache directory
base_cache_dir = './cache'

# Define and create necessary subdirectories within the base cache directory
cache_dirs = {
    'WANDB_DIR': os.path.join(base_cache_dir, 'wandb'),
    'WANDB_CACHE_DIR': os.path.join(base_cache_dir, 'wandb_cache'),
    'WANDB_CONFIG_DIR': os.path.join(base_cache_dir, 'config'),
    'WANDB_DATA_DIR': os.path.join(base_cache_dir, 'data'),
    'TRANSFORMERS_CACHE': os.path.join(base_cache_dir, 'transformers'),
    'MPLCONFIGDIR': os.path.join(base_cache_dir, 'mplconfig')
}

# Create the base cache directory if it doesn't exist
os.makedirs(base_cache_dir, exist_ok=True)

# Create the necessary subdirectories and set the environment variables
for key, path in cache_dirs.items():
    os.makedirs(path, exist_ok=True)
    os.environ[key] = path

# set the environment variable to use the GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["OMP_THREAD_LIMIT"] = "1"

import wandb

wandb.login(key = "41c33ee621453a8afcc7b208674132e0e8bfafdb")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is: ", device)

# import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import time
import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import pytorch_lightning as pl

from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence
from torch.optim import lr_scheduler

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

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
    # random flip and rotate
    RandFlipd,
    RandRotated,
)

from vector_quantize_pytorch import VectorQuantize as lucidrains_VQ

random_seed = 729
volume_size = 64
pix_dim = 1.5
num_workers_train_dataloader = 8
num_workers_val_dataloader = 4
num_workers_train_cache_dataset = 8
num_workers_val_cache_dataset = 4
batch_size_train = 32
batch_size_val = 16
cache_ratio_train = 0.2
cache_ratio_val = 0.2
IS_LOGGER_WANDB = True

val_per_epoch = 25
save_per_epoch = 100

# set random seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

pyramid_channels = [32, 64, 128, 256]
pyramid_codebook_size = [32, 64, 128, 256]
pyramid_strides = [2, 2, 2, 1]
pyramid_num_res_units = [3, 4, 5, 6]
pyramid_num_epoch = [100, 200, 400, 800]
pyramid_freeze_previous_stages = False

VQ_optimizer = "AdamW"
VQ_optimizer_lr = 1e-4
VQ_optimizer_weight_decay = 5e-5

VQ_loss_weight_recon_L2 = 1.0
VQ_loss_weight_recon_L1 = 0.1
VQ_loss_weight_codebook = 0.1

VQ_train_epoch = 1000
VQ_train_gradiernt_clip = 1.0

model_message = "this is the first try to use cascaded VQ-VAE model of image pyramid"


wandb.init(
    # set the wandb project where this run will be logged
    project="CT_ViT_VQGAN",

    # # set the cache directory to the local cache directory
    dir=os.getenv("WANDB_DIR", "cache/wandb"),

    # track hyperparameters and run metadata
    config={
        "volume_size": volume_size,
        "pix_dim": pix_dim,
        "num_workers_train_dataloader": num_workers_train_dataloader,
        "num_workers_val_dataloader": num_workers_val_dataloader,
        "num_workers_train_cache_dataset": num_workers_train_cache_dataset,
        "num_workers_val_cache_dataset": num_workers_val_cache_dataset,
        "batch_size_train": batch_size_train,
        "batch_size_val": batch_size_val,
        "cache_ratio_train": cache_ratio_train,
        "cache_ratio_val": cache_ratio_val,
        "VQ_optimizer": VQ_optimizer,
        "VQ_optimizer_lr": VQ_optimizer_lr,
        "VQ_optimizer_weight_decay": VQ_optimizer_weight_decay,
        "VQ_loss_weight_recon_L2": VQ_loss_weight_recon_L2,
        "VQ_loss_weight_recon_L1": VQ_loss_weight_recon_L1,
        "VQ_loss_weight_codebook": VQ_loss_weight_codebook,
        "VQ_train_epoch": VQ_train_epoch,
        "VQ_train_gradiernt_clip": VQ_train_gradiernt_clip,
        "pyramid_channels": pyramid_channels,
        "pyramid_codebook_size": pyramid_codebook_size,
        "pyramid_strides": pyramid_strides,
        "pyramid_num_res_units": pyramid_num_res_units,
        "pyramid_num_epoch": pyramid_num_epoch,
        "pyramid_freeze_previous_stages": pyramid_freeze_previous_stages,
        "model_message": model_message,
    }
)

class UNet3D_encoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()
        
        self.dimensions = spatial_dims
        self.in_channels = in_channels
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

        # input - down1 ------------- up1 -- output
        #         |                   |
        #         down2 ------------- up2
        #         |                   |
        #         down3 ------------- up3
        # 1 -> (32, 64, 128, 256) -> 1

        self.depth = len(self.channels)
        self.down_blocks = nn.ModuleList()


        for i in range(self.depth):
            self.down_blocks.append(
                ResidualUnit(3, self.in_channels, self.channels[i], self.strides[i],
                    kernel_size=self.kernel_size, subunits=self.num_res_units,
                    act=self.act, norm=self.norm, dropout=self.dropout,
                    bias=self.bias, adn_ordering=self.adn_ordering)
            )
            self.in_channels = self.channels[i]

        # flatten from (B, C, H, W, D) to (B, C, H*W*D), C is the self.channels[2]
        self.flatten = nn.Sequential(
            Rearrange('b c h w d -> b (h w d) c'),
        )

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.depth):
            x = self.down_blocks[i](x)
        
        x = self.flatten(x)

        return x
    

class UNet3D_decoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        hwd: Union[Tuple, str] = 8,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:

        super().__init__()

        self.dimensions = spatial_dims
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.hwd = hwd
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # input - down1 ------------- up1 -- output
        #         |                   |
        #         down2 ------------- up2
        #         |                   |
        #         down3 ------------- up3
        # 1 -> (32, 64, 128, 256) -> 1
        
        # take the cubic root of the second element of the tuple
        self.unflatten = nn.Sequential(
            Rearrange('b (h w d) c -> b c h w d', h=self.hwd, w=self.hwd, d=self.hwd),
        )

        self.depth = len(self.channels)
        self.up = nn.ModuleList()
        for i in range(self.depth - 1):
            self.up.append(
                nn.Sequential(
                    Convolution(3, self.channels[i], self.channels[i+1], self.strides[i], self.up_kernel_size,
                        act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, conv_only=False,
                        is_transposed=True, adn_ordering=self.adn_ordering),
                    ResidualUnit(3, self.channels[i+1], self.channels[i+1], 1, self.kernel_size, self.num_res_units,
                        act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, last_conv_only=False,
                        adn_ordering=self.adn_ordering)
                )
            )
        self.out = nn.Conv3d(self.channels[-1], self.out_channels, kernel_size=1, stride=1, padding=0)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unflatten(x)
        for i in range(self.depth - 1):
            x = self.up[i](x)
        x = self.out(x)
        return x



# class ViTVQ3D(pl.LightningModule):
# here we will receive a list, which contains several set of encoder/pre_quant/quantizer/post_quant/decoder
# we will read the list and create the model

class ViTVQ3D(nn.Module):
    def __init__(self, model_level: list, device: torch.device) -> None:
        super().__init__()
        self.num_level = len(model_level)
        self.sub_models = nn.ModuleList()
        self.device = device
        for level_setting in model_level:
            # Create a submodule to hold the encoder, decoder, quantizer, etc.
            sub_model = nn.Module() 
            sub_model.encoder = UNet3D_encoder(**level_setting["encoder"])
            sub_model.decoder = UNet3D_decoder(**level_setting["decoder"])
            sub_model.quantizer = lucidrains_VQ(**level_setting["quantizer"])
            sub_model.pre_quant = nn.Linear(level_setting["encoder"]["channels"][-1], level_setting["quantizer"]["dim"])
            sub_model.post_quant = nn.Linear(level_setting["quantizer"]["dim"], level_setting["decoder"]["channels"][0])
            
            # Append the submodule to the ModuleList
            self.sub_models.append(sub_model) 
        
        self.init_weights()
        self.freeze_gradient_all()

    def move_to_device_at_level(self, i_level: int) -> None:
        self.sub_models[i_level].encoder.to(self.device)
        self.sub_models[i_level].decoder.to(self.device)
        self.sub_models[i_level].quantizer.to(self.device)
        self.sub_models[i_level].pre_quant.to(self.device)
        self.sub_models[i_level].post_quant.to(self.device)
        print(f"Move to device at level {i_level}")

    def freeze_gradient_all(self) -> None:
        for level in range(self.num_level):
            self.freeze_gradient_at_level(level)
        print("Freeze all gradients")

    def freeze_gradient_at_level(self, i_level: int) -> None:
        self.sub_models[i_level].encoder.requires_grad_(False)
        self.sub_models[i_level].decoder.requires_grad_(False)
        self.sub_models[i_level].quantizer.requires_grad_(False)
        self.sub_models[i_level].pre_quant.requires_grad_(False)
        self.sub_models[i_level].post_quant.requires_grad_(False)
        print(f"Freeze gradient at level {i_level}")

    def unfreeze_gradient_at_level(self, i_level: int) -> None:
        self.sub_models[i_level].encoder.requires_grad_(True)
        self.sub_models[i_level].decoder.requires_grad_(True)
        self.sub_models[i_level].quantizer.requires_grad_(True)
        self.sub_models[i_level].pre_quant.requires_grad_(True)
        self.sub_models[i_level].post_quant.requires_grad_(True)
        print(f"Unfreeze gradient at level {i_level}")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def foward_at_level(self, x: torch.FloatTensor, i_level: int) -> torch.FloatTensor:
        h = self.sub_models[i_level].encoder(x) # Access using dot notation
        h = self.sub_models[i_level].pre_quant(h)
        quant, indices, loss = self.sub_models[i_level].quantizer(h)
        g = self.sub_models[i_level].post_quant(quant)
        g = self.sub_models[i_level].decoder(g)
        return g, indices, loss

    def forward(self, pyramid_x: list, active_level: int) -> torch.FloatTensor:
        # pyramid_x is a list of tensorFloat, like [8*8*8, 16*16*16, 32*32*32, 64*64*64]
        # active_level is the level of the pyramid, like 0, 1, 2, 3
        
        assert active_level <= self.num_level
        x_hat = None
        indices_list = []
        loss_list = []

        for i_level in range(active_level):
            if i_level == 0:
                x_hat, indices, loss = self.foward_at_level(pyramid_x[i_level], i_level)
                indices_list.append(indices.cpu())
                loss_list.append(loss.cpu())
            else:
                resample_x = F.interpolate(pyramid_x[i_level - 1], scale_factor=2, mode='trilinear', align_corners=False)
                input_x = pyramid_x[i_level] - resample_x
                output_x, indices, loss = self.foward_at_level(input_x, i_level)
                indices_list.append(indices.cpu())
                loss_list.append(loss.cpu())
                # upsample the x_hat to double the size in three dimensions
                x_hat = F.interpolate(x_hat, scale_factor=2, mode='trilinear', align_corners=False)
                x_hat = x_hat + output_x

        # x_hat is in cuda
        # indices_list and loss_list are in cpu
        return x_hat, indices_list, loss_list


train_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(pix_dim, pix_dim, pix_dim),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=2976,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # CropForegroundd(keys=["image"], source_key="image"),
        # random crop to the target size
        RandSpatialCropd(keys=["image"], roi_size=(volume_size, volume_size, volume_size), random_center=True, random_size=False),
        # add random flip and rotate
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        RandRotated(keys=["image"], prob=0.5, range_x=15, range_y=15, range_z=15),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(pix_dim, pix_dim, pix_dim),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=2976, b_min=0.0, b_max=1.0, clip=True),
        # CropForegroundd(keys=["image"], source_key="image"),
        RandSpatialCropd(keys=["image"], roi_size=(volume_size, volume_size, volume_size), random_center=True, random_size=False),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    ]
)

# load data_chunks.json and specif chunk_0 to chunk_4 for training, chunk_5 to chunk_7 for validation, chunk_8 and chunk_9 for testing
with open("data_chunks_80.json", "r") as f:
    data_chunk = json.load(f)

train_files = []
val_files = []
test_files = []

for i in range(5):
    train_files.extend(data_chunk[f"chunk_{i}"])
for i in range(5, 8):
    val_files.extend(data_chunk[f"chunk_{i}"])
for i in range(8, 10):
    test_files.extend(data_chunk[f"chunk_{i}"])

num_train_files = len(train_files)
num_val_files = len(val_files)
num_test_files = len(test_files)

print("Train files are ", len(train_files))
print("Val files are ", len(val_files))
print("Test files are ", len(test_files))


train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_num=num_train_files,
    cache_rate=cache_ratio_train, # 600 * 0.1 = 60
    num_workers=num_workers_train_cache_dataset,
)

val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms, 
    cache_num=num_val_files,
    cache_rate=cache_ratio_val, # 360 * 0.05 = 18
    num_workers=num_workers_val_cache_dataset,)



train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=num_workers_train_dataloader)
val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False, num_workers=num_workers_val_dataloader)




encoder_8 = {
    "spatial_dims": 3, "in_channels": 1,
    "channels": [pyramid_channels[0]],
    "strides": pyramid_strides[-1:],
    "num_res_units": pyramid_num_res_units[0],
}
quantizer_8 = {
    "dim": pyramid_channels[0], 
    "codebook_size": pyramid_codebook_size[0],
    "decay": 0.8, "commitment_weight": 1.0,
    "use_cosine_sim": True, "threshold_ema_dead_code": 2,
    "kmeans_init": True, "kmeans_iters": 10
}
decoder_8 = {
    "spatial_dims": 3, "out_channels": 1,
    "channels": [pyramid_channels[0]],
    "strides": pyramid_strides[-1:],
    "num_res_units": pyramid_num_res_units[0],
}

encoder_16 = {
    "spatial_dims": 3, "in_channels": 1,
    "channels": pyramid_channels[0:2],
    "strides": pyramid_strides[-2:],
    "num_res_units": pyramid_num_res_units[1],
}
quantizer_16 = {
    "dim": pyramid_channels[1], 
    "codebook_size": pyramid_codebook_size[1],
    "decay": 0.8, "commitment_weight": 1.0,
    "use_cosine_sim": True, "threshold_ema_dead_code": 2,
    "kmeans_init": True, "kmeans_iters": 10
}
decoder_16 = {
    "spatial_dims": 3, "out_channels": 1,
    "channels": pyramid_channels[0:2],
    "strides": pyramid_strides[-2:],
    "num_res_units": pyramid_num_res_units[1],
}

encoder_32 = {
    "spatial_dims": 3, "in_channels": 1,
    "channels": pyramid_channels[0:3],
    "strides": pyramid_strides[-3:],
    "num_res_units": pyramid_num_res_units[2],
}
quantizer_32 = {
    "dim": pyramid_channels[2], 
    "codebook_size": pyramid_codebook_size[2],
    "decay": 0.8, "commitment_weight": 1.0,
    "use_cosine_sim": True, "threshold_ema_dead_code": 2,
    "kmeans_init": True, "kmeans_iters": 10
}
decoder_32 = {
    "spatial_dims": 3, "out_channels": 1,
    "channels": pyramid_channels[0:3],
    "strides": pyramid_strides[-3:],
    "num_res_units": pyramid_num_res_units[2],
}

encoder_64 = {
    "spatial_dims": 3, "in_channels": 1,
    "channels": pyramid_channels[0:4],
    "strides": pyramid_strides[-4:],
    "num_res_units": pyramid_num_res_units[3],
}
quantizer_64 = {
    "dim": pyramid_channels[3],
    "codebook_size": pyramid_codebook_size[3],
    "decay": 0.8, "commitment_weight": 1.0,
    "use_cosine_sim": True, "threshold_ema_dead_code": 2,
    "kmeans_init": True, "kmeans_iters": 10
}
decoder_64 = {
    "spatial_dims": 3, "out_channels": 1,
    "channels": pyramid_channels[0:4],
    "strides": pyramid_strides[-4:],
    "num_res_units": pyramid_num_res_units[3],
}

model = ViTVQ3D(
    model_level=[
        {
            "encoder": encoder_8,
            "decoder": decoder_8,
            "quantizer": quantizer_8
        },
        {
            "encoder": encoder_16,
            "decoder": decoder_16,
            "quantizer": quantizer_16
        },
        {
            "encoder": encoder_32,
            "decoder": decoder_32,
            "quantizer": quantizer_32
        },
        {
            "encoder": encoder_64,
            "decoder": decoder_64,
            "quantizer": quantizer_64
        },
    ],
    device=device,
)


learning_rate = VQ_optimizer_lr
# use AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=VQ_optimizer_weight_decay)

# here we set the encoder model parameters

# total_dist = preceptual_model(nii_data_norm_cut_tensor, out)
# print("total_dist is ", total_dist)

class simple_logger():
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_dict = dict()
    
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
        if IS_LOGGER_WANDB and isinstance(msg, (int, float)):
            wandb.log({key: msg})

def plot_and_save_x_xrec(x, xrec, num_per_direction=1, savename=None, wandb_name="val_snapshots"):
    numpy_x = x[0, :, :, :, :].cpu().numpy().squeeze()
    numpy_xrec = xrec[0, :, :, :, :].cpu().numpy().squeeze()
    x_clip = np.clip(numpy_x, 0, 1)
    rec_clip = np.clip(numpy_xrec, 0, 1)
    fig_width = num_per_direction * 3
    fig_height = 4
    fig, axs = plt.subplots(3, fig_width, figsize=(fig_width, fig_height), dpi=100)
    # for axial
    for i in range(num_per_direction):
        img_x = x_clip[x_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        img_rec = rec_clip[rec_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        axs[0, 3*i].imshow(img_x, cmap="gray")
        axs[0, 3*i].set_title(f"A x {x_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i].axis("off")
        axs[1, 3*i].imshow(img_rec, cmap="gray")
        axs[1, 3*i].set_title(f"A xrec {rec_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i].axis("off")
        axs[2, 3*i].imshow(img_x - img_rec, cmap="bwr")
        axs[2, 3*i].set_title(f"A diff {rec_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i].axis("off")
    # for sagittal
    for i in range(num_per_direction):
        img_x = x_clip[:, :, x_clip.shape[2]//(num_per_direction+1)*(i+1)]
        img_rec = rec_clip[:, :, rec_clip.shape[2]//(num_per_direction+1)*(i+1)]
        axs[0, 3*i+1].imshow(img_x, cmap="gray")
        axs[0, 3*i+1].set_title(f"S x {x_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i+1].axis("off")
        axs[1, 3*i+1].imshow(img_rec, cmap="gray")
        axs[1, 3*i+1].set_title(f"S xrec {rec_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i+1].axis("off")
        axs[2, 3*i+1].imshow(img_x - img_rec, cmap="bwr")
        axs[2, 3*i+1].set_title(f"S diff {rec_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i+1].axis("off")

    # for coronal
    for i in range(num_per_direction):
        img_x = x_clip[:, x_clip.shape[1]//(num_per_direction+1)*(i+1), :]
        img_rec = rec_clip[:, rec_clip.shape[1]//(num_per_direction+1)*(i+1), :]
        axs[0, 3*i+2].imshow(img_x, cmap="gray")
        axs[0, 3*i+2].set_title(f"C x {x_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i+2].axis("off")
        axs[1, 3*i+2].imshow(img_rec, cmap="gray")
        axs[1, 3*i+2].set_title(f"C xrec {rec_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i+2].axis("off")
        axs[2, 3*i+2].imshow(img_x - img_rec, cmap="bwr")
        axs[2, 3*i+2].set_title(f"C diff {rec_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i+2].axis("off")

    plt.tight_layout()
    plt.savefig(savename)
    wandb.log({wandb_name: fig})
    plt.close()
    print(f"Save the plot to {savename}")

def compute_average_gradient(model_part):
    total_grad = 0.0
    num_params = 0
    for param in model_part.parameters():
        if param.grad is not None:
            total_grad += param.grad.abs().mean().item()
            num_params += 1
    return total_grad / num_params if num_params > 0 else 0.0

# Effective Number of Classes
def effective_number_of_classes(probabilities):
    return 1 / np.sum(probabilities ** 2)


current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
log_file_path = f"train_log_{current_time}.json"
logger = simple_logger(log_file_path)

num_epoch = VQ_train_epoch
loss_weights = {
    "reconL2": VQ_loss_weight_recon_L2,
    "reconL1": VQ_loss_weight_recon_L1,
    "codebook": VQ_loss_weight_codebook,
}

num_train_batch = len(train_loader)
num_val_batch = len(val_loader)
best_val_loss = 1e6
save_folder = "./results/"
# create the folder if not exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def generate_input_data_pyramid(x, levels):
    pyramid_x = []
    if levels == 1:
        x_8 = F.interpolate(x, size=(8, 8, 8), mode="trilinear", align_corners=False).to(device)
        pyramid_x.append(x_8)
    if levels == 2:
        x_16 = F.interpolate(x, size=(16, 16, 16), mode="trilinear", align_corners=False).to(device)
        pyramid_x.append(x_16)
    if levels == 3:
        x_32 = F.interpolate(x, size=(32, 32, 32), mode="trilinear", align_corners=False).to(device)
        pyramid_x.append(x_32)
    if levels == 4:
        pyramid_x.append(x)
    return pyramid_x

def train_model_at_level(num_epoch, current_level):

    # move the model to the device
    for i_level in range(current_level):
        model.move_to_device_at_level(i_level)

    # set the gradient freeze
    if pyramid_freeze_previous_stages:
        model.freeze_gradient_all()
        model.unfreeze_gradient_at_level(current_level)
    else:
        model.freeze_gradient_all()
        for i_level in range(current_level):
            model.unfreeze_gradient_at_level(i_level)

    # start the training
    for idx_epoch in range(num_epoch):
        model.train()
        epoch_loss_train = {
            "reconL2": [],
            "reconL1": [],
            "codebook": [],
            "total": [],
        }
        epoch_codebook_train = {
            "indices": [],
        }

        for idx_batch, batch in enumerate(train_loader):
            x = batch["image"]
            # generate the input data pyramid
            pyramid_x = generate_input_data_pyramid(x, current_level)
            # target_x is the last element of the pyramid_x, which is to be reconstructed
            target_x = pyramid_x[-1]
            # input the pyramid_x to the model
            xrec, indices_list, cb_loss_list = model(pyramid_x, current_level)
            # initialize the optimizer
            optimizer.zero_grad()
            # compute the loss
            reconL2_loss = F.mse_loss(target_x, xrec)
            reconL1_loss = F.l1_loss(target_x, xrec)
            if pyramid_freeze_previous_stages:
                codebook_loss = cb_loss_list[-1]
            else:
                # average the codebook loss
                codebook_loss = np.asanyarray(cb_loss_list).mean()
            # take the weighted sum of the loss
            total_loss = loss_weights["reconL2"] * reconL2_loss + \
                            loss_weights["reconL1"] * reconL1_loss + \
                            loss_weights["codebook"] * codebook_loss
            # record the loss
            epoch_loss_train["reconL2"].append(reconL2_loss.item())
            epoch_loss_train["reconL1"].append(reconL1_loss.item())
            epoch_loss_train["codebook"].append(codebook_loss.item())
            epoch_loss_train["total"].append(total_loss.item())
            # print the loss
            print(f"<{idx_epoch}> [{idx_batch}/{num_train_batch}] Total loss: {total_loss.item()}")
            # add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=VQ_train_gradiernt_clip)
            total_loss.backward()
            optimizer.step()

            # record the codebook indices
            if pyramid_freeze_previous_stages:
                epoch_codebook_train["indices"].extend(indices_list[-1].numpy().squeeze().flatten())
            else:
                for current_indices in indices_list:
                    epoch_codebook_train["indices"].extend(current_indices.numpy().squeeze().flatten())
        
        for key in epoch_loss_train.keys():
            epoch_loss_train[key] = np.asanyarray(epoch_loss_train[key])
            logger.log(idx_epoch, f"train_{key}_mean", epoch_loss_train[key].mean())
            # logger.log(idx_epoch, f"train_{key}_std", epoch_loss_train[key].std())
        
        
        for key in epoch_codebook_train.keys():
            epoch_codebook_train[key] = np.asanyarray(epoch_codebook_train[key])
        
        activated_value, activated_counts = np.unique(epoch_codebook_train["indices"], return_counts=True)
        if len(activated_counts) < pyramid_codebook_size[current_level]:
            activated_counts = np.append(activated_counts, np.zeros(pyramid_codebook_size[current_level] - len(activated_counts)))
        effective_num = effective_number_of_classes(activated_counts / np.sum(activated_counts))
        embedding_num = len(activated_counts)
        logger.log(idx_epoch, "train_effective_num", effective_num)
        logger.log(idx_epoch, "train_embedding_num", embedding_num)
        

        # validation
        if idx_epoch % val_per_epoch == 0:
            model.eval()
            epoch_loss_val = {
                "reconL2": [],
                "reconL1": [],
                "codebook": [],
                "total": [],
            }
            epoch_codebook_val = {
                "indices": [],
            }
            with torch.no_grad():
                for idx_batch, batch in enumerate(val_loader):
                    x = batch["image"]
                    # generate the input data pyramid
                    pyramid_x = generate_input_data_pyramid(x, current_level)
                    # target_x is the last element of the pyramid_x, which is to be reconstructed
                    target_x = pyramid_x[-1]
                    xrec, indices_list, cb_loss_list = model(x)
                    # compute the loss
                    reconL2_loss = F.mse_loss(target_x, xrec)
                    reconL1_loss = F.l1_loss(target_x, xrec)
                    if pyramid_freeze_previous_stages:
                        codebook_loss = cb_loss_list[-1]
                    else:
                        # average the codebook loss
                        codebook_loss = np.asanyarray(cb_loss_list).mean()
                    # take the weighted sum of the loss
                    total_loss = loss_weights["reconL2"] * reconL2_loss + \
                                    loss_weights["reconL1"] * reconL1_loss + \
                                    loss_weights["codebook"] * codebook_loss
                    epoch_loss_val["reconL2"].append(reconL2_loss.item())
                    epoch_loss_val["reconL1"].append(reconL1_loss.item())
                    epoch_loss_val["codebook"].append(codebook_loss.item())
                    epoch_loss_val["total"].append(total_loss.item())
                    print(f"<{idx_epoch}> [{idx_batch}/{num_val_batch}] Total loss: {total_loss.item()}")

                    if pyramid_freeze_previous_stages:
                        epoch_codebook_val["indices"].extend(indices_list[-1].numpy().squeeze().flatten())
                    else:
                        for current_indices in indices_list:
                            epoch_codebook_val["indices"].extend(current_indices.numpy().squeeze().flatten())

            save_name = f"epoch_{idx_epoch}_batch_{idx_batch}"
            plot_and_save_x_xrec(x, xrec, num_per_direction=3, savename=save_folder+f"{save_name}_{current_level}.png", wandb_name="val_snapshots")
            
            for key in epoch_loss_val.keys():
                epoch_loss_val[key] = np.asanyarray(epoch_loss_val[key])
                logger.log(idx_epoch, f"val_{key}_mean", epoch_loss_val[key].mean())
                # logger.log(idx_epoch, f"val_{key}_std", epoch_loss_val[key].std())

            if epoch_loss_val["total"].mean() < best_val_loss:
                best_val_loss = epoch_loss_val["total"].mean()
                torch.save(model.state_dict(), save_folder+f"model_best_{idx_epoch}_state_dict_{current_level}.pth")
                torch.save(optimizer.state_dict(), save_folder+f"optimizer_best_{idx_epoch}_state_dict_{current_level}.pth")
                logger.log(idx_epoch, "best_val_loss", best_val_loss)

            for key in epoch_codebook_val.keys():
                epoch_codebook_val[key] = np.asanyarray(epoch_codebook_val[key])
            
            activated_value, activated_counts = np.unique(epoch_codebook_val["indices"], return_counts=True)
            if len(activated_counts) < pyramid_codebook_size[current_level]:
                activated_counts = np.append(activated_counts, np.zeros(pyramid_codebook_size[current_level] - len(activated_counts)))
            effective_num = effective_number_of_classes(activated_counts / np.sum(activated_counts))
            embedding_num = len(activated_counts)
            logger.log(idx_epoch, "val_effective_num", effective_num)
            logger.log(idx_epoch, "val_embedding_num", embedding_num)
        
        # save the model every save_per_epoch
        if idx_epoch % save_per_epoch == 0:
            torch.save(model.state_dict(), save_folder+f"model_{idx_epoch}_state_dict_{current_level}.pth")
            torch.save(optimizer.state_dict(), save_folder+f"optimizer_{idx_epoch}_state_dict_{current_level}.pth")
            logger.log(idx_epoch, "model_saved", f"model_{idx_epoch}_state_dict.pth")
            
for current_level in range(4):
    # current level starts at 1
    train_model_at_level(pyramid_num_epoch[current_level], current_level + 1)

wandb.finish()