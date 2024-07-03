"""
This is changed from the orginal code and modified by ChatGPT from
https://github.com/thuanz123/enhancing-transformers/blob/1778fc497ea11ed2cef134404f99d4d6b921cda9/enhancing/modules/stage1/layers.py
"""

# This is for the simple UNet encoder decoder as the baseline


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
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import pytorch_lightning as pl

from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import List, Tuple, Dict, Any, Optional, Union
from torch.optim import lr_scheduler

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from typing import Optional, Sequence, Tuple, Union
from monai.networks.layers.factories import Act, Norm

from monai.data import (
    DataLoader,
    CacheDataset,
)

import torch
from torch.utils.data import DataLoader, Dataset, default_collate
import numpy as np
import random


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

# model = ViTVQ3D(
#     volume_key="volume", volume_size=volume_size, patch_size=8,
#     encoder={
#         "dim": 360, "depth": 6, "heads": 16, "mlp_dim": 1024, "channels": 1, "dim_head": 128
#     },
#     decoder={
#         "dim": 360, "depth": 6, "heads": 16, "mlp_dim": 1024, "channels": 1, "dim_head": 128
#     },
#     quantizer={
#         "embed_dim": 128, "n_embed": 1024, "beta": 0.25, "use_norm": True, "use_residual": False
#     }
# ).to(device)
# learning_rate = 5e-4
# # use AdamW optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# num_epoch = 1000
# loss_weights = {
#     "reconL2": 1.0, 
#     "reconL1": 0.1, 
#     "perceptual": 0.05, 
#     "codebook": 0.1}
# val_per_epoch = 20
# save_per_epoch = 20
# num_train_batch = len(train_loader)
# num_val_batch = len(val_loader)
# best_val_loss = 1e6

# VQ_patch_size = 8

# VQ_encoder_dim = 360
# VQ_encoder_depth = 6
# VQ_encoder_heads = 16
# VQ_encoder_mlp_dim = 1024
# VQ_encoder_dim_head = 128

# VQ_decoder_dim = 360
# VQ_decoder_depth = 6
# VQ_decoder_heads = 16
# VQ_decoder_mlp_dim = 1024
# VQ_decoder_dim_head = 128

VQ_encoder_channels = [256, 512, 1024]
VQ_encoder_num_res_units = 6

VQ_decoder_channels = [256, 512, 1024]
VQ_decoder_num_res_units = 6

# VQ_quantizer_embed_dim = 128
# VQ_quantizer_n_embed = 1024
# VQ_quantizer_beta = 0.25
# VQ_quantizer_use_norm = True
# VQ_quantizer_use_residual = False

# vanilla vq
VQ_lucidrains_VQ_type = "VectorQuantize"
VQ_lucidrains_VQ_embed_dim = 1024
VQ_lucidrains_VQ_n_embed = 512
VQ_lucidrains_VQ_decay = 0.8
VQ_lucidrains_VQ_commiment_weight = 1.0

# kmeans init
VQ_lucidrains_VQ_kmeans_init = True
VQ_lucidrains_VQ_kmeans_iters = 10

# low codebook dim
# VQ_lucidrains_VQ_codebook_dim = 16

# cosine similarity for spherical embedding
VQ_lucidrains_VQ_use_cosine_sim = True
# VQ_lucidrains_VQ_message = "cosine similarity for spherical embedding"

# expiring stale embeddings
VQ_lucidrains_threshold_ema_dead_code = 2
# VQ_lucidrains_VQ_message = "randomly remove embeddings that haven't been used in the last 2 epochs"

VQ_lucidrains_VQ_message = "conv_encoder, conv_decoder, VQ -> kmeans init, cosine similarity, expiring stale embeddings"

VQ_optimizer = "AdamW"
VQ_optimizer_lr = 5e-4
VQ_optimizer_weight_decay = 5e-5

VQ_loss_weight_recon_L2 = 1.0
VQ_loss_weight_recon_L1 = 0.1
VQ_loss_weight_perceptual = 0.
VQ_loss_weight_codebook = 0.1

VQ_train_epoch = 1000
VQ_train_gradiernt_clip = 1.0



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
        # "VQ_patch_size": VQ_patch_size,
        # "VQ_encoder_dim": VQ_encoder_dim,
        # "VQ_encoder_depth": VQ_encoder_depth,
        # "VQ_encoder_heads": VQ_encoder_heads,
        # "VQ_encoder_mlp_dim": VQ_encoder_mlp_dim,
        # "VQ_encoder_dim_head": VQ_encoder_dim_head,
        # "VQ_decoder_dim": VQ_decoder_dim,
        # "VQ_decoder_depth": VQ_decoder_depth,
        # "VQ_decoder_heads": VQ_decoder_heads,
        # "VQ_decoder_mlp_dim": VQ_decoder_mlp_dim,
        # "VQ_decoder_dim_head": VQ_decoder_dim_head,
        "VQ_encoder_channels": VQ_encoder_channels,
        "VQ_encoder_num_res_units": VQ_encoder_num_res_units,
        "VQ_decoder_channels": VQ_decoder_channels,
        "VQ_decoder_num_res_units": VQ_decoder_num_res_units,
        # "VQ_quantizer_embed_dim": VQ_quantizer_embed_dim,
        # "VQ_quantizer_n_embed": VQ_quantizer_n_embed,
        # "VQ_quantizer_beta": VQ_quantizer_beta,
        # "VQ_quantizer_use_norm": VQ_quantizer_use_norm,
        # "VQ_quantizer_use_residual": VQ_quantizer_use_residual,
        "VQ_optimizer": VQ_optimizer,
        "VQ_optimizer_lr": VQ_optimizer_lr,
        "VQ_optimizer_weight_decay": VQ_optimizer_weight_decay,
        "VQ_loss_weight_recon_L2": VQ_loss_weight_recon_L2,
        "VQ_loss_weight_recon_L1": VQ_loss_weight_recon_L1,
        "VQ_loss_weight_perceptual": VQ_loss_weight_perceptual,
        "VQ_loss_weight_codebook": VQ_loss_weight_codebook,
        "VQ_train_epoch": VQ_train_epoch,
        "VQ_train_gradiernt_clip": VQ_train_gradiernt_clip,
        "VQ_lucidrains_VQ_type": VQ_lucidrains_VQ_type,
        "VQ_lucidrains_VQ_embed_dim": VQ_lucidrains_VQ_embed_dim,
        "VQ_lucidrains_VQ_n_embed": VQ_lucidrains_VQ_n_embed,
        "VQ_lucidrains_VQ_decay": VQ_lucidrains_VQ_decay,
        "VQ_lucidrains_VQ_commiment_weight": VQ_lucidrains_VQ_commiment_weight,
        "VQ_lucidrains_VQ_kmeans_init": VQ_lucidrains_VQ_kmeans_init,
        "VQ_lucidrains_VQ_kmeans_iters": VQ_lucidrains_VQ_kmeans_iters,
        # "VQ_lucidrains_VQ_codebook_dim": VQ_lucidrains_VQ_codebook_dim,
        "VQ_lucidrains_VQ_use_cosine_sim": VQ_lucidrains_VQ_use_cosine_sim,
        "VQ_lucidrains_threshold_ema_dead_code": VQ_lucidrains_threshold_ema_dead_code,
        "VQ_lucidrains_VQ_message": VQ_lucidrains_VQ_message,
    }
)



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (D*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (D*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (D*H*W, D/3)

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1) # (D*H*W, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size):
    grid_size = (grid_size, grid_size, grid_size) if type(grid_size) != tuple else grid_size
    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w, h, d goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

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

        self.down1 = ResidualUnit(3, self.in_channels, self.channels[0], self.strides[0],
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
        self.down2 = ResidualUnit(3, self.channels[0], self.channels[1], self.strides[1],
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
        self.down3 = ResidualUnit(3, self.channels[1], self.channels[2], self.strides[2],
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
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
        # x is (B, C, H, W, D), where C = self.in_channels
        x1 = self.down1(x)
        # x1 is (B, C1, H1, W1, D1), where C1 = self.channels[0], H1 = H//2, W1 = W//2, D1 = D//2
        x2 = self.down2(x1)
        # x2 is (B, C2, H2, W2, D2), where C2 = self.channels[1], H2 = H//4, W2 = W//4, D2 = D//4
        x3 = self.down3(x2)
        # x3 is (B, C3, H3, W3, D3), where C3 = self.channels[2], H3 = H//8, W3 = W//8, D3 = D//8
        x3_flatten = self.flatten(x3)
        return x3_flatten
    

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

        self.up3 = nn.Sequential(
            Convolution(3, self.channels[2], self.channels[1], strides=self.strides[2], kernel_size=self.up_kernel_size, act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, conv_only=False, is_transposed=True, adn_ordering=self.adn_ordering),
            ResidualUnit(3, self.channels[1], self.channels[1], strides=1, kernel_size=self.kernel_size, subunits=1, act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, last_conv_only=False, adn_ordering=self.adn_ordering,),
        )
        self.up2 = nn.Sequential(
            Convolution(3, self.channels[1], self.channels[0], strides=self.strides[1], kernel_size=self.up_kernel_size, act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, conv_only=False, is_transposed=True, adn_ordering=self.adn_ordering),
            ResidualUnit(3, self.channels[0], self.channels[0], strides=1, kernel_size=self.kernel_size, subunits=1, act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, last_conv_only=False, adn_ordering=self.adn_ordering,),
        )
        self.up1 = nn.Sequential(
            Convolution(3, self.channels[0], self.out_channels, strides=self.strides[0], kernel_size=self.up_kernel_size, act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, conv_only=False, is_transposed=True, adn_ordering=self.adn_ordering),
            ResidualUnit(3, self.out_channels, self.out_channels, strides=1, kernel_size=self.kernel_size, subunits=1, act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, last_conv_only=False, adn_ordering=self.adn_ordering,),
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
        x3 = self.unflatten(x)
        # x3 is (B, C3, H3, W3, D3), where C3 = self.channels[2], H3 = H//8, W3 = W//8, D3 = D//8
        x2 = self.up3(x3)
        # x2 is (B, C2, H2, W2, D2), where C2 = self.channels[1], H2 = H//4, W2 = W//4, D2 = D//4
        x1 = self.up2(x2)
        # x1 is (B, C1, H1, W1, D1), where C1 = self.channels[0], H1 = H//2, W1 = W//2, D1 = D//2
        x = self.up1(x1)
        # x is (B, C, H, W, D), where C = self.out_channels, H = H, W = W, D = D

        return x



# class ViTVQ3D(pl.LightningModule):
class ViTVQ3D(nn.Module):
    def __init__(self, volume_key: str, volume_size: int, encoder: dict, decoder: dict, quantizer: dict,
                 path: Optional[str] = None, ignore_keys: List[str] = list(), scheduler: Optional[dict] = None) -> None:
        super().__init__()
        self.path = path
        self.ignore_keys = ignore_keys 
        self.volume_key = volume_key
        self.scheduler = scheduler 
        
        # self.encoder = ViTEncoder3D(volume_size=volume_size, patch_size=patch_size, **encoder)
        # self.decoder = ViTDecoder3D(volume_size=volume_size, patch_size=patch_size, **decoder)
        self.encoder = UNet3D_encoder(**encoder)
        self.decoder = UNet3D_decoder(**decoder)
        # self.quantizer = VectorQuantizer(**quantizer)
        # quantizer={
        #     "dim": VQ_lucidrains_VQ_embed_dim, "codebook_size": VQ_lucidrains_VQ_n_embed, "decay": VQ_lucidrains_VQ_decay, "commitment_weight": VQ_lucidrains_VQ_commiment_weight,
        #     "kmeans_init": VQ_lucidrains_VQ_kmeans_init, "kmeans_iter": VQ_lucidrains_VQ_kmeans_iters,
        # }
        self.quantizer = lucidrains_VQ(
            dim = quantizer["embed_dim"],
            codebook_size = quantizer["codebook_size"],
            decay = quantizer["decay"],
            commitment_weight = quantizer["commitment_weight"],
            # codebook_dim = quantizer["codebook_dim"],
            kmeans_init = quantizer["kmeans_init"],
            kmeans_iters = quantizer["kmeans_iters"],
            use_cosine_sim = quantizer["use_cosine_sim"],
            threshold_ema_dead_code = quantizer["threshold_ema_dead_code"],
        )
        self.pre_quant = nn.Linear(encoder["channels"][2], quantizer["embed_dim"])
        self.post_quant = nn.Linear(quantizer["embed_dim"], decoder["channels"][2])
        if path is not None:
            self.init_from_ckpt(path, ignore_keys)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:    
        quant, indices, diff = self.encode(x)
        dec = self.decode(quant)
        
        return dec, indices, diff

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        
    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.pre_quant(h)
        # x = torch.randn(1, 1024, 256)
        # quantized, indices, commit_loss = vq(x) # (1, 1024, 256), (1, 1024), (1)
        # quant, emb_loss, _ = self.quantizer(h)
        quant, indices, loss = self.quantizer(h)
        return quant, indices, loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant)
        
        return dec

    def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        h = self.encoder(x)
        h = self.pre_quant(h)
        _, _, codes = self.quantizer(h)
        
        return codes

    def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
        quant = self.quantizer.embedding(code)
        quant = self.quantizer.norm(quant)
        
        if self.quantizer.use_residual:
            quant = quant.sum(-2)  
            
        dec = self.decode(quant)
        
        return dec

    def get_input(self, batch: Tuple[Any, Any], key: str = 'volume') -> Any:
        x = batch[key]
        if len(x.shape) == 4:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()

        return x.contiguous()

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        x = self.get_input(batch, self.volume_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencoder
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, batch_idx,
                                            last_layer=self.decoder.get_last_layer(), split="train")

            self.log("train/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            del log_dict_ae["train/total_loss"]
            
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, batch_idx,
                                                last_layer=self.decoder.get_last_layer(), split="train")
            
            self.log("train/disc_loss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            del log_dict_disc["train/disc_loss"]
            
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            
            return discloss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = self.get_input(batch, self.volume_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, batch_idx,
                                        last_layer=self.decoder.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/rec_loss"]

        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        del log_dict_ae["val/total_loss"]

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        if hasattr(self.loss, 'discriminator'):
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, batch_idx,
                                                last_layer=self.decoder.get_last_layer(), split="val")
            
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        return self.log_dict

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate
        optim_groups = list(self.encoder.parameters()) + \
                       list(self.decoder.parameters()) + \
                       list(self.pre_quant.parameters()) + \
                       list(self.post_quant.parameters()) + \
                       list(self.quantizer.parameters())
        
        optimizers = [torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)]
        schedulers = []
        
        if hasattr(self.loss, 'discriminator'):
            optimizers.append(torch.optim.AdamW(self.loss.discriminator.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4))

        if self.scheduler is not None:
            self.scheduler.params.start = lr
            scheduler = initialize_from_config(self.scheduler)
            
            schedulers = [
                {
                    'scheduler': lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                } for optimizer in optimizers
            ]
   
        return optimizers, schedulers
        
    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.volume_key).to(self.device)
        quant, _ = self.encode(x)
        
        log["originals"] = x
        log["reconstructions"] = self.decode(quant)
        
        return log
    

# apply the channel wise norm for all feature maps
# Normalize activations
def normalize_activations(features):
    normalized_features = []
    for feature in features:
        # Compute the norm along the channel dimension
        norm = torch.norm(feature, p=2, dim=1, keepdim=True)
        # Normalize the feature map
        normalized_feature = feature / (norm + 1e-8)  # Add a small value to avoid division by zero
        normalized_features.append(normalized_feature)
    return normalized_features

# Compute l2 distance for each pair of feature maps
def l2_distance(features1, features2):
    l2_distances = []
    for f1, f2 in zip(features1, features2):
        # Compute the l2 distance
        l2_dist = torch.norm(f1 - f2, p=2, dim=1)  # Sum across channel dimension
        l2_distances.append(l2_dist)
    return l2_distances

# Average the l2 distances across spatial dimensions
def average_spatial(distances):
    averaged_distances = []
    for dist in distances:
        # Average across spatial dimensions (dimensions 1, 2, 3)
        avg_dist = torch.mean(dist, dim=[1, 2, 3])
        averaged_distances.append(avg_dist)
    return averaged_distances

# Overall average across all layers
def average_all_layers(distances):
    total_distance = torch.stack(distances).mean()
    return total_distance

class UNet3D_perceptual(nn.Module):
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
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
        pretrained_path = None,
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
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
        self.pretrained_path = pretrained_path


        # UNet( 
        # spatial_dims=unet_dict["spatial_dims"],
        # in_channels=unet_dict["in_channels"],
        # out_channels=unet_dict["out_channels"],
        # channels=unet_dict["channels"],
        # strides=unet_dict["strides"],
        # num_res_units=unet_dict["num_res_units"],
        # act=unet_dict["act"],
        # norm=unet_dict["normunet"],
        # dropout=unet_dict["dropout"],
        # bias=unet_dict["bias"],
        # )

        # input - down1 ------------- up1 -- output
        #         |                   |
        #         down2 ------------- up2
        #         |                   |
        #         down3 ------------- up3
        #         |                   |
        #         down4 -- bottom --  up4
        # 1 -> (32, 64, 128, 256) -> 1

        self.down1 = ResidualUnit(3, self.in_channels, self.channels[0], self.strides[0],
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
        self.down2 = ResidualUnit(3, self.channels[0], self.channels[1], self.strides[1],
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
        self.down3 = ResidualUnit(3, self.channels[1], self.channels[2], self.strides[2],
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
        self.bottom = ResidualUnit(3, self.channels[2], self.channels[3], 1,
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)

        self.load_weights(self.pretrained_path)
        self.set_all_parameters_ignore_grad()
    
    def load_weights(self, path: str):
        # the path is to the whole model, where we only need the encoder part
        # iterate the current model weight names and load the weights from the pre-trained model
        pretrain_dict = torch.load(path)
        current_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in current_dict}
        # 2. overwrite entries in the existing state dict
        current_dict.update(pretrain_dict)
        # 3. load the new state dict
        self.load_state_dict(current_dict)

    def set_all_parameters_ignore_grad(self):
        for param in self.parameters():
            param.requires_grad = False


    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x1 = self.down1(x)
    #     x2 = self.down2(x1)
    #     x3 = self.down3(x2)
    #     x4 = self.bottom(x3)
    #     return x1, x2, x3, x4

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y1 = self.down1(y)
        y2 = self.down2(y1)
        y3 = self.down3(y2)
        y4 = self.bottom(y3)

        z1 = self.down1(z)
        z2 = self.down2(z1)
        z3 = self.down3(z2)
        z4 = self.bottom(z3)

        y_features = [y1, y2, y3, y4]
        z_features = [z1, z2, z3, z4]

        y_fea_norm = normalize_activations(y_features)
        z_fea_norm = normalize_activations(z_features)

        l2_dist = l2_distance(y_fea_norm, z_fea_norm)
        avg_dist = average_spatial(l2_dist)
        total_dist = average_all_layers(avg_dist)

        return total_dist
    


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
with open("data_chunks.json", "r") as f:
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

class RobustCacheDataset(CacheDataset):
    def __init__(self, data, transform=None, cache_num=0, cache_rate=1.0, num_workers=0):
        super().__init__(data, transform=transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def __getitem__(self, idx):
        try:
            # Fetch the cached data
            data = super().__getitem__(idx)
            return data
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # Handle the error appropriately, for example, return None or a default value
            return None

def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)


train_ds = RobustCacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_num=num_train_files,
    cache_rate=cache_ratio_train, # 600 * 0.1 = 60
    num_workers=num_workers_train_cache_dataset,
)

val_ds = RobustCacheDataset(
    data=val_files,
    transform=val_transforms, 
    cache_num=num_val_files,
    cache_rate=cache_ratio_val, # 360 * 0.05 = 18
    num_workers=num_workers_val_cache_dataset,)



train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=num_workers_train_dataloader, worker_init_fn=worker_init_fn, collate_fn=collate_fn, timeout=60)
val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False, num_workers=num_workers_val_dataloader, worker_init_fn=worker_init_fn, collate_fn=collate_fn, timeout=60)

# model = ViTVQ3D(
#     volume_key="volume", volume_size=volume_size, patch_size=8,
#     encoder={
#         "dim": 360, "depth": 6, "heads": 16, "mlp_dim": 1024, "channels": 1, "dim_head": 128
#     },
#     decoder={
#         "dim": 360, "depth": 6, "heads": 16, "mlp_dim": 1024, "channels": 1, "dim_head": 128
#     },
#     quantizer={
#         "embed_dim": 128, "n_embed": 1024, "beta": 0.25, "use_norm": True, "use_residual": False
#     }
# ).to(device)

model = ViTVQ3D(
    volume_key="volume", volume_size=volume_size, 
    encoder={
        # spatial_dims: int,
        # in_channels: int,
        # channels: Sequence[int],
        # strides: Sequence[int],
        "spatial_dims": 3, "in_channels": 1, "channels": VQ_encoder_channels, "strides": [2, 2, 2], "num_res_units": VQ_decoder_num_res_units,
    },
    decoder={
        # spatial_dims: int,
        # out_channels: int,
        # channels: Sequence[int],
        # strides: Sequence[int],
        # num_res_units: int = 0,
        # hwd: Union[Tuple, str] = 8,
        "spatial_dims": 3, "out_channels": 1, "channels": VQ_decoder_channels, "strides": [2, 2, 2], "num_res_units": VQ_decoder_num_res_units, "hwd": volume_size // 8,

    },
    quantizer={
        "embed_dim": VQ_lucidrains_VQ_embed_dim, "codebook_size": VQ_lucidrains_VQ_n_embed, "decay": VQ_lucidrains_VQ_decay, "commitment_weight": VQ_lucidrains_VQ_commiment_weight,
        "threshold_ema_dead_code": VQ_lucidrains_threshold_ema_dead_code,
        "use_cosine_sim": VQ_lucidrains_VQ_use_cosine_sim,
        # "codebook_dim": VQ_lucidrains_VQ_codebook_dim,
        "kmeans_init": VQ_lucidrains_VQ_kmeans_init, "kmeans_iters": VQ_lucidrains_VQ_kmeans_iters,
    },
).to(device)


learning_rate = VQ_optimizer_lr
# use AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=VQ_optimizer_weight_decay)

# here we set the encoder model parameters

preceptual_loss = dict()
preceptual_loss["spatial_dims"] = 3
preceptual_loss["in_channels"] = 1
preceptual_loss["out_channels"] = 1
preceptual_loss["channels"] = [32, 64, 128, 256]
preceptual_loss["strides"] = [2, 2, 2]
preceptual_loss["num_res_units"] = 4
preceptual_loss["pretrained_path"] = "model_best_181_state_dict.pth"

if VQ_loss_weight_perceptual > 0:
    preceptual_model = UNet3D_perceptual(**preceptual_loss).to(device)
    preceptual_model.eval()
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


current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
log_file_path = f"train_log_{current_time}.json"
logger = simple_logger(log_file_path)

num_epoch = VQ_train_epoch
loss_weights = {
    "reconL2": VQ_loss_weight_recon_L2,
    "reconL1": VQ_loss_weight_recon_L1,
    "perceptual": VQ_loss_weight_perceptual,
    "codebook": VQ_loss_weight_codebook,
}

num_train_batch = len(train_loader)
num_val_batch = len(val_loader)
best_val_loss = 1e6
save_folder = "./results/"
# create the folder if not exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for idx_epoch in range(num_epoch):
    model.train()
    epoch_loss_train = {
        "reconL2": [],
        "reconL1": [],
        "perceptual": [],
        "codebook": [],
        "total": [],
    }
    epoch_grad_train = {
        "encoder": [],
        "decoder": [],
        "quantizer": [],
    }
    epoch_codebook = {
        "indices": [],
    }

    for idx_batch, batch in enumerate(train_loader):
        x = batch["image"].to(device)
        # print x size
        # print("x size is ", x.size())
        xrec, indices, cb_loss = model(x)
        optimizer.zero_grad()
        
        if VQ_loss_weight_perceptual > 0:
            total_dist = preceptual_model(x, xrec)
        else:
            # make the total_dist.item() = 0
            total_dist = F.mse_loss(torch.zeros(1), torch.zeros(1))
        # total_dist = preceptual_model(x, xrec)
        reconL2_loss = F.mse_loss(x, xrec)
        reconL1_loss = F.l1_loss(x, xrec)
        perceptual_loss = total_dist
        codebook_loss = cb_loss
        total_loss = loss_weights["reconL2"] * reconL2_loss + \
                        loss_weights["reconL1"] * reconL1_loss + \
                        loss_weights["perceptual"] * perceptual_loss + \
                        loss_weights["codebook"] * codebook_loss
        epoch_loss_train["reconL2"].append(reconL2_loss.item())
        epoch_loss_train["reconL1"].append(reconL1_loss.item())
        epoch_loss_train["perceptual"].append(perceptual_loss.item())
        epoch_loss_train["codebook"].append(codebook_loss.item())
        epoch_loss_train["total"].append(total_loss.item())
        print(f"<{idx_epoch}> [{idx_batch}/{num_train_batch}] Total loss: {total_loss.item()}")
        # add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=VQ_train_gradiernt_clip)
        total_loss.backward()
        optimizer.step()

        # compute the average gradient
        encoder_grad = compute_average_gradient(model.encoder)
        decoder_grad = compute_average_gradient(model.decoder)
        quantizer_grad = compute_average_gradient(model.quantizer)
        epoch_grad_train["encoder"].append(encoder_grad)
        epoch_grad_train["decoder"].append(decoder_grad)
        epoch_grad_train["quantizer"].append(quantizer_grad)

        # record the codebook indices
        epoch_codebook["indices"].append(indices.cpu().numpy())
    
    for key in epoch_loss_train.keys():
        epoch_loss_train[key] = np.asanyarray(epoch_loss_train[key])
        logger.log(idx_epoch, f"train_{key}_mean", epoch_loss_train[key].mean())
        # logger.log(idx_epoch, f"train_{key}_std", epoch_loss_train[key].std())
    
    for key in epoch_grad_train.keys():
        epoch_grad_train[key] = np.asanyarray(epoch_grad_train[key])
        logger.log(idx_epoch, f"train_{key}_grad_mean", epoch_grad_train[key].mean())
        # logger.log(idx_epoch, f"train_{key}_grad_std", epoch_grad_train[key].std())
    
    for key in epoch_codebook.keys():
        epoch_codebook[key] = np.asanyarray(epoch_codebook[key])
    activated_value, activated_counts = np.unique(epoch_codebook["indices"], return_counts=True)
    # a = np.array([1, 2, 6, 4, 2, 3, 2])
    # values, counts = np.unique(a, return_counts=True)
    # values = array([1, 2, 3, 4, 6])
    # counts = array([1, 3, 1, 1, 1])
    activated_percentage = len(activated_counts) / VQ_lucidrains_VQ_n_embed # percentage
    activated_dispersion = np.std(activated_counts) # dispersion
    logger.log(idx_epoch, "train_activated_percentage", activated_percentage)
    logger.log(idx_epoch, "train_activated_dispersion", activated_dispersion)
    
    

    # validation
    if idx_epoch % val_per_epoch == 0:
        model.eval()
        epoch_loss_val = {
            "reconL2": [],
            "reconL1": [],
            "perceptual": [],
            "codebook": [],
            "total": [],
        }
        epoch_codebook_val = {
            "indices": [],
        }
        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                x = batch["image"].to(device)
                xrec, indices, cb_loss = model(x)
                if VQ_loss_weight_perceptual > 0:
                    total_dist = preceptual_model(x, xrec)
                else:
                    total_dist = total_dist = F.mse_loss(torch.zeros(1), torch.zeros(1))
                # total_dist = preceptual_model(x, xrec)
                reconL2_loss = F.mse_loss(x, xrec)
                reconL1_loss = F.l1_loss(x, xrec)
                perceptual_loss = total_dist
                codebook_loss = cb_loss
                total_loss = loss_weights["reconL2"] * reconL2_loss + \
                                loss_weights["reconL1"] * reconL1_loss + \
                                loss_weights["perceptual"] * perceptual_loss + \
                                loss_weights["codebook"] * codebook_loss
                epoch_loss_val["reconL2"].append(reconL2_loss.item())
                epoch_loss_val["reconL1"].append(reconL1_loss.item())
                epoch_loss_val["perceptual"].append(perceptual_loss.item())
                epoch_loss_val["codebook"].append(codebook_loss.item())
                epoch_loss_val["total"].append(total_loss.item())
                print(f"<{idx_epoch}> [{idx_batch}/{num_val_batch}] Total loss: {total_loss.item()}")

                # record the codebook indices (1, n_embed)
                current_indices = indices.cpu().numpy().squeeze()
                # append each element in current_indices to the list
                for i in range(current_indices.shape[0]):
                    epoch_codebook_val["indices"].append(current_indices[i])
        
        # save the last batch images
        numpy_x = x.cpu().numpy().squeeze()
        numpy_xrec = xrec.cpu().numpy().squeeze()

        save_name = f"epoch_{idx_epoch}_batch_{idx_batch}"
        plot_and_save_x_xrec(x, xrec, num_per_direction=3, savename=save_folder+f"{save_name}.png", wandb_name="val_snapshots")
        
        for key in epoch_loss_val.keys():
            epoch_loss_val[key] = np.asanyarray(epoch_loss_val[key])
            logger.log(idx_epoch, f"val_{key}_mean", epoch_loss_val[key].mean())
            # logger.log(idx_epoch, f"val_{key}_std", epoch_loss_val[key].std())

        if epoch_loss_val["total"].mean() < best_val_loss:
            best_val_loss = epoch_loss_val["total"].mean()
            torch.save(model.state_dict(), save_folder+f"model_best_{idx_epoch}_state_dict.pth")
            torch.save(optimizer.state_dict(), save_folder+f"optimizer_best_{idx_epoch}_state_dict.pth")
            logger.log(idx_epoch, "best_val_loss", best_val_loss)

        for key in epoch_codebook_val.keys():
            epoch_codebook_val[key] = np.asanyarray(epoch_codebook_val[key])
        activated_value, activated_counts = np.unique(epoch_codebook_val["indices"], return_counts=True)
        activated_percentage = len(activated_counts) / VQ_lucidrains_VQ_n_embed
        activated_dispersion = np.std(activated_counts)
        logger.log(idx_epoch, "val_activated_percentage", activated_percentage)
        logger.log(idx_epoch, "val_activated_dispersion", activated_dispersion)
    
    # save the model every save_per_epoch
    if idx_epoch % save_per_epoch == 0:
        torch.save(model.state_dict(), save_folder+f"model_{idx_epoch}_state_dict.pth")
        torch.save(optimizer.state_dict(), save_folder+f"optimizer_{idx_epoch}_state_dict.pth")
        logger.log(idx_epoch, "model_saved", f"model_{idx_epoch}_state_dict.pth")
        

wandb.finish()