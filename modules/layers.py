"""
This is changed from the orginal code and modified by ChatGPT from
https://github.com/thuanz123/enhancing-transformers/blob/1778fc497ea11ed2cef134404f99d4d6b921cda9/enhancing/modules/stage1/layers.py
"""

import math
import numpy as np
from typing import Union, Tuple, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (D*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (D*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (D*H*W, D/3)

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1) # (D*H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

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
        q, k, v = map(lambda t: rearrange(t, 'b (d h w) (head dim) -> b head (d h w) dim', head = self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head (d h w) dim -> b (d h w) (head dim)')

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

class ViTEncoder3D(nn.Module):
    def __init__(self, volume_size: Union[Tuple[int, int, int], int], patch_size: Union[Tuple[int, int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 1, dim_head: int = 64) -> None:
        super().__init__()
        volume_depth, volume_height, volume_width = volume_size if isinstance(volume_size, tuple) else (volume_size, volume_size, volume_size)
        patch_depth, patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)

        assert volume_depth % patch_depth == 0 and volume_height % patch_height == 0 and volume_width % patch_width == 0, 'Volume dimensions must be divisible by the patch size.'
        en_pos_embedding = get_3d_sincos_pos_embed(dim, (volume_depth // patch_depth, volume_height // patch_height, volume_width // patch_width))

        self.num_patches = (volume_depth // patch_depth) * (volume_height // patch_height) * (volume_width // patch_width)
        self.patch_dim = channels * patch_depth * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.Conv3d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c d h w -> b (d h w) c'),
        )
        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.apply(init_weights)

    def forward(self, volume: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_patch_embedding(volume)
        x = x + self.en_pos_embedding
        x = self.transformer(x)

        return x


class ViTDecoder3D(nn.Module):
    def __init__(self, volume_size: Union[Tuple[int, int, int], int], patch_size: Union[Tuple[int, int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 1, dim_head: int = 64) -> None:
        super().__init__()
        volume_depth, volume_height, volume_width = volume_size if isinstance(volume_size, tuple) else (volume_size, volume_size, volume_size)
        patch_depth, patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)

        assert volume_depth % patch_depth == 0 and volume_height % patch_height == 0 and volume_width % patch_width == 0, 'Volume dimensions must be divisible by the patch size.'
        de_pos_embedding = get_3d_sincos_pos_embed(dim, (volume_depth // patch_depth, volume_height // patch_height, volume_width // patch_width))

        self.num_patches = (volume_depth // patch_depth) * (volume_height // patch_height) * (volume_width // patch_width)
        self.patch_dim = channels * patch_depth * patch_height * patch_width

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.to_voxel = nn.Sequential(
            Rearrange('b (d h w) c -> b c d h w', d=volume_depth // patch_depth),
            nn.ConvTranspose3d(dim, channels, kernel_size=patch_size, stride=patch_size)
        )

        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor) -> torch.FloatTensor:
        x = token + self.de_pos_embedding
        x = self.transformer(x)
        x = self.to_voxel(x)

        return x

    def get_last_layer(self) -> nn.Parameter:
        return self.to_voxel[-1].weight