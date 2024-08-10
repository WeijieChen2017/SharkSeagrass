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

# set the environment variable to use the GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import wandb
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


from einops.layers.torch import Rearrange
from typing import Tuple, Union, Sequence

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

from vector_quantize_pytorch import VectorQuantize as lucidrains_VQ

from monai.data import (
    DataLoader,
    CacheDataset,
)

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotated,
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


class simple_logger():
    def __init__(self, log_file_path, global_config):
        self.log_file_path = log_file_path
        self.log_dict = dict()
        # self.IS_LOGGER_WANDB = global_config["IS_LOGGER_WANDB"]
        # self.wandb_run = global_config["wandb_run"]
    
    def log(self, global_epoch, key, msg):
        if key not in self.log_dict.keys():
            self.log_dict[key] = dict()
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dict[key] = {
            "time": current_time,
            "global_epoch": global_epoch,
            "msg": msg
        }
        log_str = f"{current_time} Global epoch: {global_epoch}, {key}, {msg}"
        with open(self.log_file_path, "a") as f:
            f.write(log_str)
        print(log_str)

        # # log to wandb if msg is number
        # if self.IS_LOGGER_WANDB and isinstance(msg, (int, float)):
        #     self.wandb_run.log({key: msg})

def collate_fn(batch, pet_valid_th=0.01):
    # batch is a list of list of dictionary
    idx = len(batch)
    jdx = len(batch[0])
    
    modalities = batch[0][0].keys()
    print("The modalities are: ", modalities)
    print(batch[0])
    valid_samples = {
        modal : [] for modal in modalities
    }
    # here we need to filter out the samples with PET_raw mean value less than pet_valid_th
    for i in range(idx):
        for j in range(jdx):
            if batch[i][j]["PET_raw"].mean() > pet_valid_th:
                for modal in modalities:
                    valid_samples[modal].append(batch[i][j][modal])
    
    # here we need to stack the valid samples
    for modal in modalities:
        valid_samples[modal] = torch.stack(valid_samples[modal])
    # here we need to return the valid samples
    return valid_samples


def build_dataloader_train_val_PET_CT(batch_size, global_config):

    volume_size = global_config["volume_size"]
    input_modality = global_config["input_modality"]
    gap_sign = global_config["gap_sign"]

    # set the data transform
    train_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality),
            RandSpatialCropSamplesd(keys=input_modality,
                                    roi_size=(volume_size, volume_size, volume_size),
                                    num_samples=global_config["batches_from_each_nii"],
                                    random_size=False, random_center=True),
            RandFlipd(keys=input_modality, prob=0.5, spatial_axis=0),
            RandFlipd(keys=input_modality, prob=0.5, spatial_axis=1),
            RandFlipd(keys=input_modality, prob=0.5, spatial_axis=2),
            RandRotated(keys=input_modality, prob=0.5, range_x=15, range_y=15, range_z=15),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality),
            RandSpatialCropSamplesd(keys=input_modality,
                                    roi_size=(volume_size, volume_size, volume_size),
                                    num_samples=global_config["batches_from_each_nii"],
                                    random_size=False, random_center=True),
        ]
    )

    data_division_file = global_config["data_division"]
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
    print(gap_sign*23)

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
                              batch_size=batch_size,
                              shuffle=True, 
                              num_workers=global_config["num_workers_train_dataloader"],
                              collate_fn=collate_fn
                              )
    val_loader = DataLoader(val_ds, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=global_config["num_workers_val_dataloader"],
                            collate_fn=collate_fn
                            )
    
    return train_loader, val_loader

def generate_model_levels(global_config):
    num_level = len(global_config['pyramid_channels'])
    model_levels = []
    for i in range(num_level):
        encoder = {
            "spatial_dims": 3, "in_channels": 1,
            "channels": global_config['pyramid_channels'][:i+1],
            "strides": global_config['pyramid_strides'][-(i+1):],
            "num_res_units": global_config['pyramid_num_res_units'][i],
        }
        second_encoder = {
            "spatial_dims": 3, "in_channels": len(global_config['input_modality'])-1,
            "channels": global_config['pyramid_channels'][:i+1],
            "strides": global_config['pyramid_strides'][-(i+1):],
            "num_res_units": global_config['pyramid_num_res_units'][i] + 2,
        }
        quantizer = {
            "dim": global_config['pyramid_channels'][i]*2, 
            "codebook_size": global_config['pyramid_codebook_size'][i],
            "decay": 0.8, "commitment_weight": 1.0,
            "use_cosine_sim": True, "threshold_ema_dead_code": 2,
            "kmeans_init": True, "kmeans_iters": 10
        }
        decoder = {
            "spatial_dims": 3, "out_channels": 1,
            "channels": global_config['pyramid_channels'][:i+1],
            "strides": global_config['pyramid_strides'][-(i+1):],
            "num_res_units": global_config['pyramid_num_res_units'][i],
            "hwd": global_config['pyramid_mini_resolution'],
        }

        # double the channels for the second encoder
        second_encoder["channels"] = [value * 2 for value in second_encoder["channels"]]


        model_levels.append({
            "encoder": encoder,
            "second_encoder": second_encoder,
            "decoder": decoder,
            "quantizer": quantizer
        })

    # output the model levels in strucutre
    print("The model levels are: ")
    for i, level in enumerate(model_levels):
        print(f"Level {i} is: ")
        print(level)
        print("==="*10)
    return model_levels

def generate_input_data_pyramid(x, levels, global_config):
    pyramid_mini_resolution = global_config['pyramid_mini_resolution']
    pyramid_x = []
    for i in range(levels + 1):
        x_at_level = F.interpolate(x, size=(pyramid_mini_resolution*2**i,
                                            pyramid_mini_resolution*2**i, 
                                            pyramid_mini_resolution*2**i), mode="trilinear", align_corners=False).to(device)
        # print(f"Level {i} shape is {x`_at_level.shape}")
        pyramid_x.append(x_at_level)
    
    return pyramid_x

def build_optimizer(model, learning_rate, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

class InfoNCELoss(nn.Module):
    # InfoNCELoss is a class to compute the InfoNCE loss
    # Initialize the class with
    # - codebook: the codebook is a K * D tensor
    # - similarity_type: the similarity type is either "cosine", "euclidean", or "manhattan"
    # - temperature: the temperature is a float value

    def __init__(self, 
                 codebook : torch.FloatTensor,
                 similarity_type: str = "cosine",
                 temperature: float = 0.1,
                 device: torch.device = torch.device("cuda:"+os.environ["CUDA_VISIBLE_DEVICES"])):
        self.codebook = codebook
        self.K, self.D = codebook.shape
        self.similarity_type = similarity_type
        self.temperature = temperature
        self.device = device

        # Compute the similarity matrix
        self.similarity_matrix = self.get_similarity_matrix(codebook).to(self.device)
        self.InfoNCEloss_matrix = self.precompute_all_pairs().to(self.device)

    def compute_InfoNCEloss_list(self, indices_pair_list):
        losses = [self.InfoNCEloss_matrix[i, j] for i, j in indices_pair_list]
        # print(f"Losses are {losses}")
        # return torch.tensor(losses).mean()
        # Convert MetaTensors to regular tensors and flatten them if necessary
        losses = [loss.flatten() if isinstance(loss, torch.Tensor) else torch.tensor(loss) for loss in losses]

        # Concatenate all loss tensors to compute the overall mean
        losses = torch.cat(losses).mean()
        return losses


    def precompute_all_pairs(self):
        loss_matrix = torch.zeros((self.K, self.K), device=self.device)
        for i in range(self.K):
            for j in range(self.K):
                if i != j:
                    loss_matrix[i, j] = self.precompute_InfoNCEloss_pair(i, j)

        print(f"The precomputed InfoNCE loss matrix is computed with {self.K} entries of shape {self.K} * {self.K}")
        return loss_matrix

    def precompute_InfoNCEloss_pair(self, index_i, index_j):

        # Load positive similarity from similarity_matrix
        pos_sim = self.similarity_matrix[index_i, index_j].to(self.device)

        # Load similarities with the rest of the codebook from similarity_matrix
        sim_i = self.similarity_matrix[index_i, :].to(self.device) / self.temperature  # Shape: (K,)
        
        # Combine similarities
        logits = torch.cat((torch.tensor([pos_sim], device=self.device), sim_i.view(-1)), dim=0)

        # Create labels (the positive pair is at index 0)
        labels = torch.tensor([0], dtype=torch.long, device=self.device)

        # Compute the InfoNCE loss using cross-entropy
        loss = F.cross_entropy(logits.unsqueeze(0), labels)

        return loss.item()

    def compute_similarity(self, vector_i, vector_j):
        # Normalize the vectors
        vector_i = F.normalize(vector_i, dim=-1)
        vector_j = F.normalize(vector_j, dim=-1)
        if self.similarity_type == "cosine":
            sim_vector_i_j = F.cosine_similarity(vector_i, vector_j, dim=-1)
        elif self.similarity_type == "euclidean":
            sim_vector_i_j = -F.pairwise_distance(vector_i, vector_j, p=2)
        elif self.similarity_type == "manhattan":
            sim_vector_i_j = -F.pairwise_distance(vector_i, vector_j, p=1)
        else:
            raise NotImplementedError
        return sim_vector_i_j 

    def get_similarity_matrix(self, codebook: torch.FloatTensor) -> torch.FloatTensor:
        if self.similarity_type == "cosine":
            similarity_matrix = F.cosine_similarity(codebook.unsqueeze(0), codebook.unsqueeze(1), dim=-1)
        elif self.similarity_type == "euclidean":
            similarity_matrix = -F.pairwise_distance(codebook.unsqueeze(0), codebook.unsqueeze(1), p=2)
        elif self.similarity_type == "manhattan":
            similarity_matrix = -F.pairwise_distance(codebook.unsqueeze(0), codebook.unsqueeze(1), p=1)
        else:
            raise NotImplementedError
        return similarity_matrix  


class ViTVQ3D(nn.Module):
    def __init__(self, model_level: list) -> None:
        super().__init__()
        self.num_level = len(model_level)
        self.sub_models = nn.ModuleList()
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


class ViTVQ3D_dualEncoder(nn.Module):
    def __init__(self, model_levels: list) -> None:
        super().__init__()
        self.num_level = len(model_levels)
        self.sub_models = nn.ModuleList()
        for level_setting in model_levels:
            # Create a submodule to hold the encoder, decoder, quantizer, etc.
            sub_model = nn.Module() 
            sub_model.encoder = UNet3D_encoder(**level_setting["encoder"])
            sub_model.second_encoder = UNet3D_encoder(**level_setting["second_encoder"])
            sub_model.decoder = UNet3D_decoder(**level_setting["decoder"])
            sub_model.quantizer = lucidrains_VQ(**level_setting["quantizer"])
            sub_model.pre_quant = nn.Linear(level_setting["encoder"]["channels"][-1], level_setting["quantizer"]["dim"])
            sub_model.second_pre_quant = nn.Linear(level_setting["second_encoder"]["channels"][-1], level_setting["quantizer"]["dim"])
            sub_model.post_quant = nn.Linear(level_setting["quantizer"]["dim"], level_setting["decoder"]["channels"][0])
            
            # Append the submodule to the ModuleList
            self.sub_models.append(sub_model) 


        self.init_weights()
        self.freeze_gradient_all()

    # def load_weights_for_module(self, model_path):
    #     # only load the weights for the encoder, decoder, quantizer, pre_quant, post_quant
    #     # the model_path is the path to the model weights with the same structure .pth file
    #     # Load the weights from the given path
    #     checkpoint = torch.load(model_path)
    #     print(checkpoint.keys())
        
    #     for i, sub_model in enumerate(self.sub_models):
    #         # Load encoder weights
    #         sub_model.encoder.load_state_dict(checkpoint[f'sub_models.{i}.encoder'])
    #         # sub_model.second_encoder.load_state_dict(checkpoint[f'sub_models.{i}.second_encoder'])
            
    #         # Load decoder weights
    #         sub_model.decoder.load_state_dict(checkpoint[f'sub_models.{i}.decoder'])
            
    #         # Load quantizer weights
    #         sub_model.quantizer.load_state_dict(checkpoint[f'sub_models.{i}.quantizer'])
            
    #         # Load pre_quant and post_quant weights
    #         sub_model.pre_quant.load_state_dict(checkpoint[f'sub_models.{i}.pre_quant'])
    #         # sub_model.second_pre_quant.load_state_dict(checkpoint[f'sub_models.{i}.second_pre_quant'])
    #         sub_model.post_quant.load_state_dict(checkpoint[f'sub_models.{i}.post_quant'])
            
    #     print("Model weights loaded successfully from:", model_path)

    def pre_compute_InfoNCE_loss(self):
        # Compute the InfoNCE loss
        self.codebook_list = [submodel.quantizer.codebook for submodel in self.sub_models]
        self.InfoNCE_loss_list = [InfoNCELoss(codebook) for codebook in self.codebook_list]


    def compute_InfoNCE_loss(self, indices_list, level):
        return self.InfoNCE_loss_list[level].compute_InfoNCEloss_list(indices_list)
        
    def freeze_gradient_all(self) -> None:
        for level in range(self.num_level):
            self.freeze_gradient_at_level(level)
        print("Freeze all gradients")

    def unfreeze_second_encder(self, i_level: int) -> None:
        self.sub_models[i_level].second_encoder.requires_grad_(True)
        print(f"Unfreeze second encoder at level {i_level}")

    def freeze_gradient_at_level(self, i_level: int) -> None:
        self.sub_models[i_level].encoder.requires_grad_(False)
        self.sub_models[i_level].second_encoder.requires_grad_(False)
        self.sub_models[i_level].decoder.requires_grad_(False)
        self.sub_models[i_level].quantizer.requires_grad_(False)
        self.sub_models[i_level].pre_quant.requires_grad_(False)
        self.sub_models[i_level].post_quant.requires_grad_(False)
        print(f"Freeze gradient at level {i_level}")

    def unfreeze_gradient_at_level(self, i_level: int) -> None:
        self.sub_models[i_level].encoder.requires_grad_(True)
        self.sub_models[i_level].second_encoder.requires_grad_(True)
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
        # print("x shape is ", x.shape)
        h = self.sub_models[i_level].encoder(x) # Access using dot notation
        # print("after encoder, h shape is ", h.shape)
        h = self.sub_models[i_level].pre_quant(h)
        # print("after pre_quant, h shape is ", h.shape)
        quant, indices, loss = self.sub_models[i_level].quantizer(h)
        # print("after quantizer, quant shape is ", quant.shape)
        g = self.sub_models[i_level].post_quant(quant)
        # print("after post_quant, g shape is ", g.shape)
        g = self.sub_models[i_level].decoder(g)
        # print("after decoder, g shape is ", g.shape)
        return g, indices, loss

    def at_level_forward_to_encoder(self, x: torch.FloatTensor, i_level: int, second_encoder: bool = False) -> torch.FloatTensor:
        if second_encoder:
            h = self.sub_models[i_level].second_encoder(x)
            h = self.sub_models[i_level].second_pre_quant(h)
        else:
            h = self.sub_models[i_level].encoder(x)
            h = self.sub_models[i_level].pre_quant(h)
        return h

    def at_level_forward_to_codebook(self, x: torch.FloatTensor, i_level: int, second_encoder: bool = False) -> torch.FloatTensor:
        x_embed = self.at_level_forward_to_encoder(x, i_level, second_encoder)
        quant, indices, loss = self.sub_models[i_level].quantizer(x_embed)
        return quant, indices, loss, x_embed
    
    def at_level_forward_to_decoder(self, x: torch.FloatTensor, i_level: int, second_encoder: bool = False) -> torch.FloatTensor:
        quant, indices, loss, x_embed = self.at_level_forward_to_codebook(x, i_level, second_encoder)
        x_hat = self.sub_models[i_level].post_quant(quant)
        x_hat = self.sub_models[i_level].decoder(x_hat)
        return x_hat, quant, indices, loss, x_embed

    def foward_to_decoder(self, pyramid_x: torch.FloatTensor, active_level: int, second_encoder: bool = False) -> torch.FloatTensor:

        x_fea_map_list = []
        x_embbding_list = []
        indices_list = []
        loss_list = []

        for current_level in range(active_level + 1):
            if current_level == 0:
                x_hat, quant, indices, loss, x_embed = self.at_level_forward_to_decoder(pyramid_x[current_level], current_level, second_encoder)
                x_fea_map_list.append(x_embed)
                x_embbding_list.append(quant)
                indices_list.append(indices)
                loss_list.append(loss)
            else:
                resample_x = F.interpolate(pyramid_x[current_level - 1], scale_factor=2, mode='trilinear', align_corners=False)
                input_x = pyramid_x[current_level] - resample_x

                output_x, quant, indices, loss, x_embed = self.at_level_forward_to_decoder(input_x, current_level, second_encoder)
                x_fea_map_list.append(x_embed)
                x_embbding_list.append(quant)
                indices_list.append(indices)
                loss_list.append(loss)

                x_hat = F.interpolate(x_hat, scale_factor=2, mode='trilinear', align_corners=False)
                x_hat = x_hat + output_x
        
        # # show every output's shape
        # for i in range(len(x_fea_map_list)):
        #     print(f"Level {i} feature map shape is {x_fea_map_list[i].shape}")
        #     print(f"Level {i} embedding shape is {x_embbding_list[i].shape}")
        #     print(f"Level {i} indices shape is {indices_list[i].shape}")
        #     print(f"Level {i} loss shape is {loss_list[i].shape}")
        # print(f"Output shape is {x_hat.shape}")

        return x_hat, x_fea_map_list, x_embbding_list, indices_list, loss_list

    def forward(self, pyramid_x: list, active_level: int) -> torch.FloatTensor:
        # pyramid_x is a list of tensorFloat, like [8*8*8, 16*16*16, 32*32*32, 64*64*64]
        # active_level is the level of the pyramid, like 0, 1, 2, 3
        
        assert active_level <= self.num_level
        x_hat = None
        indices_list = []
        loss_list = []

        for current_level in range(active_level + 1):
            if current_level == 0:
                x_hat, indices, loss = self.foward_at_level(pyramid_x[current_level], current_level)
                indices_list.append(indices)
                loss_list.append(loss)
            else:
                resample_x = F.interpolate(pyramid_x[current_level - 1], scale_factor=2, mode='trilinear', align_corners=False)
                input_x = pyramid_x[current_level] - resample_x
                output_x, indices, loss = self.foward_at_level(input_x, current_level)
                indices_list.append(indices)
                loss_list.append(loss)
                # upsample the x_hat to double the size in three dimensions
                x_hat = F.interpolate(x_hat, scale_factor=2, mode='trilinear', align_corners=False)
                x_hat = x_hat + output_x

        return x_hat, indices_list, loss_list
    
def compute_loss_alpha(loss_weights, level):
    # here for each level, we will compute the loss from the list of indices
    # loss for different level will be scaled by the level_decay
    # For example
    #  - If current_level is 0, the loss coefs will be [a_f, a_i, a_s, a_r] / (a_f + a_i + a_s + a_r)
    #  - If current_level is 1, the loss coefs will be 
    #       - level 0: [a_f, a_i, a_s, a_r] / (a_f + a_i + a_s + a_r) * level_decay
    #       - level 1: [a_f, a_i, a_s, a_r] / (a_f + a_i + a_s + a_r)
    #       - After normalization between levels
    #       - level 0: [a_f, a_i, a_s, a_r] / (a_f + a_i + a_s + a_r) * level_decay / (1+level_decay)
    #       - level 1: [a_f, a_i, a_s, a_r] / (a_f + a_i + a_s + a_r) / (1+level_decay)
    #  - If current_level is 2, the loss coefs will be
    #       - level 0: [a_f, a_i, a_s, a_r] / (a_f + a_i + a_s + a_r) * level_decay^2 / (1+level_decay+level_decay^2)
    #       - level 1: [a_f, a_i, a_s, a_r] / (a_f + a_i + a_s + a_r) * level_decay / (1+level_decay+level_decay^2)
    #       - level 2: [a_f, a_i, a_s, a_r] / (a_f + a_i + a_s + a_r) / (1+level_decay+level_decay^2)

    a_f = loss_weights["dE_loss_alpha_fea_map"]
    a_i = loss_weights["dE_loss_alpha_infoNCE"]
    a_s = loss_weights["dE_loss_alpha_similarity"]
    a_r = loss_weights["dE_loss_alpha_recon"]
    a_e = loss_weights["de_loss_alpha_Eucsim"]
    a_sum = a_f + a_i + a_s + a_r + a_e
    level_decay = loss_weights["dE_loss_level_decay"]

    loss_alpha = []
    # level_coef is the sum of 1, level_decay, level_decay^2, level_decay^3, ...
    level_coef = 1
    level_sum = 0
    for i in range(level+1):
        print(level_coef)
        level_sum += level_coef
        level_coef *= level_decay

    for i in range(level+1):
        current_alpha = np.asarray([a_f, a_i, a_s, a_r, a_e]) / a_sum
        for j in range(level-i):
            current_alpha *= level_decay
        loss_alpha.append(current_alpha / level_sum)

    return loss_alpha

def train_model_at_level(current_level, global_config, model, optimizer_weights):

    pyramid_batch_size = global_config['pyramid_batch_size']
    pyramid_learning_rate = global_config['pyramid_learning_rate']
    pyramid_weight_decay = global_config['pyramid_weight_decay']
    pyramid_num_epoch = global_config['pyramid_num_epoch']
    # pyramid_freeze_previous_stages = global_config['pyramid_freeze_previous_stages']
    # VQ_train_gradiernt_clip = global_config['VQ_train_gradiernt_clip']
    # pyramid_codebook_size = global_config['pyramid_codebook_size']
    val_per_epoch = global_config['val_per_epoch']
    # tag = global_config['tag']
    save_per_epoch = global_config['save_per_epoch']
    # wandb_run = global_config['wandb_run']
    logger = global_config['logger']
    save_folder = global_config['save_folder']

    # dE_loss_alpha_fea_map: 1.0
    # dE_loss_alpha_infoNCE: 1.0
    # dE_loss_alpha_similarity: 0.0
    # dE_loss_alpha_recon: 1.0
    # dE_loss_level_decay: 0.5
    loss_weights = {
        "dE_loss_alpha_fea_map": global_config['dE_loss_alpha_fea_map'],
        "dE_loss_alpha_infoNCE": global_config['dE_loss_alpha_infoNCE'],
        "dE_loss_alpha_similarity": global_config['dE_loss_alpha_similarity'],
        "dE_loss_alpha_recon": global_config['dE_loss_alpha_recon'],
        "de_loss_alpha_Eucsim": global_config['de_loss_alpha_Eucsim'],
        "dE_loss_level_decay": global_config['dE_loss_level_decay'],
    }

    loss_alpha = compute_loss_alpha(loss_weights, current_level)

    best_val_loss = 1e6

    

    # set the data loaders for the current level
    train_loader, val_loader = build_dataloader_train_val_PET_CT(pyramid_batch_size[current_level], global_config)
    # set the optimizer for the current level
    optimizer = build_optimizer(model, pyramid_learning_rate[current_level], pyramid_weight_decay[current_level])

    if optimizer_weights is not None:
        optimizer.load_state_dict(optimizer_weights)
        print("Load optimizer weights")

    
    num_train_batch = len(train_loader)
    num_val_batch = len(val_loader)
    # unfreeze the gradient at the current level for the second encoder
    model.unfreeze_second_encder(current_level)
    input_modality = global_config["input_modality"]

    # start the training
    for idx_epoch in range(pyramid_num_epoch[current_level]):
        model.train()
        epoch_loss_train = {
            "fea_map": [],
            "infoNCE": [],
            "similarity": [],
            "recon": [],
            "euc_sim": [],
            "total": [],
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

            # generate the input data pyramid
            pyramid_x = generate_input_data_pyramid(x, current_level, global_config)
            pyramid_y = generate_input_data_pyramid(y, current_level, global_config)

            # foward the pyramid_x to the model
            x_hat, x_fea_map_list, x_embbding_list, x_indices_list, _ = model.foward_to_decoder(pyramid_x, current_level, second_encoder=True)
            y_hat, y_fea_map_list, y_embbding_list, y_indices_list, _ = model.foward_to_decoder(pyramid_y, current_level, second_encoder=False)
            
            # compute loss per level
            batch_overall_loss = 0
            batch_fea_map_loss = []
            batch_infoNCE_loss = []
            batch_similarity_loss = []
            batch_recon_loss = []
            batch_euc_sim_loss = []
            batch_total_loss = []
            for i_level in range(current_level+1):
                fea_map_loss = F.mse_loss(x_fea_map_list[i_level], y_fea_map_list[i_level])
                # indice_pair_list = [(x_indices_list[i_level][i], y_indices_list[i_level][i]) for i in range(len(x_indices_list[i_level]))]
                # infoNCE_loss = model.compute_InfoNCE_loss(indice_pair_list, i_level)
                infoNCE_loss = torch.tensor(0.0, device=device)
                similarity_loss = torch.abs(F.cosine_similarity(x_embbding_list[i_level], y_embbding_list[i_level], dim=-1).mean())
                recon_loss = F.l1_loss(x_hat, y_hat)
                Eucildean_similarity = F.pairwise_distance(x_embbding_list[i_level], y_embbding_list[i_level], p=2).mean()
                total_loss = fea_map_loss * loss_alpha[i_level][0] + \
                             infoNCE_loss * loss_alpha[i_level][1] + \
                             similarity_loss * loss_alpha[i_level][2] + \
                             recon_loss * loss_alpha[i_level][3] + \
                             Eucildean_similarity * loss_alpha[i_level][4]
                batch_overall_loss += total_loss
                batch_fea_map_loss.append(fea_map_loss.item())
                batch_infoNCE_loss.append(infoNCE_loss.item())
                batch_similarity_loss.append(similarity_loss.item())
                batch_recon_loss.append(recon_loss.item())
                batch_euc_sim_loss.append(Eucildean_similarity.item())
                batch_total_loss.append(total_loss.item())
            
            epoch_loss_train["fea_map"].append(batch_fea_map_loss)
            epoch_loss_train["infoNCE"].append(batch_infoNCE_loss)
            epoch_loss_train["similarity"].append(batch_similarity_loss)
            epoch_loss_train["recon"].append(batch_recon_loss)
            epoch_loss_train["euc_sim"].append(batch_euc_sim_loss)
            epoch_loss_train["total"].append(batch_total_loss)

            # print the loss
            current_fea_map_loss = np.asarray(batch_fea_map_loss).sum()
            current_infoNCE_loss = np.asarray(batch_infoNCE_loss).sum()
            current_similarity_loss = np.asarray(batch_similarity_loss).sum()
            current_recon_loss = np.asarray(batch_recon_loss).sum()
            current_euc_sim_loss = np.asarray(batch_euc_sim_loss).sum()
            current_total_loss = np.asarray(batch_total_loss).sum()
            loss_message = f"<{idx_epoch+1}> [{idx_batch+1}/{num_train_batch}] " + \
                            f"Total: {current_total_loss:.4f}, " + \
                            f"Fea_map : {current_fea_map_loss:.4f}, " + \
                            f"InfoNCE : {current_infoNCE_loss:.4f}, " + \
                            f"Cos_sim : {current_similarity_loss:.4f}, " + \
                            f"Recon : {current_recon_loss:.4f}, " + \
                            f"Euc_sim : {current_euc_sim_loss:.4f}"
            print(loss_message)

            # initialize the optimizer
            optimizer.zero_grad()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # backpropagation
            total_loss.backward()
            # optimizer step
            optimizer.step()

        
        for key in epoch_loss_train.keys():
            current_key_loss_list = []
            for sub_list in epoch_loss_train[key]:
                current_key_loss_list.append(np.asarray(sub_list).sum())
            current_key_loss = np.asarray(current_key_loss_list).mean()
            # msg = f"{current_key_loss:.4f}"
            # logger.log(idx_epoch, f"train_{key}_mean", msg)
        

        # for key in epoch_loss_train.keys():
        #     epoch_loss_train[key] = np.asanyarray(epoch_loss_train[key])
        #     logger.log(idx_epoch, f"train_{key}_mean", epoch_loss_train[key].mean())
        #     # logger.log(idx_epoch, f"train_{key}_std", epoch_loss_train[key].std())
        
        
        # for key in epoch_codebook_train.keys():
        #     epoch_codebook_train[key] = np.asanyarray(epoch_codebook_train[key])
        
        # activated_value, activated_counts = np.unique(epoch_codebook_train["indices"], return_counts=True)
        # if len(activated_counts) < pyramid_codebook_size[current_level]:
        #     activated_counts = np.append(activated_counts, np.zeros(pyramid_codebook_size[current_level] - len(activated_counts)))
        # effective_num = effective_number_of_classes(activated_counts / np.sum(activated_counts))
        # embedding_num = len(activated_counts)
        # logger.log(idx_epoch, "train_effective_num", effective_num)
        # logger.log(idx_epoch, "train_embedding_num", embedding_num)
        

        # validation
        if idx_epoch % val_per_epoch == 0:
            model.eval()
            epoch_loss_val = {
                "fea_map": [],
                "infoNCE": [],
                "similarity": [],
                "recon": [],
                "euc_sim": [],
                "total": [],
            }
            with torch.no_grad():
                for idx_batch, batch in enumerate(val_loader):
                    if batch is None:
                        continue
                    y = batch["CT"].to(device)
                    x = batch["PET_raw"].to(device)
                    for modality in input_modality:
                        if modality != "PET_raw" and modality != "CT":
                            x = torch.cat((x, batch[modality].to(device)), dim=1)
                    pyramid_x = generate_input_data_pyramid(x, current_level, global_config)
                    pyramid_y = generate_input_data_pyramid(y, current_level, global_config)
                    x_hat, x_fea_map_list, x_embbding_list, x_indices_list, _ = model.foward_to_decoder(pyramid_x, current_level, second_encoder=True)
                    y_hat, y_fea_map_list, y_embbding_list, y_indices_list, _ = model.foward_to_decoder(pyramid_y, current_level, second_encoder=False)
                    batch_overall_loss = 0
                    batch_fea_map_loss = []
                    batch_infoNCE_loss = []
                    batch_similarity_loss = []
                    batch_recon_loss = []
                    batch_euc_sim_loss = []
                    batch_total_loss = []
                    for i_level in range(current_level+1):
                        fea_map_loss = F.mse_loss(x_fea_map_list[i_level], y_fea_map_list[i_level])
                        # indice_pair_list = [(x_indices_list[i_level][i], y_indices_list[i_level][i]) for i in range(len(x_indices_list[i_level]))]
                        # infoNCE_loss = model.compute_InfoNCE_loss(indice_pair_list, i_level)
                        infoNCE_loss = torch.tensor(0.0, device=device)
                        similarity_loss = torch.abs(F.cosine_similarity(x_embbding_list[i_level], y_embbding_list[i_level], dim=-1).mean())
                        recon_loss = F.l1_loss(x_hat, y_hat)
                        Eucildean_similarity = F.pairwise_distance(x_embbding_list[i_level], y_embbding_list[i_level], p=2).mean()
                        total_loss = fea_map_loss * loss_alpha[i_level][0] + \
                                     infoNCE_loss * loss_alpha[i_level][1] + \
                                     similarity_loss * loss_alpha[i_level][2] + \
                                     recon_loss * loss_alpha[i_level][3] + \
                                     Eucildean_similarity * loss_alpha[i_level][4]
                        batch_overall_loss += total_loss
                        batch_fea_map_loss.append(fea_map_loss.item())
                        batch_infoNCE_loss.append(infoNCE_loss.item())
                        batch_similarity_loss.append(similarity_loss.item())
                        batch_recon_loss.append(recon_loss.item())
                        batch_euc_sim_loss.append(Eucildean_similarity.item())
                        batch_total_loss.append(total_loss.item())
                    epoch_loss_val["fea_map"].append(batch_fea_map_loss)
                    epoch_loss_val["infoNCE"].append(batch_infoNCE_loss)
                    epoch_loss_val["similarity"].append(batch_similarity_loss)
                    epoch_loss_val["recon"].append(batch_recon_loss)
                    epoch_loss_val["euc_sim"].append(batch_euc_sim_loss)
                    epoch_loss_val["total"].append(batch_total_loss)
            
            for key in epoch_loss_val.keys():
                current_key_loss_list = []
                for sub_list in epoch_loss_val[key]:
                    current_key_loss_list.append(np.asarray(sub_list).sum())
                current_key_loss = np.asarray(current_key_loss_list).mean()
                msg = f"{current_key_loss:.4f}"
                logger.log(idx_epoch, f"val_{key}_mean", msg)
            
            # if the current val loss is the best, save the model
            current_val_loss = np.asarray(epoch_loss_val["total"]).mean()
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                model_save_name = save_folder+f"best_model_{idx_epoch}_state_dict.pth"
                optimizer_save_name = save_folder+f"best_optimizer_{idx_epoch}_state_dict.pth"
                torch.save(model.state_dict(), model_save_name)
                torch.save(optimizer.state_dict(), optimizer_save_name)
                # # log the model
                # wandb_run.log_model(path=model_save_name, name=f"model_best_save", aliases=tag+f"_{current_level}")
                # wandb_run.log_model(path=optimizer_save_name, name=f"optimizer_best_save", aliases=tag+f"_{current_level}")
                # logger.log(idx_epoch, "model_saved", f"model_{idx_epoch}_state_dict.pth")

            # plot the x_hat using plot_and_save_x_xrec
            plot_and_save_x_xrec(pyramid_x[current_level], 
                                 x_hat, 
                                 num_per_direction=3, 
                                 savename=save_folder+f"val_{idx_epoch}_x_xrec.png")

        # save the model every save_per_epoch
        if idx_epoch % save_per_epoch == 0:
            # delete previous model
            for f in glob.glob(save_folder+"latest_*"):
                os.remove(f)
            model_save_name = save_folder+f"latest_model_{idx_epoch}_state_dict.pth"
            optimizer_save_name = save_folder+f"latest_optimizer_{idx_epoch}_state_dict.pth"
            torch.save(model.state_dict(), model_save_name)
            torch.save(optimizer.state_dict(), optimizer_save_name)
            # log the model
            # wandb_run.log_model(path=model_save_name, name=f"model_latest_save", aliases=tag+f"_{current_level}")
            # wandb_run.log_model(path=optimizer_save_name, name=f"optimizer_latest_save", aliases=tag+f"_{current_level}")
            # logger.log(idx_epoch, "model_saved", f"model_{idx_epoch}_state_dict.pth")

def plot_and_save_x_xrec(x, xrec, num_per_direction=1, savename=None):
    numpy_x = x[0, 0, :, :, :].cpu().numpy().squeeze()
    numpy_xrec = xrec[0, 0, :, :, :].cpu().numpy().squeeze()
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
    # wandb_run.log({wandb_name: fig})
    plt.close()
    print(f"Save the plot to {savename}")

def parse_yaml_arguments():
    parser = argparse.ArgumentParser(description='Train a 3D ViT-VQGAN model.')
    parser.add_argument('--config_file_path', type=str, default="config_v5_mini16_nonfixed.yaml")
    return parser.parse_args()

def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def main():
    # args = parse_arguments()
    # global_config = vars(args)
    config_file_path = parse_yaml_arguments().config_file_path
    global_config = load_yaml_config(config_file_path)
    global_config['pyramid_mini_resolution'] = global_config['volume_size'] // 2**(len(global_config['pyramid_channels'])-1)
    pyramid_channels = global_config['pyramid_channels']
    tag = global_config['tag']
    save_folder = global_config['save_folder']
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    global_config["save_folder"] = f"{save_folder}/{time_stamp}/{tag}/"
    os.makedirs(global_config['save_folder'], exist_ok=True)

    # copy the current config file to the save folder
    copy_command = "cp " + config_file_path + " " + global_config['save_folder']
    print("Copy the config file to the save folder")
    os.system(copy_command)

    # set the random seed
    random.seed(global_config['random_seed'])
    np.random.seed(global_config['random_seed'])
    torch.manual_seed(global_config['random_seed'])

    # # initialize wandb
    # wandb.login(key = "41c33ee621453a8afcc7b208674132e0e8bfafdb")
    # wandb_run = wandb.init(project="CT_ViT_VQGAN", dir=os.getenv("WANDB_DIR", "cache/wandb"), config=global_config)
    # wandb_run.log_code(root=".", name=tag+"train_v5.py")
    # global_config["wandb_run"] = wandb_run

    # set the logger
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file_path = f"train_log_{current_time}.json"
    logger = simple_logger(log_file_path, global_config)
    global_config["logger"] = logger

    # set the model
    model_levels = generate_model_levels(global_config)
    model = ViTVQ3D_dualEncoder(model_levels=model_levels)

    # load model from the previous training
    state_dict_model_path = global_config['state_dict_model_path']
    vitvq3d_model = ViTVQ3D(model_level=model_levels)  # Initialize the original model
    vitvq3d_checkpoint = torch.load(state_dict_model_path, map_location="cpu") # Load the checkpoint
    vitvq3d_model.load_state_dict(vitvq3d_checkpoint)  # Load the state dictionary

    # transfer the weights from the original model to the new model
    # Assuming both models have the same structure in terms of number of levels
    for level_idx, (dual_sub_model, vit_sub_model) in enumerate(zip(model.sub_models, vitvq3d_model.sub_models)):
        # Copy the weights for the first encoder, decoder, quantizer, and other components
        dual_sub_model.encoder.load_state_dict(vit_sub_model.encoder.state_dict())
        dual_sub_model.decoder.load_state_dict(vit_sub_model.decoder.state_dict())
        dual_sub_model.quantizer.load_state_dict(vit_sub_model.quantizer.state_dict())
        dual_sub_model.pre_quant.load_state_dict(vit_sub_model.pre_quant.state_dict())
        dual_sub_model.post_quant.load_state_dict(vit_sub_model.post_quant.state_dict())
        
        # # Optionally, initialize the second encoder with the same weights as the first one
        # dual_sub_model.second_encoder.load_state_dict(vit_sub_model.encoder.state_dict())
        # dual_sub_model.second_pre_quant.load_state_dict(vit_sub_model.pre_quant.state_dict())

    print(f"Load model from {state_dict_model_path}")
    # model.pre_compute_InfoNCE_loss()
    # model.load_weights_for_module(state_dict_model_path)

    # move the model to the device
    model.to(device)

    # load previous trained epochs
    # # num_epoch is the number for each stage need to be trained, we need to find out which stage we are in
    # num_epoch = global_config['pyramid_num_epoch']
    # num_epoch_sum = 0
    # previous_epochs_trained = global_config['previous_epochs_trained']
    # for i in range(len(num_epoch)):
    #     num_epoch_sum += num_epoch[i]
    #     if num_epoch_sum >= previous_epochs_trained:
    #         break
    # current_level = i
    # print(f"Current level is {current_level}")
    # global_config["pyramid_num_epoch"][current_level] = num_epoch_sum - previous_epochs_trained
    # train_model_at_level(current_level, global_config, model, optimizer_weights=None)

    # if there are more stages to train
    # for i in range(current_level+1, len(pyramid_channels)):
    #     train_model_at_level(i, global_config, model, None)

    # start the training
    for i in range(len(pyramid_channels)):
        train_model_at_level(i, global_config, model, None)

    # wandb.finish()

if __name__ == "__main__":
    main()