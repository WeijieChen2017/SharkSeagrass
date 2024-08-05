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

import wandb
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

from train_v4_utils import UNet3D_encoder, UNet3D_decoder
from train_v4_utils import plot_and_save_x_xrec, simple_logger, effective_number_of_classes

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



def collate_fn(batch, pet_valid_th=0.01):
    # batch is a list of list of dictionary
    idx = len(batch)
    jdx = len(batch[0])
    modalities = batch[0][0].keys()
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






def train_model_at_level(current_level, global_config, model, optimizer_weights):

    pyramid_batch_size = global_config['pyramid_batch_size']
    pyramid_learning_rate = global_config['pyramid_learning_rate']
    pyramid_weight_decay = global_config['pyramid_weight_decay']
    pyramid_num_epoch = global_config['pyramid_num_epoch']
    pyramid_freeze_previous_stages = global_config['pyramid_freeze_previous_stages']
    VQ_train_gradiernt_clip = global_config['VQ_train_gradiernt_clip']
    pyramid_codebook_size = global_config['pyramid_codebook_size']
    val_per_epoch = global_config['val_per_epoch']
    tag = global_config['tag']
    save_per_epoch = global_config['save_per_epoch']
    wandb_run = global_config['wandb_run']
    loss_weights = {
        "dE_l2": global_config['dE_loss_l2'],
        "dE_infoNCE": global_config['dE_loss_infoNCE'],
        "dE_commit": global_config['dE_loss_commit'],
    }
    logger = global_config['logger']
    save_folder = global_config['save_folder']

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
            "dE_l2": [],
            "dE_infoNCE": [],
            "dE_commit": [],
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
            xrec, x_indices_list, x_cb_loss_list, x_embed_list = model.foward_to_decoder(pyramid_x, current_level, second_encoder=True)
            yrec, y_indices_list, y_cb_loss_list, y_embed_list = model.foward_to_decoder(pyramid_y, current_level, second_encoder=False)
            
            # initialize the optimizer
            optimizer.zero_grad()
            # compute the loss
            
            
            
            
            
            
            
            
            
            
            
            
            reconL2_loss = F.mse_loss(target_x, xrec)
            reconL1_loss = F.l1_loss(target_x, xrec)
            if pyramid_freeze_previous_stages:
                codebook_loss = cb_loss_list[-1]
            else:
                # cb_loss_list is a list of tensor with gradient
                # Sum the tensors
                sum_codebook_loss = torch.stack(cb_loss_list).sum(dim=0)
                # Compute the average
                codebook_loss = sum_codebook_loss / len(cb_loss_list)
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
                epoch_codebook_train["indices"].extend(indices_list[-1].cpu().numpy().squeeze().flatten())
            else:
                for current_indices in indices_list:
                    epoch_codebook_train["indices"].extend(current_indices.cpu().numpy().squeeze().flatten())
        
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
                    pyramid_x = generate_input_data_pyramid(x, current_level, global_config)
                    # target_x is the last element of the pyramid_x, which is to be reconstructed
                    target_x = pyramid_x[-1]
                    xrec, indices_list, cb_loss_list = model(pyramid_x, current_level)
                    # compute the loss
                    reconL2_loss = F.mse_loss(target_x, xrec)
                    reconL1_loss = F.l1_loss(target_x, xrec)
                    if pyramid_freeze_previous_stages:
                        codebook_loss = cb_loss_list[-1]
                    else:
                        # cb_loss_list is a list of tensor with gradient
                        # Sum the tensors
                        sum_codebook_loss = torch.stack(cb_loss_list).sum(dim=0)
                        # Compute the average
                        codebook_loss = sum_codebook_loss / len(cb_loss_list)
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
                        epoch_codebook_val["indices"].extend(indices_list[-1].cpu().numpy().squeeze().flatten())
                    else:
                        for current_indices in indices_list:
                            epoch_codebook_val["indices"].extend(current_indices.cpu().numpy().squeeze().flatten())

            save_name = f"epoch_{idx_epoch}_batch_{idx_batch}"
            plot_and_save_x_xrec(target_x, xrec, 
                                 num_per_direction=3, 
                                 savename=save_folder+f"{save_name}_{current_level}.png", 
                                 wandb_name="val_snapshots",
                                 global_config=global_config)
            
            for key in epoch_loss_val.keys():
                epoch_loss_val[key] = np.asanyarray(epoch_loss_val[key])
                logger.log(idx_epoch, f"val_{key}_mean", epoch_loss_val[key].mean())
                # logger.log(idx_epoch, f"val_{key}_std", epoch_loss_val[key].std())

            if epoch_loss_val["total"].mean() < best_val_loss:
                best_val_loss = epoch_loss_val["total"].mean()
                model_save_name = save_folder+f"model_best_{idx_epoch}_state_dict_{current_level}.pth"
                optimizer_save_name = save_folder+f"optimizer_best_{idx_epoch}_state_dict_{current_level}.pth"
                torch.save(model.state_dict(), model_save_name)
                torch.save(optimizer.state_dict(), optimizer_save_name)
                # log the model
                wandb_run.log_model(path=model_save_name, name="model_best_eval", aliases=tag+f"_{current_level}")
                wandb_run.log_model(path=optimizer_save_name, name="optimizer_best_eval", aliases=tag+f"_{current_level}")
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
            # delete previous model
            for f in glob.glob(save_folder+"latest_*"):
                os.remove(f)
            model_save_name = save_folder+f"latest_model_{idx_epoch}_state_dict.pth"
            optimizer_save_name = save_folder+f"latest_optimizer_{idx_epoch}_state_dict.pth"
            torch.save(model.state_dict(), model_save_name)
            torch.save(optimizer.state_dict(), optimizer_save_name)
            # log the model
            wandb_run.log_model(path=model_save_name, name=f"model_latest_save", aliases=tag+f"_{current_level}")
            wandb_run.log_model(path=optimizer_save_name, name=f"optimizer_latest_save", aliases=tag+f"_{current_level}")
            logger.log(idx_epoch, "model_saved", f"model_{idx_epoch}_state_dict.pth")

def generate_model_levels(global_config):
    num_level = len(global_config['pyramid_channels'])
    model_level = []
    for i in range(num_level):
        encoder = {
            "spatial_dims": 3, "in_channels": len(global_config['input_modality']),
            "channels": global_config['pyramid_channels'][:i+1],
            "strides": global_config['pyramid_strides'][-(i+1):],
            "num_res_units": global_config['pyramid_num_res_units'][i],
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
        model_level.append({
            "encoder": encoder,
            "decoder": decoder,
            "quantizer": quantizer
        })
    return model_level

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

class ViTVQ3D_dualEncoder(nn.Module):
    def __init__(self, model_level: list) -> None:
        super().__init__()
        self.num_level = len(model_level)
        self.sub_models = nn.ModuleList()
        for level_setting in model_level:
            # Create a submodule to hold the encoder, decoder, quantizer, etc.
            sub_model = nn.Module() 
            sub_model.encoder = UNet3D_encoder(**level_setting["encoder"])
            sub_model.second_encoder = UNet3D_encoder(**level_setting["encoder"])
            sub_model.decoder = UNet3D_decoder(**level_setting["decoder"])
            sub_model.quantizer = lucidrains_VQ(**level_setting["quantizer"])
            sub_model.pre_quant = nn.Linear(level_setting["encoder"]["channels"][-1], level_setting["quantizer"]["dim"])
            sub_model.second_pre_quant = nn.Linear(level_setting["encoder"]["channels"][-1], level_setting["quantizer"]["dim"])
            sub_model.post_quant = nn.Linear(level_setting["quantizer"]["dim"], level_setting["decoder"]["channels"][0])
            
            # Append the submodule to the ModuleList
            self.sub_models.append(sub_model) 
        
        self.init_weights()
        self.freeze_gradient_all()

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
                x_hat, quant, indices, loss, x_embed = self.at_level_forward_to_decoder
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
    os.makedirs(global_config['save_folder'], exist_ok=True)

    # set the random seed
    random.seed(global_config['random_seed'])
    np.random.seed(global_config['random_seed'])
    torch.manual_seed(global_config['random_seed'])

    # initialize wandb
    wandb.login(key = "41c33ee621453a8afcc7b208674132e0e8bfafdb")
    wandb_run = wandb.init(project="CT_ViT_VQGAN", dir=os.getenv("WANDB_DIR", "cache/wandb"), config=global_config)
    wandb_run.log_code(root=".", name=tag+"train_v5.py")
    global_config["wandb_run"] = wandb_run

    # set the logger
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file_path = f"train_log_{current_time}.json"
    logger = simple_logger(log_file_path, global_config)
    global_config["logger"] = logger

    # set the model
    model_levels = generate_model_levels(global_config)
    model = ViTVQ3D_dualEncoder(model_level=model_levels).to(device)

    # # load model from the previous training
    state_dict_model_path = global_config['state_dict_model_path']
    print(state_dict_model_path)
    state_dict_model = torch.load(state_dict_model_path)
    
    # print the model state_dict loaded from the checkpoint
    print("Model state_dict loaded from the checkpoint: ")
    print(state_dict_model_path)
    print("The following keys are loaded: ")
    for key in state_dict_model.keys():
        print(key)
    model.load_state_dict(state_dict_model)
    print("Model state_dict loaded successfully")
    # model.to(device)

    # load previous trained epochs
    # num_epoch is the number for each stage need to be trained, we need to find out which stage we are in
    num_epoch = global_config['pyramid_num_epoch']
    num_epoch_sum = 0
    previous_epochs_trained = global_config['previous_epochs_trained']
    for i in range(len(num_epoch)):
        num_epoch_sum += num_epoch[i]
        if num_epoch_sum >= previous_epochs_trained:
            break
    current_level = i
    print(f"Current level is {current_level}")
    global_config["pyramid_num_epoch"][current_level] = num_epoch_sum - previous_epochs_trained
    train_model_at_level(current_level, global_config, model, optimizer_weights=None)

    # if there are more stages to train
    for i in range(current_level+1, len(pyramid_channels)):
        train_model_at_level(i, global_config, model, None)

    wandb.finish()

if __name__ == "__main__":
    main()