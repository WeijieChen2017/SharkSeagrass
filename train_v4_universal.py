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

from train_v4_utils import UNet3D_encoder, UNet3D_decoder, build_dataloader_train_val
from train_v4_utils import plot_and_save_x_xrec, simple_logger, effective_number_of_classes

from vector_quantize_pytorch import VectorQuantize as lucidrains_VQ

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
        "reconL2": global_config['VQ_loss_weight_recon_L2'],
        "reconL1": global_config['VQ_loss_weight_recon_L1'],
        "codebook": global_config['VQ_loss_weight_codebook'],
    }
    logger = global_config['logger']
    save_folder = global_config['save_folder']

    best_val_loss = 1e6

    # set the data loaders for the current level
    train_loader, val_loader = build_dataloader_train_val(pyramid_batch_size[current_level], global_config)
    # set the optimizer for the current level
    optimizer = build_optimizer(model, pyramid_learning_rate[current_level], pyramid_weight_decay[current_level])

    if optimizer_weights is not None:
        optimizer.load_state_dict(optimizer_weights)
        print("Load optimizer weights")

    num_train_batch = len(train_loader)
    num_val_batch = len(val_loader)

    # set the gradient freeze
    if pyramid_freeze_previous_stages:
        model.freeze_gradient_all()
        model.unfreeze_gradient_at_level(current_level)
    else:
        model.freeze_gradient_all()
        for i_level in range(current_level):
            model.unfreeze_gradient_at_level(i_level)

    # start the training
    for idx_epoch in range(pyramid_num_epoch[current_level]):
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
                    pyramid_x = generate_input_data_pyramid(x, current_level)
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
            plot_and_save_x_xrec(target_x, xrec, num_per_direction=3, savename=save_folder+f"{save_name}_{current_level}.png", wandb_name="val_snapshots")
            
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
            "spatial_dims": 3, "in_channels": 1,
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
        pyramid_x.append(x_at_level)
    
    return pyramid_x

def build_optimizer(model, learning_rate, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

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
        
        self.init_weights()
        self.freeze_gradient_all()

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

    def forward(self, pyramid_x: list, active_level: int) -> torch.FloatTensor:
        # pyramid_x is a list of tensorFloat, like [8*8*8, 16*16*16, 32*32*32, 64*64*64]
        # active_level is the level of the pyramid, like 0, 1, 2, 3
        
        assert active_level <= self.num_level
        x_hat = None
        indices_list = []
        loss_list = []

        for current_level in range(active_level):
            if current_level == 0:
                x_hat, indices, loss = self.foward_at_level(pyramid_x[current_level], current_level)
                indices_list.append(indices)
                loss_list.append(loss)
            else:
                resample_x = F.interpolate(pyramid_x[current_level], scale_factor=2, mode='trilinear', align_corners=False)
                input_x = pyramid_x[current_level] - resample_x
                output_x, indices, loss = self.foward_at_level(input_x, current_level)
                indices_list.append(indices)
                loss_list.append(loss)
                # upsample the x_hat to double the size in three dimensions
                x_hat = F.interpolate(x_hat, scale_factor=2, mode='trilinear', align_corners=False)
                x_hat = x_hat + output_x

        return x_hat, indices_list, loss_list

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a 3D ViT-VQGAN model.')
    parser.add_argument('--tag', type=str, default="pyramid_mini16_fixed")
    parser.add_argument('--random_seed', type=int, default=426)
    parser.add_argument('--volume_size', type=int, default=64)
    parser.add_argument('--pix_dim', type=float, default=1.5)
    parser.add_argument('--num_workers_train_dataloader', type=int, default=8)
    parser.add_argument('--num_workers_val_dataloader', type=int, default=4)
    parser.add_argument('--num_workers_train_cache_dataset', type=int, default=8)
    parser.add_argument('--num_workers_val_cache_dataset', type=int, default=4)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_val', type=int, default=16)
    parser.add_argument('--cache_ratio_train', type=float, default=0.2)
    parser.add_argument('--cache_ratio_val', type=float, default=0.2)
    parser.add_argument('--val_per_epoch', type=int, default=50)
    parser.add_argument('--save_per_epoch', type=int, default=100)
    parser.add_argument('--IS_LOGGER_WANDB', type=bool, default=True)
    parser.add_argument('--VQ_optimizer', type=str, default="AdamW")
    parser.add_argument('--VQ_loss_weight_recon_L2', type=float, default=0.1)
    parser.add_argument('--VQ_loss_weight_recon_L1', type=float, default=1.0)
    parser.add_argument('--VQ_loss_weight_codebook', type=float, default=0.1)
    parser.add_argument('--VQ_train_gradiernt_clip', type=float, default=1.0)
    parser.add_argument('--pyramid_channels', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--pyramid_codebook_size', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--pyramid_strides', type=int, nargs='+', default=[2, 2, 1])
    parser.add_argument('--pyramid_num_res_units', type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--pyramid_num_epoch', type=int, nargs='+', default=[500, 500, 500])
    parser.add_argument('--pyramid_batch_size', type=int, nargs='+', default=[128, 128, 16])
    parser.add_argument('--pyramid_learning_rate', type=float, nargs='+', default=[1e-3, 5e-4, 2e-4])
    parser.add_argument('--pyramid_weight_decay', type=float, nargs='+', default=[1e-4, 5e-5, 2e-5])
    parser.add_argument('--pyramid_freeze_previous_stages', type=bool, default=True)
    parser.add_argument('--save_folder', type=str, default="./results/")
    return parser.parse_args()

def parse_yaml_arguments():
    parser = argparse.ArgumentParser(description='Train a 3D ViT-VQGAN model.')
    parser.add_argument('--config_file_path', type=str, default="config_v4_mini8_fixed.yaml")
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
    wandb_run.log_code(root=".", name=tag+"train_v4_universal.py")
    global_config["wandb_run"] = wandb_run

    # set the logger
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file_path = f"train_log_{current_time}.json"
    logger = simple_logger(log_file_path, global_config)
    global_config["logger"] = logger

    # set the model
    model_levels = generate_model_levels(global_config)
    model = ViTVQ3D(model_level=model_levels)

    # load model from the previous training
    if global_config["load_checkpoints"]:
        model_artifact_name = global_config["model_artifact_name"]
        model_artifact_version = global_config["model_artifact_version"]
        optim_artifact_name = global_config["optim_artifact_name"]
        optim_artifact_version = global_config["optim_artifact_version"]
        model_checkpoint_name = model_artifact_name+":"+model_artifact_version
        optim_checkpoint_name = optim_artifact_name+":"+optim_artifact_version
        for artifact_name in [model_checkpoint_name, optim_checkpoint_name]:
            artifact = wandb_run.use_artifact(f"convez376/CT_ViT_VQGAN/{artifact_name}")
            artifact_dir = artifact.download()

        # search the model and optimizer checkpoint
        state_dict_model_path = glob.glob("./artifacts/"+model_checkpoint_name+"/"+"*.pth")[0]
        state_dict_optim_path = glob.glob("./artifacts/"+optim_checkpoint_name+"/"+"*.pth")[0]
        print(state_dict_model_path)
        print(state_dict_optim_path)
        state_dict_model = torch.load(state_dict_model_path)
        state_dict_optim = torch.load(state_dict_optim_path)
        
        # print the model state_dict loaded from the checkpoint
        print("Model state_dict loaded from the checkpoint: ")
        print(state_dict_model_path)
        print("The following keys are loaded: ")
        for key in state_dict_model.keys():
            print(key)
        model.load_state_dict(state_dict_model).to(device)

        # load previous trained epochs
        # num_epoch is the number for each stage need to be trained, we need to find out which stage we are in
        num_epoch = global_config['pyramid_num_epoch']
        num_epoch_sum = 0
        previous_training_epoch = global_config['previous_training_epoch']
        for i in range(len(num_epoch)):
            num_epoch_sum += num_epoch[i]
            if num_epoch_sum >= previous_training_epoch:
                break
        current_level = i
        print(f"Current level is {current_level}")
        optimizer_weights = state_dict_optim
        global_config["pyramid_num_epoch"][current_level] = num_epoch_sum - previous_training_epoch
        train_model_at_level(current_level, global_config, model, optimizer_weights)

        # if there are more stages to train
        for i in range(current_level+1, len(pyramid_channels)):
            train_model_at_level(i, global_config, model, None)

    else:
        model.to(device)
        for current_level in range(len(pyramid_channels)):
            # current level starts at 1
            train_model_at_level(current_level, global_config, model, None)

    wandb.finish()