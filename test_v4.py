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
import nibabel as nib
import torch.nn.functional as F

from train_v4_utils import UNet3D_encoder, UNet3D_decoder
from train_v4_utils import plot_and_save_x_xrec, simple_logger, effective_number_of_classes
from vector_quantize_pytorch import VectorQuantize as lucidrains_VQ

# import slide_window_inference from monai
from monai.inferers import sliding_window_inference

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import confusion_matrix
from skimage.metrics import mean_squared_error


from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged
from typing import Dict, Any
import json
import os


MAX_DEPTH = 3
MINI_RES = 16


def denorm_CT(data):
    data *= 4000
    data -= 1024
    return data

def rmse(x,y):
    return np.mean(np.sqrt(np.sum(np.square(x-y))))

def nrmse(x,y):
    # compute the normalized root mean squared error
    return rmse(x,y) / (np.max(x) - np.min(x))

def mse(x,y):
    return mean_squared_error(x,y)

def mae(x,y):
    return np.mean(np.absolute(x-y))

def acutance(x):
    return np.mean(np.absolute(sobel(x)))

def dice_coe(x, y, tissue="air"):
    if tissue == "air":
        x_mask = filter_data(x, -1124, -500)
        y_mask = filter_data(y, -1124, -500)
    if tissue == "soft":
        x_mask = filter_data(x, -500, 250)
        y_mask = filter_data(y, -500, 250)
    if tissue == "bone":
        x_mask = filter_data(x, 250, 3100)
        y_mask = filter_data(y, 250, 3100)
    CM = confusion_matrix(np.ravel(x_mask), np.ravel(y_mask))
    TN, FP, FN, TP = CM.ravel()
    return 2*TP / (2*TP + FN + FP)

def filter_data(data, range_min, range_max):
    mask_1 = data < range_max
    mask_2 = data > range_min
    mask_1 = mask_1.astype(int)
    mask_2 = mask_2.astype(int)
    mask = mask_1 * mask_2
    return mask

def build_dataloader_test(batch_size: int, global_config: Dict[str, Any]) -> DataLoader:
    pix_dim = global_config["pix_dim"]
    num_workers_test_cache_dataset = global_config["num_workers_test_cache_dataset"]
    num_workers_test_dataloader = global_config["num_workers_test_dataloader"]
    cache_ratio_test = global_config["cache_ratio_test"]
    
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(pix_dim, pix_dim, pix_dim),
                mode=("bilinear"),
            ),
            ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=2976, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    # load data_chunks.json and specify chunk_0 to chunk_4 for training, chunk_5 to chunk_7 for validation, chunk_8 and chunk_9 for testing
    with open("data_chunks.json", "r") as f:
        data_chunk = json.load(f)

    test_files = []

    for i in range(8, 10):
        test_files.extend(data_chunk[f"chunk_{i}"])

    num_test_files = len(test_files)

    print("Test files are ", len(test_files))

    test_ds = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_num=num_test_files,
        cache_rate=cache_ratio_test,  # 600 * 0.1 = 60
        num_workers=num_workers_test_cache_dataset,
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers_test_dataloader)

    return test_loader



def compute_metrics(global_config, data_x, data_y, idx_batch):

    # input is between 0 to 1, x is the prediction, y is the ground truth

    metrics_name_list = global_config['metrics']
    logger = global_config['logger']
    metrics_value_list = dict()

    denormed_data_x = denorm_CT(data_x)
    denormed_data_y = denorm_CT(data_y)
    positive_data_x = np.clip(denormed_data_x+1024, 0, 4000)
    positive_data_y = np.clip(denormed_data_y+1024, 0, 4000)

    # - MAE
    # - PSNR
    # - SSIM
    # - Actuance
    # - Dice_air
    # - Dice_bone
    # - Dice_soft
    if "MAE" in metrics_name_list:
        mae_value = mae(denormed_data_x, denormed_data_y)
        metrics_value_list["MAE"] = mae_value
        logger.log(idx_batch, "MAE", mae_value)
    if "PSNR" in metrics_name_list:
        psnr_value = psnr(positive_data_x, positive_data_y, data_range=4000)
        metrics_value_list["PSNR"] = psnr_value
        logger.log(idx_batch, "PSNR", psnr_value)
    if "SSIM" in metrics_name_list:
        ssim_value = ssim(positive_data_x, positive_data_y, data_range=4000)
        metrics_value_list["SSIM"] = ssim_value
        logger.log(idx_batch, "SSIM", ssim_value)
    if "Actuance" in metrics_name_list:
        actuance_value = acutance(denormed_data_x)
        metrics_value_list["Actuance"] = actuance_value
        logger.log(idx_batch, "Actuance", actuance_value)
    if "Dice_air" in metrics_name_list:
        dice_air_value = dice_coe(denormed_data_x, denormed_data_y, tissue="air")
        metrics_value_list["Dice_air"] = dice_air_value
        logger.log(idx_batch, "Dice_air", dice_air_value)
    if "Dice_bone" in metrics_name_list:
        dice_bone_value = dice_coe(denormed_data_x, denormed_data_y, tissue="bone")
        metrics_value_list["Dice_bone"] = dice_bone_value
        logger.log(idx_batch, "Dice_bone", dice_bone_value)
    if "Dice_soft" in metrics_name_list:
        dice_soft_value = dice_coe(denormed_data_x, denormed_data_y, tissue="soft")
        metrics_value_list["Dice_soft"] = dice_soft_value
        logger.log(idx_batch, "Dice_soft", dice_soft_value)
    
    return metrics_value_list

def test_model(global_config, model):

    volume_size = global_config['volume_size']
    save_folder = global_config['save_folder']

    # set the data loaders for the current level
    test_batch_size = 1
    test_loader = build_dataloader_test(test_batch_size, global_config)
    num_test = len(test_loader)

    # ========================== ========================== ========================== ========================== ==========================

    model.eval()
    case_metrics_list = []

    for idx_batch, batch in enumerate(test_loader):
        x, meta = batch
        x = batch["image"].to(device)
        meta = batch["image_meta_dict"]
        print("Processing case ", idx_batch+1, "/", num_test)
        print("Current case is: ", meta)
        # load the ct_file
        ct_path = meta["filename_or_obj"][0]
        ct_file = nib.load(ct_path)
        ct_filename = os.path.basename(ct_path)

        # image_meta = batch["image_meta"]
        # print(f"Processing case {idx_batch+1}/{num_test}", image_meta)
        # print("Current case is: ", ct_path)
        # ct_file = nib.load(ct_path)
        # ct_filename = os.path.basename(ct_path)

        with torch.no_grad():
            # xrec, indices_list, cb_loss_list = model(pyramid_x, max_depth)
            # x_hat = infer(input_data=x, model=model)
            x_hat = sliding_window_inference(
                inputs = x,
                predictor = model,
                roi_size = (volume_size, volume_size, volume_size),
                sw_batch_size=1, 
                overlap=0.25, 
                mode="gaussian", 
                sigma_scale=0.125, 
                padding_mode="constant"
            )
        
        # compute the metrics
        metrics_value_list = compute_metrics(global_config, x_hat, x, idx_batch)
        print(f"Case {idx_batch+1}/{num_test} metrics: ")
        for key in metrics_value_list.keys():
            print(f"--->{key}: {metrics_value_list[key]:4f}")
        case_metrics_list.append(metrics_value_list)

        # save x_hat using the ct_file header and affine
        x_hat_filename = os.path.join(save_folder, f"{ct_filename}_recon.nii.gz")
        x_hat_nii = nib.Nifti1Image(x_hat, ct_file.affine, ct_file.header)
        nib.save(x_hat_nii, x_hat_filename)
        print(f"Reconstructed image saved to {x_hat_filename}")

    print()
    # compute the average metrics
    average_metrics = dict()
    for key in case_metrics_list[0].keys():
        average_metrics[key] = np.mean([case[key] for case in case_metrics_list])
        print(f"Average {key}: {average_metrics[key]:4f}")
    
    print("All cases are processed")

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

def generate_input_data_pyramid(x: torch.FloatTensor):
    levels = MAX_DEPTH
    pyramid_mini_resolution = MINI_RES
    pyramid_x = []
    for i in range(levels):
        x_at_level = F.interpolate(x, size=(pyramid_mini_resolution*2**i,
                                            pyramid_mini_resolution*2**i, 
                                            pyramid_mini_resolution*2**i), mode="trilinear", align_corners=False).to(device)
        # print(f"Level {i} shape is {x`_at_level.shape}")
        pyramid_x.append(x_at_level)
    
    return pyramid_x

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

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # pyramid_x is a list of tensorFloat, like [8*8*8, 16*16*16, 32*32*32, 64*64*64]
        # active_level is the level of the pyramid, like 0, 1, 2, 3
        active_level = MAX_DEPTH
        x_hat = None
        # indices_list = []
        # loss_list = []

        pyramid_x = generate_input_data_pyramid(x)

        for current_level in range(active_level):
            if current_level == 0:
                x_hat, indices, loss = self.foward_at_level(pyramid_x[current_level], current_level)
                # indices_list.append(indices)
                # loss_list.append(loss)
            else:
                resample_x = F.interpolate(pyramid_x[current_level - 1], scale_factor=2, mode='trilinear', align_corners=False)
                input_x = pyramid_x[current_level] - resample_x
                output_x, indices, loss = self.foward_at_level(input_x, current_level)
                # indices_list.append(indices)
                # loss_list.append(loss)
                # upsample the x_hat to double the size in three dimensions
                x_hat = F.interpolate(x_hat, scale_factor=2, mode='trilinear', align_corners=False)
                x_hat = x_hat + output_x

        return x_hat

def parse_yaml_arguments():
    parser = argparse.ArgumentParser(description='Train a 3D ViT-VQGAN model.')
    parser.add_argument('--config_file_path', type=str, default="config_v4_mini16_nonfixed_test.yaml")
    return parser.parse_args()

def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def main():
    config_file_path = parse_yaml_arguments().config_file_path
    global_config = load_yaml_config(config_file_path)
    global_config['pyramid_mini_resolution'] = global_config['volume_size'] // 2**(len(global_config['pyramid_channels'])-1)
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
    model = ViTVQ3D(model_level=model_levels).to(device)

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

    # test the model
    test_model(global_config, model)

    # finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()