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

import argparse
import json
import time
import random
import numpy as np

from UNetUNet_v1_py2_train_util import VQModel, simple_logger, two_segment_scale, prepare_dataset




    

def main():
    # here I will use argparse to parse the arguments
    argparser = argparse.ArgumentParser(description='Prepare dataset for training')
    argparser.add_argument('--train_fold', type=str, default="0,1,2", help='Path to the training fold')
    argparser.add_argument('--val_fold', type=str, default="3", help='Path to the validation fold')
    argparser.add_argument('--test_fold', type=str, default="4", help='Path to the testing fold')
    args = argparser.parse_args()
    tag = f"fold{args.test_fold}"

    random_seed = 42
    # set the random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_loader_params = {
        "train": {
            "batch_size": 4,
            "shuffle": True,
            "num_workers_cache": 4,
            "num_workers_loader": 8,
            "cache_rate": 1.0,
        },
        "val": {
            "batch_size": 4,
            "shuffle": False,
            "num_workers_cache": 2,
            "num_workers_loader": 4,
            "cache_rate": 1.0,
        },
        "test": {
            "batch_size": 4,
            "shuffle": False,
            "num_workers_cache": 1,
            "num_workers_loader": 2,
            "cache_rate": 1.0,
        },
    }

    model_step1_params = {
        "VQ_NAME": "f4-noattn",
        "n_embed": 8192,
        "embed_dim": 3,
        "img_size" : 400,
        "input_modality" : ["TOFNAC", "CTAC"],
        "ckpt_path": "vq_f4-noattn.ckpt",
        "ddconfig": {
            "attn_type": "none",
            "double_z": False,
            "z_channels": 3,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 1,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }
    }

    wandb_config = {
        "train_fold": args.train_fold,
        "val_fold": args.val_fold,
        "test_fold": args.test_fold,
        "random_seed": random_seed,
        "model_step1_params": model_step1_params,
    }

    global_config = {}

    # initialize wandb
    wandb.login(key = "41c33ee621453a8afcc7b208674132e0e8bfafdb")
    wandb_run = wandb.init(project="UNetUNet", dir=os.getenv("WANDB_DIR", "cache/wandb"), config=wandb_config)
    wandb_run.log_code(root=".", name=tag+"UNetUNet_v1_py2_train.py")
    global_config["wandb_run"] = wandb_run
    global_config["IS_LOGGER_WANDB"] = True
    global_config["input_modality"] = ["TOFNAC", "CTAC"]
    global_config["model_step1_params"] = model_step1_params
    global_config["data_loader_params"] = data_loader_params


    test_fold = args.test_fold
    root_folder = f"./results/test_fold_{test_fold}/"
    data_div_json = "data_div.json"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)

    # load step 1 model and step 2 model
    model = VQModel(
        ddconfig=model_step1_params["ddconfig"],
        n_embed=model_step1_params["n_embed"],
        embed_dim=model_step1_params["embed_dim"],
        ckpt_path=model_step1_params["ckpt_path"],
        ignore_keys=[],
        image_key="image",
    )

    print("Model step 1 loaded from", model_step1_params["ckpt_path"])
    model.to(device)
    model.train()

    # set the logger
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file_path = f"train_log_{current_time}.json"
    logger = simple_logger(log_file_path, global_config)
    global_config["logger"] = logger
    
    train_data_loader, val_data_loader, test_data_loader = prepare_dataset(data_div_json, global_config)

if __name__ == "__main__":
    main()
