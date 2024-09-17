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

from UNetUNet_v1_py5_step2_util import simple_logger, prepare_dataset
from monai.networks.nets import DynUNet
from monai.losses import DeepSupervisionLoss

def is_batch_meaningful(batch_data):
    is_meaningful = True
    key = "STEP1"
    # batch size is 1
    cube_mean = torch.mean(batch_data[key], dim=(1, 2, 3, 4))
    if cube_mean[i] < meaningful_batch_th:
        is_meaningful = False
    return is_meaningful




def main():
    # here I will use argparse to parse the arguments
    argparser = argparse.ArgumentParser(description='Prepare dataset for training')
    argparser.add_argument('--cross_validation', type=int, default=5, help='Index of the cross validation')
    args = argparser.parse_args()
    tag = f"fold{args.cross_validation}_256"

    random_seed = 729
    # set the random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # MID_PET = 5000
    # MIQ_PET = 0.9
    # MAX_PET = 20000
    # MAX_CT = 1976
    # MIN_CT = -1024
    # MIN_PET = 0
    # RANGE_CT = MAX_CT - MIN_CT
    # RANGE_PET = MAX_PET - MIN_PET

    data_loader_params = {
        "norm": {
            "MID_PET": 5000,
            "MIQ_PET": 0.9,
            "MAX_PET": 20000,
            "MAX_CT": 1976,
            "MIN_CT": -1024,
            "MIN_PET": 0,
            "RANGE_CT": 3000,
            "RANGE_PET": 20000,
        },
        "train": {
            "batch_size": 1,
            "shuffle": True,
            "num_workers_cache": 4,
            "num_workers_loader": 8,
            "cache_rate": 1.0,
        },
        "val": {
            "batch_size": 1,
            "shuffle": False,
            "num_workers_cache": 2,
            "num_workers_loader": 4,
            "cache_rate": 0.1,
        },
        "test": {
            "batch_size": 1,
            "shuffle": False,
            "num_workers_cache": 1,
            "num_workers_loader": 2,
            "cache_rate": 0.1,
        },
    }

    model_step2_params = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "kernels": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
        "filters": (64, 128, 256),
        "dropout": 0.,
        "norm_name": ('INSTANCE', {'affine': True}),
        "act_name": ('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
        "deep_supervision": True,
        "deep_supr_num": 1,
        "res_block": True,
        "trans_bias": False,
        "ckpt_path": "d3f64_tsv1.pth",
        "cube_size": 256,
    }

    train_params = {
        "num_epoch": 5000, # 50 epochs
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "loss": "MAE",
        "val_per_epoch": 50,
        "save_per_epoch": 100,
        "meaningful_batch_th": -0.9,
    }

    wandb_config = {
        "cross_validation": args.cross_validation,
        "random_seed": random_seed,
        # "model_step1_params": model_step1_params,
        "model_step2_params": model_step2_params,
        "data_loader_params": data_loader_params,
        "train_params": train_params,
    }

    global_config = {}

    # initialize wandb
    wandb.login(key = "41c33ee621453a8afcc7b208674132e0e8bfafdb")
    wandb_run = wandb.init(project="UNetUNet", dir=os.getenv("WANDB_DIR", "cache/wandb"), config=wandb_config)
    wandb_run.log_code(root=".", name=tag+"UNetUNet_v1_py5_step2.py")
    global_config["tag"] = tag
    global_config["wandb_run"] = wandb_run
    global_config["IS_LOGGER_WANDB"] = True
    global_config["input_modality"] = ["STEP1", "STEP2"]
    # global_config["model_step1_params"] = model_step1_params
    global_config["model_step2_params"] = model_step2_params
    global_config["data_loader_params"] = data_loader_params
    global_config["train_params"] = train_params
    global_config["cross_validation"] = args.cross_validation
    global_config["cube_size"] = model_step2_params["cube_size"]


    cross_validation = args.cross_validation
    root_folder = f"./results/cv{cross_validation}_256/"
    data_div_json = "UNetUNet_v1_data_split.json"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)
    global_config["root_folder"] = root_folder

    model = DynUNet(
        spatial_dims=model_step2_params["spatial_dims"],
        in_channels=model_step2_params["in_channels"],
        out_channels=model_step2_params["out_channels"],
        kernels=model_step2_params["kernels"],
        strides=model_step2_params["strides"],
        filters=model_step2_params["filters"],
        dropout=model_step2_params["dropout"],
        norm_name=model_step2_params["norm_name"],
        act_name=model_step2_params["act_name"],
        deep_supervision=model_step2_params["deep_supervision"],
        deep_supr_num=model_step2_params["deep_supr_num"],
        res_block=model_step2_params["res_block"],
        trans_bias=model_step2_params["trans_bias"],
    )
    
    model.load_state_dict(torch.load(model_step2_params["ckpt_path"], map_location=torch.device('cpu')), strict=False)
    print("Model step 1 loaded from", model_step2_params["ckpt_path"])
    model.to(device)
    model.train()

    # set the logger
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file_path = f"train_log_{current_time}.json"
    logger = simple_logger(log_file_path, global_config)
    global_config["logger"] = logger
    
    train_data_loader, val_data_loader, test_data_loader = prepare_dataset(data_div_json, global_config)

    # set the optimizer
    if train_params["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_params["lr"], weight_decay=train_params["weight_decay"])
    else:
        raise ValueError(f"Optimizer {train_params['optimizer']} is not supported")
    
    # set the loss function
    if train_params["loss"] == "MAE":
        output_loss = torch.nn.L1Loss()
    elif train_params["loss"] == "MSE":
        output_loss = torch.nn.MSELoss()

    ds_loss = DeepSupervisionLoss(
        loss = output_loss,
        weight_mode = "exp",
        weights = None,
    )

    # train the model
    best_val_loss = 1e10
    print("Start training")
    for idx_epoch in range(train_params["num_epoch"]):

        # train the model
        model.train()
        train_loss = 0
        for idx_case, case_data in enumerate(train_data_loader):
            if not is_batch_meaningful(case_data):
                continue
            
            inputs = case_data["STEP1"].to(device)
            targets = case_data["STEP2"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = ds_loss(torch.unbind(outputs, 1), targets-inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_data_loader) * data_loader_params["norm"]["RANGE_CT"]
        logger.log(idx_epoch, "train_loss", train_loss)

        # evaluate the model
        if idx_epoch % train_params["val_per_epoch"] == 0:
                model.eval()
                val_loss = 0
                for idx_case, case_data in enumerate(val_data_loader):
                    if not is_batch_meaningful(case_data):
                        continue
                    
                    inputs = case_data["STEP1"].to(device)
                    targets = case_data["STEP2"].to(device)
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = output_loss(outputs, targets-inputs)
                        val_loss += loss.item()
                
                val_loss /= len(val_data_loader) * data_loader_params["norm"]["RANGE_CT"]
                logger.log(idx_epoch, "val_loss", val_loss)
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(root_folder, f"best_model_cv{cross_validation}_step2.pth")
                    torch.save(model.state_dict(), save_path)
                    logger.log(idx_epoch, "best_val_loss", best_val_loss)
                    print(f"Save the best model with val_loss: {val_loss} at epoch {idx_epoch}")
                    logger.log(idx_epoch, "best_model_epoch", idx_epoch)
                    logger.log(idx_epoch, "best_model_val_loss", val_loss)
                    wandb_run.log_model(path=save_path, name="model_best_eval", aliases=tag+f"cv{cross_validation}_step2")
                    
                    # test the model
                    test_loss = 0
                    for idx_case, case_data in enumerate(test_data_loader):
                        if not is_batch_meaningful(case_data):
                            continue
                        
                        inputs = case_data["STEP1"].to(device)
                        targets = case_data["STEP2"].to(device)
                        with torch.no_grad():
                            outputs = model(inputs)
                            loss = output_loss(outputs, targets-inputs)
                            test_loss += loss.item()
                    
                    test_loss /= len(test_data_loader) * data_loader_params["norm"]["RANGE_CT"]
                    logger.log(idx_epoch, "test_loss", test_loss)
        
        # save the model
        if idx_epoch % train_params["save_per_epoch"] == 0:
            save_path = os.path.join(root_folder, f"model_epoch_{idx_epoch}_step2.pth")
            torch.save(model.state_dict(), save_path)
            logger.log(idx_epoch, "save_model_path", save_path)
            print(f"Save model to {save_path}")
            wandb_run.log_model(path=save_path, name="model_checkpoint", aliases=tag+f"cv{cross_validation}_step2")

print("Done!")

if __name__ == "__main__":
    main()
