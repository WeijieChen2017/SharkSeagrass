# T5_v1.1:
# google/t5-v1_1-small ---> batch_size=8, 17.1GB GPU memory ---> batch_size=16, 36.3GB GPU memory
# google/t5-v1_1-base ---> batch_size=4, 22.4GB GPU memory ---> batch_size=6, 31.5GB GPU memory
# google/t5-v1_1-large ---> batch_size=1, 21.1GB GPU memory ---> batch_size=1, 34.0 GB GPU memory
# google/t5-v1_1-xl
# google/t5-v1_1-xxl

# T5:
# google/t5-small ---> batch_size=4, 7.1GB GPU memory
# google/t5-base ---> batch_size=4, 18.0GB GPU memory
# google/t5-large
# google/t5-3b
# google/t5-11b

# byte-T5:
# google/byt5-small 
# google/byt5-base 
# google/byt5-large
# google/byt5-xl
# google/byt5-xxl

# mT5(multilingual): 
# google/mt5-small ---> batch_size=4, 23.4 GB GPU memory ---> batch_size=6, 34.0 GB GPU memory
# google/mt5-base ---> batch_size=1, 15.0 GB GPU memory ---> batch_size=4, 38.9 GB GPU memory
# google/mt5-large
# google/mt5-xl
# google/mt5-xxl

import os

# # This is for both train axial, coronal, sagittal slices
# # Define the base cache directory
base_cache_dir = './cache'

# # Define and create necessary subdirectories within the base cache directory
cache_dirs = {
    # 'WANDB_DIR': os.path.join(base_cache_dir, 'wandb'),
    # 'WANDB_CACHE_DIR': os.path.join(base_cache_dir, 'wandb_cache'),
    # 'WANDB_CONFIG_DIR': os.path.join(base_cache_dir, 'config'),
    # 'WANDB_DATA_DIR': os.path.join(base_cache_dir, 'data'),
    'TRANSFORMERS_CACHE': os.path.join(base_cache_dir, 'transformers'),
    "HF_HOME": os.path.join(base_cache_dir, 'huggingface'),
    # 'MPLCONFIGDIR': os.path.join(base_cache_dir, 'mplconfig')
}

# # Create the base cache directory if it doesn't exist
os.makedirs(base_cache_dir, exist_ok=True)

# Create the necessary subdirectories and set the environment variables
for key, path in cache_dirs.items():
    os.makedirs(path, exist_ok=True)
    os.environ[key] = path

# set the environment variable to use the GPU if available
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import wandb
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device is: ", device)

from T5_v1_eval_img_utils import VQModel_decoder

import argparse
import json
import time
import random
import numpy as np

# from UNetUNet_v1_py2_train_acs_util import VQModel, simple_logger, prepare_dataset

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
        log_str = f"{current_time} Global epoch: {global_epoch}, {key}, {msg}\n"
        with open(self.log_file_path, "a") as f:
            f.write(log_str)
        print(log_str)

def prepare_dataset(data_div_json, global_config):
    
    with open(data_div_json, "r") as f:
        data_div = json.load(f)
    
    cv = global_config["cross_validation"]

    train_list = data_div[f"cv_{cv}"]["train"]
    val_list = data_div[f"cv_{cv}"]["val"]
    test_list = data_div[f"cv_{cv}"]["test"]

    str_train_list = ", ".join(train_list)
    str_val_list = ", ".join(val_list)
    str_test_list = ", ".join(test_list)

    global_config["logger"].log(0, "data_split_train", str_train_list)
    global_config["logger"].log(0, "data_split_val", str_val_list)
    global_config["logger"].log(0, "data_split_test", str_test_list)

    # construct the data path list
    train_path_list = []
    val_path_list = []
    test_path_list = []

    for hashname in train_list:
        train_path_list.append({
            "case_name": hashname,
            "TOFNAC": {
                "axial": f"TC256_v2_vq_f8/ind_axial/{hashname}_x_axial_ind.npy",
                "coronal": f"TC256_v2_vq_f8/ind_coronal/{hashname}_x_coronal_ind.npy",
                "sagittal": f"TC256_v2_vq_f8/ind_sagittal/{hashname}_x_sagittal_ind.npy",
            },
            "CTAC": {
                "axial": f"TC256_v2_vq_f8/ind_axial/{hashname}_y_axial_ind.npy",
                "coronal": f"TC256_v2_vq_f8/ind_coronal/{hashname}_y_coronal_ind.npy",
                "sagittal": f"TC256_v2_vq_f8/ind_sagittal/{hashname}_y_sagittal_ind.npy",
            },
        })

    for hashname in val_list:
        val_path_list.append({
            "case_name": hashname,
            "TOFNAC":{
                "axial": f"TC256_v2_vq_f8/ind_axial/{hashname}_x_axial_ind.npy",
                "coronal": f"TC256_v2_vq_f8/ind_coronal/{hashname}_x_coronal_ind.npy",
                "sagittal": f"TC256_v2_vq_f8/ind_sagittal/{hashname}_x_sagittal_ind.npy",
            },
            "CTAC": {
                "axial": f"TC256_v2_vq_f8/ind_axial/{hashname}_y_axial_ind.npy",
                "coronal": f"TC256_v2_vq_f8/ind_coronal/{hashname}_y_coronal_ind.npy",
                "sagittal": f"TC256_v2_vq_f8/ind_sagittal/{hashname}_y_sagittal_ind.npy",
            },
        })

    for hashname in test_list:
        test_path_list.append({
            "case_name": hashname,
            "TOFNAC":{
                "axial": f"TC256_v2_vq_f8/ind_axial/{hashname}_x_axial_ind.npy",
                "coronal": f"TC256_v2_vq_f8/ind_coronal/{hashname}_x_coronal_ind.npy",
                "sagittal": f"TC256_v2_vq_f8/ind_sagittal/{hashname}_x_sagittal_ind.npy",
            },
            "CTAC": {
                "axial": f"TC256_v2_vq_f8/ind_axial/{hashname}_y_axial_ind.npy",
                "coronal": f"TC256_v2_vq_f8/ind_coronal/{hashname}_y_coronal_ind.npy",
                "sagittal": f"TC256_v2_vq_f8/ind_sagittal/{hashname}_y_sagittal_ind.npy",
            },
        })

    # save the data division file
    root_folder = global_config["root_folder"]
    data_division_file = os.path.join(root_folder, "data_division.json")
    data_division_dict = {
        "train": train_path_list,
        "val": val_path_list,
        "test": test_path_list,
    }
    for key in data_division_dict.keys():
        print(key)
        for key2 in data_division_dict[key]:
            print(key2)

    with open(data_division_file, "w") as f:
        json.dump(data_division_dict, f, indent=4)

    return train_path_list, val_path_list, test_path_list

def main():
    # here I will use argparse to parse the arguments
    argparser = argparse.ArgumentParser(description='Prepare dataset for training')
    argparser.add_argument('--cross_validation', type=int, default=0, help='Index of the cross validation')
    argparser.add_argument('--model_architecture', type=str, default='mT5', help='The architecture of the model')
    argparser.add_argument('--model_scale', type=str, default='small', help='The scale of the model')
    args = argparser.parse_args()
    cross_validation = args.cross_validation
    model_architecture = args.model_architecture
    model_scale = args.model_scale

    if not model_architecture in ["byte_T5", "T5_v1.1", "mT5"]:
        raise ValueError(f"Model architecture {model_architecture} is not supported")
    
    if not model_scale in ["small", "base", "large", "xl", "xxl"]:
        raise ValueError(f"Model scale {model_scale} is not supported")

    tag = f"fold{args.cross_validation}_pretrain_{model_architecture}_{model_scale}_"

    random_seed = 729
    # set the random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_params = {
        "zoom": 8,
    }

    model_params = {
        "vq_weights_path": "f8_vq_weights.npy",
        "model_architecture": model_architecture,
        "model_scale": model_scale,
        "VQ_NAME": "f8",
        "n_embed": 16384,
        "embed_dim": 4,
        "img_size" : 256,
        "input_modality" : ["TOFNAC", "CTAC"],
        # "ckpt_path": f"B100/TC256_best_ckpt/best_model_cv{cross_validation}.pth",
        "ckpt_path": f"vq_f8_nn.pth",
        "ddconfig": {
            "double_z": False,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [32],
            "dropout": 0.0,
        },
    }

    train_params = {
        "num_epoch": 51,
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "loss": "NLL",
        "val_per_epoch": 5,
        "save_per_epoch": 10,
    }

    global_config = {}
    global_config["tag"] = tag
    global_config["model_params"] = model_params
    global_config["data_params"] = data_params
    global_config["train_params"] = train_params
    global_config["cross_validation"] = args.cross_validation
    global_config["save_pred"] = True

    cross_validation = args.cross_validation
    root_folder = f"./results/{tag}root/"
    data_div_json = "T5_v1_data_split.json"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)
    global_config["root_folder"] = root_folder
    # pre_train_ckpt = os.path.join(root_folder, f"best_model_cv{cross_validation}.pth")

    # load step 1 model and step 2 model
    model = VQModel_decoder(
        ddconfig=model_params["ddconfig"],
        n_embed=model_params["n_embed"],
        embed_dim=model_params["embed_dim"],
    )
    model.load_pretrain_weights(model_params["ckpt_path"])
    model.to(device)

    # set the logger
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file_path = f"train_log_{current_time}.json"
    logger = simple_logger(log_file_path, global_config)
    global_config["logger"] = logger
    
    # set the optimizer
    if train_params["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_params["lr"], weight_decay=train_params["weight_decay"])
    else:
        raise ValueError(f"Optimizer {train_params['optimizer']} is not supported")
    
    # load the dataloader
    train_path_list, val_path_list, test_path_list = prepare_dataset(data_div_json, global_config)

    # load vq_weights
    vq_weights = np.load(model_params["vq_weights_path"])
    print(f"Loading vq weights from {model_params['vq_weights_path']}, shape: {vq_weights.shape}")
    ############################################################################################################

    for idx_case, case_paths in enumerate(test_path_list):
        case_name = case_paths["case_name"]
        path_list_x = case_paths["TOFNAC"]
        path_list_y = case_paths["CTAC"]

        if case_name == "ZTW155":
            continue
            
        temp_img_size = (256, 256, 720)
        direction = "axial"
        len_axial = temp_img_size[2]
        for i in range(len_axial):
            index = len_axial - i - 1
            pred_ind_path = f"{root_folder}/{case_name}_{direction}_pred_ind{index:03d}.npy"
            pred_ind = np.load(pred_ind_path)            
            pred_ind = pred_ind.reshape((32, 32))
            # load each 
            pred_post_quan = vq_weights[pred_ind.astype(int)].reshape(32, 32, 4)
            pred_img = VQModel_decoder(pred_post_quan)
            print(pred_img.shape)
            exit()



print("Done!")

if __name__ == "__main__":
    main()
