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

from transformers import T5ForConditionalGeneration, T5Config

import os

# # This is for both train axial, coronal, sagittal slices
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

# Create the necessary subdirectories and set the environment variables
# for key, path in cache_dirs.items():
#     os.makedirs(path, exist_ok=True)
#     os.environ[key] = path

# set the environment variable to use the GPU if available
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import wandb
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("The device is: ", device)

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

        # log to wandb if msg is number
        # if self.IS_LOGGER_WANDB and isinstance(msg, (int, float)):
        #     self.wandb_run.log({key: msg})

def train_or_eval_or_test(train_phase, model, path_list_x, path_list_y, optimizer, global_config):

    # z, d tensor
    batch_train = global_config["data_params"]["batch_train"]
    batch_val = global_config["data_params"]["batch_val"]
    batch_test = global_config["data_params"]["batch_test"]

    data_ind_axial_x = np.load(path_list_x["axial"])
    data_ind_coronal_x = np.load(path_list_x["coronal"])
    data_ind_sagittal_x = np.load(path_list_x["sagittal"])
    data_ind_axial_y = np.load(path_list_y["axial"])
    data_ind_coronal_y = np.load(path_list_y["coronal"])
    data_ind_sagittal_y = np.load(path_list_y["sagittal"])

    len_axial = data_ind_axial_x.shape[0]
    len_coronal = data_ind_coronal_x.shape[0]
    len_sagittal = data_ind_sagittal_x.shape[0]

    indices_list_axial = [i for i in range(1, len_axial-1)]
    indices_list_coronal = [i for i in range(1, len_coronal-1)]
    indices_list_sagittal = [i for i in range(1, len_sagittal-1)]

    # divide the indices into batches
    if train_phase == "train":
        batch_size = batch_train
    elif train_phase == "val":
        batch_size = batch_val
    elif train_phase == "test":
        batch_size = batch_test
    else:
        raise ValueError(f"train_phase {train_or_eval_or_test} is not supported")

    batch_indices_list_axial = [indices_list_axial[i:i+batch_train] for i in range(0, len(indices_list_axial), batch_size)]
    batch_indices_list_coronal = [indices_list_coronal[i:i+batch_train] for i in range(0, len(indices_list_coronal), batch_size)]
    batch_indices_list_sagittal = [indices_list_sagittal[i:i+batch_train] for i in range(0, len(indices_list_sagittal), batch_size)]

    random.shuffle(indices_list_axial)
    random.shuffle(indices_list_coronal)
    random.shuffle(indices_list_sagittal)

    axial_case_loss = []
    axial_cass_diff = []
    coronal_case_loss = []
    coronal_case_diff = []
    sagittal_case_loss = []
    sagittal_case_diff = []

    # axial slices
    for indices in batch_indices_list_axial:
        # according to the batch indices list, get the corresponding data
        batch_x = torch.tensor(data_ind_axial_x[indices].astype(int)).to(device)
        batch_y = torch.tensor(data_ind_axial_y[indices].astype(int)).to(device)
        # n_batch, len_seq
        diff_count = torch.sum(batch_x != batch_y, dim=1)
        diff_ratio = diff_count / batch_x.shape[1]
        diff_avg = torch.mean(diff_ratio)

        if train_phase == "train":
            optimizer.zero_grad()
            loss = model(input_ids=batch_x, labels=batch_y).loss
            loss.backward()
            optimizer.step()
        elif train_phase == "val" or train_phase == "test":
            with torch.no_grad():
                loss = model(input_ids=batch_x, labels=batch_y).loss

        axial_case_loss.append(loss.item())
        axial_case_diff.append(diff_avg.item())

    axial_case_loss /= len(batch_indices_list_axial)
    axial_case_diff /= len(batch_indices_list_axial)

    # coronal slices
    for indices in batch_indices_list_coronal:
        batch_x = torch.tensor(data_ind_coronal_x[indices].astype(int)).to(device)
        batch_y = torch.tensor(data_ind_coronal_y[indices].astype(int)).to(device)
        diff_count = torch.sum(batch_x != batch_y, dim=1)
        diff_ratio = diff_count / batch_x.shape[1]
        diff_avg = torch.mean(diff_ratio)

        if train_phase == "train":
            optimizer.zero_grad()
            loss = model(input_ids=batch_x, labels=batch_y).loss
            loss.backward()
            optimizer.step()
        elif train_phase == "val" or train_phase == "test":
            with torch.no_grad():
                loss = model(input_ids=batch_x, labels=batch_y).loss

        coronal_case_loss.append(loss.item())
        coronal_case_diff.append(diff_avg.item())
    
    coronal_case_loss /= len(batch_indices_list_coronal)
    coronal_case_diff /= len(batch_indices_list_coronal)

    # sagittal slices
    for indices in batch_indices_list_sagittal:
        batch_x = torch.tensor(data_ind_sagittal_x[indices].astype(int)).to(device)
        batch_y = torch.tensor(data_ind_sagittal_y[indices].astype(int)).to(device)
        diff_count = torch.sum(batch_x != batch_y, dim=1)
        diff_ratio = diff_count / batch_x.shape[1]
        diff_avg = torch.mean(diff_ratio)

        if train_phase == "train":
            optimizer.zero_grad()
            loss = model(input_ids=batch_x, labels=batch_y).loss
            loss.backward()
            optimizer.step()
        elif train_phase == "val" or train_phase == "test":
            with torch.no_grad():
                loss = model(input_ids=batch_x, labels=batch_y).loss

        sagittal_case_loss.append(loss.item())
        sagittal_case_diff.append(diff_avg.item())
    
    sagittal_case_loss /= len(batch_indices_list_sagittal)
    sagittal_case_diff /= len(batch_indices_list_sagittal)

    return_dict = {
        "axial_case_loss": axial_case_loss,
        "axial_case_diff": axial_case_diff,
        "coronal_case_loss": coronal_case_loss,
        "coronal_case_diff": coronal_case_diff,
        "sagittal_case_loss": sagittal_case_loss,
        "sagittal_case_diff": sagittal_case_diff,
    }

    return return_dict


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
    argparser.add_argument('--pretrain', type=str, default='Y', help='Whether to pretrain the model')
    argparser.add_argument('--model_architecture', type=str, default='T5_v1.1', help='The architecture of the model')
    argparser.add_argument('--model_scale', type=str, default='small', help='The scale of the model')
    argparser.add_argument('--batch_size', type=int, default=8, help='The batch size for training')
    argparser.add_argument('--SSL_available', type=str, default='Y', help='Whether the SSL is available')
    args = argparser.parse_args()
    cross_validation = args.cross_validation
    is_pretrained = True if args.pretrain == 'Y' or args.pretrain == 'y' else False
    model_architecture = args.model_architecture
    model_scale = args.model_scale
    batch_size = args.batch_size
    SSL_available = True if args.SSL_available == 'Y' or args.SSL_available == 'y' else False

    if not model_architecture in ["byte_T5", "T5_v1.1", "mT5"]:
        raise ValueError(f"Model architecture {model_architecture} is not supported")
    
    if not model_scale in ["small", "base", "large", "xl", "xxl"]:
        raise ValueError(f"Model scale {model_scale} is not supported")

    if is_pretrained:
        tag = f"fold{args.cross_validation}_pretrain_{model_architecture}_{model_scale}_"
    else:
        tag = f"fold{args.cross_validation}_{model_architecture}_{model_scale}_"

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
        "batch_train": batch_size, 
        "batch_val": batch_size, 
        "batch_test": batch_size, 
    }

    model_params = {
        "model_architecture": model_architecture,
        "model_scale": model_scale,
        "is_pretrained": is_pretrained,
    }

    train_params = {
        "num_epoch": 101,
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "loss": "NLL",
        "val_per_epoch": 10,
        "save_per_epoch": 25,
    }

    # wandb_config = {
    #     "cross_validation": args.cross_validation,
    #     "pretrain": args.pretrain,
    #     "model_scale": args.model_scale,
    #     "random_seed": random_seed,
    #     "model_params": model_step1_params,
    #     "data_params": data_loader_params,
    #     "train_params": train_params,
    # }

    global_config = {}

    # initialize wandb
    # wandb.login(key = "41c33ee621453a8afcc7b208674132e0e8bfafdb")
    # wandb_run = wandb.init(project="T5", dir=os.getenv("WANDB_DIR", "cache/wandb"), config=wandb_config)
    # wandb_run.log_code(root=".", name=tag+"T5_v1.py")
    global_config["tag"] = tag
    # global_config["wandb_run"] = wandb_run
    # global_config["IS_LOGGER_WANDB"] = True
    global_config["model_params"] = model_params
    global_config["data_params"] = data_params
    global_config["train_params"] = train_params
    global_config["cross_validation"] = args.cross_validation

    cross_validation = args.cross_validation
    root_folder = f"./results/{tag}_root/"
    data_div_json = "T5_v1_data_split.json"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)
    global_config["root_folder"] = root_folder
    pre_train_ckpt = f"t5-v1_1-{model_scale}.pth"

    if model_architecture == "byte_T5":
        model_ckpt = f"google/byt5-{model_scale}"
        pre_train_ckpt = f"byt5-{model_scale}.pth"
        if SSL_available and is_pretrained:
            model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        else:
            # Create a configuration object for the T5 model
            config = T5Config.from_pretrained(model_ckpt)
            model = T5ForConditionalGeneration(config)
            if is_pretrained:
                model.load_state_dict(torch.load(pre_train_ckpt))

    elif model_architecture == "T5_v1.1":
        model_ckpt = f"google/t5-v1_1-{model_scale}"
        pre_train_ckpt = f"t5-v1_1-{model_scale}.pth"
        if SSL_available and is_pretrained:
            model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        else:
            # Create a configuration object for the T5 model
            config = T5Config.from_pretrained(model_ckpt)
            model = T5ForConditionalGeneration(config)
            if is_pretrained:
                model.load_state_dict(torch.load(pre_train_ckpt))

    elif model_architecture == "mT5":
        model_ckpt = f"google/mt5-{model_scale}"
        pre_train_ckpt = f"mt5-{model_scale}.pth"
        if SSL_available and is_pretrained:
            model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        else:
            # Create a configuration object for the T5 model
            config = T5Config.from_pretrained(model_ckpt)
            model = T5ForConditionalGeneration(config)
            if is_pretrained:
                model.load_state_dict(torch.load(pre_train_ckpt))
    else:
        print("Current supported model architectures are: byte_T5, T5_v1.1, mT5")
        print("Current supported model scales are: small, base, large, xl, xxl")
        raise ValueError(f"Model architecture {model_architecture} or model scale {model_scale} is not supported")

    model.to(device)
    model.train()

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

    # train the model
    best_val_loss = 1e10
    print("Start training")
    for idx_epoch in range(train_params["num_epoch"]):

        # train the model
        model.train()
        axial_train_loss = 0
        coronal_train_loss = 0
        sagittal_train_loss = 0
        axial_norm_loss = 0
        coronal_norm_loss = 0
        sagittal_norm_loss = 0

        for idx_case, case_paths in enumerate(train_path_list):
            case_name = case_paths["case_name"]
            path_list_x = case_paths["TOFNAC"]
            path_list_y = case_paths["CTAC"]

            return_dict = train_or_eval_or_test("train", model, path_list_x, path_list_y, optimizer, global_config)

            # keep 3 decimal digits like 123.456
            axial_case_loss = round(return_dict["axial_case_loss"], 3)
            coronal_case_loss = round(return_dict["coronal_case_loss"], 3)
            sagittal_case_loss = round(return_dict["sagittal_case_loss"], 3)

            axial_norm_loss = round(return_dict["axial_case_loss"] / return_dict["axial_case_diff"], 3)
            coronal_norm_loss = round(return_dict["coronal_case_loss"] / return_dict["coronal_case_diff"], 3)
            sagittal_norm_loss = round(return_dict["sagittal_case_loss"] / return_dict["sagittal_case_diff"], 3)
            
            logger.log(idx_epoch, "train_axial_case_loss", axial_case_loss)
            logger.log(idx_epoch, "train_coronal_case_loss", coronal_case_loss)
            logger.log(idx_epoch, "train_sagittal_case_loss", sagittal_case_loss)
            logger.log(idx_epoch, "train_axial_norm_loss", axial_norm_loss)
            logger.log(idx_epoch, "train_coronal_norm_loss", coronal_norm_loss)
            logger.log(idx_epoch, "train_sagittal_norm_loss", sagittal_norm_loss)
            
            axial_train_loss += axial_case_loss
            coronal_train_loss += coronal_case_loss
            sagittal_train_loss += sagittal_case_loss

            axial_norm_loss += axial_norm_loss
            coronal_norm_loss += coronal_norm_loss
            sagittal_norm_loss += sagittal_norm_loss
        
        axial_train_loss /= len(train_data_loader)
        coronal_train_loss /= len(train_data_loader)
        sagittal_train_loss /= len(train_data_loader)

        axial_norm_loss /= len(train_data_loader)
        coronal_norm_loss /= len(train_data_loader)
        sagittal_norm_loss /= len(train_data_loader)

        train_loss = (axial_train_loss + coronal_train_loss + sagittal_train_loss) / 3
        train_norm_loss = (axial_norm_loss + coronal_norm_loss + sagittal_norm_loss) / 3

        logger.log(idx_epoch, "train_axial_loss", axial_train_loss)
        logger.log(idx_epoch, "train_coronal_loss", coronal_train_loss)
        logger.log(idx_epoch, "train_sagittal_loss", sagittal_train_loss)
        logger.log(idx_epoch, "train_loss", train_loss)

        logger.log(idx_epoch, "train_axial_norm_loss", axial_norm_loss)
        logger.log(idx_epoch, "train_coronal_norm_loss", coronal_norm_loss)
        logger.log(idx_epoch, "train_sagittal_norm_loss", sagittal_norm_loss)
        logger.log(idx_epoch, "train_norm_loss", train_norm_loss)

        # evaluate the model
        if idx_epoch % train_params["val_per_epoch"] == 0:

            model.eval()
            axial_val_loss = 0
            coronal_val_loss = 0
            sagittal_val_loss = 0
            axial_val_norm_loss = 0
            coronal_val_norm_loss = 0
            sagittal_val_norm_loss = 0

            for idx_case, case_paths in enumerate(val_path_list):
                case_name = case_paths["case_name"]
                path_list_x = case_paths["TOFNAC"]
                path_list_y = case_paths["CTAC"]

                return_dict = train_or_eval_or_test("val", model, path_list_x, path_list_y, optimizer, global_config)

                axial_case_loss = round(return_dict["axial_case_loss"], 3)
                coronal_case_loss = round(return_dict["coronal_case_loss"], 3)
                sagittal_case_loss = round(return_dict["sagittal_case_loss"], 3)

                axial_norm_loss = round(return_dict["axial_case_loss"] / return_dict["axial_case_diff"], 3)
                coronal_norm_loss = round(return_dict["coronal_case_loss"] / return_dict["coronal_case_diff"], 3)
                sagittal_norm_loss = round(return_dict["sagittal_case_loss"] / return_dict["sagittal_case_diff"], 3)

                logger.log(idx_epoch, "val_axial_case_loss", axial_case_loss)
                logger.log(idx_epoch, "val_coronal_case_loss", coronal_case_loss)
                logger.log(idx_epoch, "val_sagittal_case_loss", sagittal_case_loss)
                logger.log(idx_epoch, "val_axial_norm_loss", axial_norm_loss)
                logger.log(idx_epoch, "val_coronal_norm_loss", coronal_norm_loss)
                logger.log(idx_epoch, "val_sagittal_norm_loss", sagittal_norm_loss)

                axial_val_loss += axial_case_loss
                coronal_val_loss += coronal_case_loss
                sagittal_val_loss += sagittal_case_loss

                axial_val_norm_loss += axial_norm_loss
                coronal_val_norm_loss += coronal_norm_loss
                sagittal_val_norm_loss += sagittal_norm_loss

            axial_val_loss /= len(val_data_loader)
            coronal_val_loss /= len(val_data_loader)
            sagittal_val_loss /= len(val_data_loader)

            axial_val_norm_loss /= len(val_data_loader)
            coronal_val_norm_loss /= len(val_data_loader)
            sagittal_val_norm_loss /= len(val_data_loader)

            val_loss = (axial_val_loss + coronal_val_loss + sagittal_val_loss) / 3
            val_norm_loss = (axial_val_norm_loss + coronal_val_norm_loss + sagittal_val_norm_loss) / 3

            logger.log(idx_epoch, "val_axial_loss", axial_val_loss)
            logger.log(idx_epoch, "val_coronal_loss", coronal_val_loss)
            logger.log(idx_epoch, "val_sagittal_loss", sagittal_val_loss)
            logger.log(idx_epoch, "val_loss", val_loss)
            
            logger.log(idx_epoch, "val_axial_norm_loss", axial_val_norm_loss)
            logger.log(idx_epoch, "val_coronal_norm_loss", coronal_val_norm_loss)
            logger.log(idx_epoch, "val_sagittal_norm_loss", sagittal_val_norm_loss)
            logger.log(idx_epoch, "val_norm_loss", val_norm_loss)

            if val_norm_loss < best_val_loss:
                best_val_loss = val_norm_loss
                save_path = os.path.join(root_folder, f"best_model_cv{cross_validation}.pth")
                torch.save(model.state_dict(), save_path)
                logger.log(idx_epoch, "best_val_norm_loss", best_val_loss)
                print(f"Save the best model with val_norm_loss: {val_loss} at epoch {idx_epoch}")
                logger.log(idx_epoch, "best_model_epoch", idx_epoch)
                logger.log(idx_epoch, "best_model_val_norm_loss", val_loss)
                # wandb_run.log_model(path=save_path, name="model_best_eval", aliases=tag+f"cv{cross_validation}_zscore")
                
                # test the model
                axial_test_loss = 0
                coronal_test_loss = 0
                sagittal_test_loss = 0
                axial_test_norm_loss = 0
                coronal_test_norm_loss = 0
                sagittal_test_norm_loss = 0

                for idx_case, case_paths in enumerate(test_path_list):
                    case_name = case_paths["case_name"]
                    path_list_x = case_paths["TOFNAC"]
                    path_list_y = case_paths["CTAC"]

                    return_dict = train_or_eval_or_test("test", model, path_list_x, path_list_y, optimizer, global_config)

                    axial_case_loss = round(return_dict["axial_case_loss"], 3)
                    coronal_case_loss = round(return_dict["coronal_case_loss"], 3)
                    sagittal_case_loss = round(return_dict["sagittal_case_loss"], 3)

                    axial_norm_loss = round(return_dict["axial_case_loss"] / return_dict["axial_case_diff"], 3)
                    coronal_norm_loss = round(return_dict["coronal_case_loss"] / return_dict["coronal_case_diff"], 3)
                    sagittal_norm_loss = round(return_dict["sagittal_case_loss"] / return_dict["sagittal_case_diff"], 3)

                    logger.log(idx_epoch, "test_axial_case_loss", axial_case_loss)
                    logger.log(idx_epoch, "test_coronal_case_loss", coronal_case_loss)
                    logger.log(idx_epoch, "test_sagittal_case_loss", sagittal_case_loss)
                    logger.log(idx_epoch, "test_axial_norm_loss", axial_norm_loss)
                    logger.log(idx_epoch, "test_coronal_norm_loss", coronal_norm_loss)
                    logger.log(idx_epoch, "test_sagittal_norm_loss", sagittal_norm_loss)

                    axial_test_loss += axial_case_loss
                    coronal_test_loss += coronal_case_loss
                    sagittal_test_loss += sagittal_case_loss

                    axial_test_norm_loss += axial_norm_loss
                    coronal_test_norm_loss += coronal_norm_loss
                    sagittal_test_norm_loss += sagittal_norm_loss

                axial_test_loss /= len(test_data_loader)
                coronal_test_loss /= len(test_data_loader)
                sagittal_test_loss /= len(test_data_loader)

                axial_test_norm_loss /= len(test_data_loader)
                coronal_test_norm_loss /= len(test_data_loader)
                sagittal_test_norm_loss /= len(test_data_loader)

                test_loss = (axial_test_loss + coronal_test_loss + sagittal_test_loss) / 3
                test_norm_loss = (axial_test_norm_loss + coronal_test_norm_loss + sagittal_test_norm_loss) / 3

                logger.log(idx_epoch, "test_axial_loss", axial_test_loss)
                logger.log(idx_epoch, "test_coronal_loss", coronal_test_loss)
                logger.log(idx_epoch, "test_sagittal_loss", sagittal_test_loss)
                logger.log(idx_epoch, "test_loss", test_loss)
                
                logger.log(idx_epoch, "test_axial_norm_loss", axial_test_norm_loss)
                logger.log(idx_epoch, "test_coronal_norm_loss", coronal_test_norm_loss)
                logger.log(idx_epoch, "test_sagittal_norm_loss", sagittal_test_norm_loss)
                logger.log(idx_epoch, "test_norm_loss", test_norm_loss)

        # save the model
        if idx_epoch % train_params["save_per_epoch"] == 0:
            save_path = os.path.join(root_folder, f"model_epoch_{idx_epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logger.log(idx_epoch, "save_model_path", save_path)
            print(f"Save model to {save_path}")
            # wandb_run.log_model(path=save_path, name="model_checkpoint", aliases=tag+f"cv{cross_validation}_zscore")

print("Done!")

if __name__ == "__main__":
    main()
