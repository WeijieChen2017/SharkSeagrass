import os

# This is for both train axial, coronal, sagittal slices
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

from UNetUNet_v1_py2_train_acs_util import VQModel, simple_logger, prepare_dataset


def train_or_eval(train_or_eval, model, volume_x, volume_y, optimizer, output_loss, LOSS_FACTOR):
    
    # 1, z, 256, 256 tensor
    axial_case_loss = 0
    coronal_case_loss = 0
    sagittal_case_loss = 0

    # print shape
    # print(volume_x.shape, volume_y.shape)
    # if z is not divided by 4, pad the volume_x and volume_y
    if volume_x.shape[1] % 4 != 0:
        pad_size = 4 - volume_x.shape[1] % 4
        # The torch.nn.functional.pad function uses a specific order for padding dimensions, which is: (left, right, top, bottom, front, back) for 4D tensors. This means that when padding in 4D, the padding should apply as follows:
        # The first two values (left, right) correspond to the last dimension.
        # The next two values (top, bottom) correspond to the second-to-last dimension.
        # The final two values (front, back) correspond to the third-to-last dimension.
        volume_x = torch.nn.functional.pad(volume_x, (0, 0, 0, 0, 0, pad_size, 0, 0), mode='constant', value=0)
        volume_y = torch.nn.functional.pad(volume_y, (0, 0, 0, 0, 0, pad_size, 0, 0), mode='constant', value=0)
    # print("After padding: ", volume_x.shape, volume_y.shape)

    indices_list_axial = [i for i in range(1, volume_x.shape[1]-1)]
    indices_list_coronal = [i for i in range(1, volume_x.shape[2]-1)]
    indices_list_sagittal = [i for i in range(1, volume_x.shape[3]-1)]
    random.shuffle(indices_list_axial)
    random.shuffle(indices_list_coronal)
    random.shuffle(indices_list_sagittal)

    if train_or_eval == "train":
        # axial slices
        for indices in indices_list_axial:
            x = volume_x[:, indices-1:indices+2, :, :]
            y = volume_y[:, indices, :, :].unsqueeze(0) # 1, 1, 256, 256
            # print(x.shape, y.shape)
        
            optimizer.zero_grad()
            outputs = model(x)
            loss = output_loss(outputs, y)
            loss.backward()
            optimizer.step()

            axial_case_loss += loss.item()
            # print(f"EPOCH {idx_epoch}, CASE {idx_case}, SLICE {indices}, LOSS {loss.item()*LOSS_FACTOR:.3f}")

        axial_case_loss /= len(indices_list_axial)
        axial_case_loss *= LOSS_FACTOR

        # coronal slices
        for indices in indices_list_coronal:
            x = volume_x[:, :, indices-1:indices+2, :] # 1, 720, 3, 256
            # convert it to 1, 3, 720, 256
            x = x.permute(0, 2, 1, 3)
            y = volume_y[:, :, indices, :].unsqueeze(0) # 1, 1, 720, 256
            # y = y.permute(0, 2, 1, 3)

            optimizer.zero_grad()
            outputs = model(x)
            loss = output_loss(outputs, y)
            loss.backward()
            optimizer.step()

            coronal_case_loss += loss.item()
            # print(f"EPOCH {idx_epoch}, CASE {idx_case}, SLICE {indices}, LOSS {loss.item()*LOSS_FACTOR:.3f}")
        
        coronal_case_loss /= len(indices_list_coronal)
        coronal_case_loss *= LOSS_FACTOR

        # sagittal slices
        for indices in indices_list_sagittal:
            x = volume_x[:, :, :, indices-1:indices+2] # 1, 720, 256, 3
            x = x.permute(0, 3, 1, 2) # 1, 3, 720, 256
            y = volume_y[:, :, :, indices].unsqueeze(0)
            # y = y.permute(0, 3, 1, 2)

            optimizer.zero_grad()
            outputs = model(x)
            loss = output_loss(outputs, y)
            loss.backward()
            optimizer.step()

            sagittal_case_loss += loss.item()
            # print(f"EPOCH {idx_epoch}, CASE {idx_case}, SLICE {indices}, LOSS {loss.item()*LOSS_FACTOR:.3f}")
        
        sagittal_case_loss /= len(indices_list_sagittal)
        sagittal_case_loss *= LOSS_FACTOR

    elif train_or_eval == "val":

        # axial slices
        for indices in indices_list_axial:
            x = volume_x[:, indices-1:indices+2, :, :]
            y = volume_y[:, indices, :, :].unsqueeze(0)

            with torch.no_grad():
                outputs = model(x)
                loss = output_loss(outputs, y)
                axial_case_loss += loss.item()
                # print(f"EPOCH {idx_epoch}, CASE {idx_case}, SLICE {indices}, LOSS {loss.item()*LOSS_FACTOR:.3f}")
        
        axial_case_loss /= len(indices_list_axial)
        axial_case_loss *= LOSS_FACTOR

        # coronal slices
        for indices in indices_list_coronal:
            x = volume_x[:, :, indices-1:indices+2, :]
            x = x.permute(0, 2, 1, 3)
            y = volume_y[:, :, indices, :].unsqueeze(0)

            with torch.no_grad():
                outputs = model(x)
                loss = output_loss(outputs, y)
                coronal_case_loss += loss.item()
                # print(f"EPOCH {idx_epoch}, CASE {idx_case}, SLICE {indices}, LOSS {loss.item()*LOSS_FACTOR:.3f}")
        
        coronal_case_loss /= len(indices_list_coronal)
        coronal_case_loss *= LOSS_FACTOR

        # sagittal slices
        for indices in indices_list_sagittal:
            x = volume_x[:, :, :, indices-1:indices+2]
            x = x.permute(0, 3, 1, 2)
            y = volume_y[:, :, :, indices].unsqueeze(0)

            with torch.no_grad():
                outputs = model(x)
                loss = output_loss(outputs, y)
                sagittal_case_loss += loss.item()
                # print(f"EPOCH {idx_epoch}, CASE {idx_case}, SLICE {indices}, LOSS {loss.item()*LOSS_FACTOR:.3f}")
        
        sagittal_case_loss /= len(indices_list_sagittal)
        sagittal_case_loss *= LOSS_FACTOR

    return axial_case_loss, coronal_case_loss, sagittal_case_loss

def main():
    # here I will use argparse to parse the arguments
    argparser = argparse.ArgumentParser(description='Prepare dataset for training')
    argparser.add_argument('--cross_validation', type=int, default=5, help='Index of the cross validation')
    args = argparser.parse_args()
    tag = f"fold{args.cross_validation}_256_zscore_scratch"

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

    model_step1_params = {
        "VQ_NAME": "f4-noattn",
        "n_embed": 8192,
        "embed_dim": 3,
        "img_size" : 256,
        "input_modality" : ["TOFNAC", "CTAC"],
        "ckpt_path": "vq_f4_noattn_nn.pth",
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

    train_params = {
        "num_epoch": 101, # 100 epochs
        "optimizer": "AdamW",
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "loss": "MAE",
        "val_per_epoch": 5,
        "save_per_epoch": 10,
    }

    wandb_config = {
        "cross_validation": args.cross_validation,
        "random_seed": random_seed,
        "model_step1_params": model_step1_params,
        "data_loader_params": data_loader_params,
        "train_params": train_params,
    }

    


    global_config = {}

    # initialize wandb
    wandb.login(key = "41c33ee621453a8afcc7b208674132e0e8bfafdb")
    wandb_run = wandb.init(project="UNetUNet", dir=os.getenv("WANDB_DIR", "cache/wandb"), config=wandb_config)
    wandb_run.log_code(root=".", name=tag+"UNetUNet_v1_py2_train.py")
    global_config["tag"] = tag
    global_config["wandb_run"] = wandb_run
    global_config["IS_LOGGER_WANDB"] = True
    global_config["input_modality"] = ["TOFNAC", "CTAC"]
    global_config["model_step1_params"] = model_step1_params
    global_config["data_loader_params"] = data_loader_params
    global_config["train_params"] = train_params
    global_config["cross_validation"] = args.cross_validation


    cross_validation = args.cross_validation
    root_folder = f"./results/cv{cross_validation}_256/"
    data_div_json = "UNetUNet_v1_data_split_acs.json"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)
    global_config["root_folder"] = root_folder

    # load step 1 model and step 2 model
    model = VQModel(
        ddconfig=model_step1_params["ddconfig"],
        n_embed=model_step1_params["n_embed"],
        embed_dim=model_step1_params["embed_dim"],
        # ckpt_path=model_step1_params["ckpt_path"],
        # ignore_keys=[],
        # image_key="image",
    )
    
    # model.load_state_dict(torch.load(model_step1_params["ckpt_path"], map_location=torch.device('cpu')), strict=False)
    # print("Model step 1 loaded from", model_step1_params["ckpt_path"])
    print("This is the model from scratch for comparison").
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

    # train the model
    best_val_loss = 1e10
    LOSS_FACTOR = data_loader_params["norm"]["RANGE_CT"]
    print("Start training")
    for idx_epoch in range(train_params["num_epoch"]):

        # train the model
        model.train()
        axial_train_loss = 0
        coronal_train_loss = 0
        sagittal_train_loss = 0

        for idx_case, case_data in enumerate(train_data_loader):
            # this will return a zx400x400 tensor
            volume_x = case_data["TOFNAC"].to(device)
            volume_y = case_data["CTAC"].to(device)

            axial_case_loss, coronal_case_loss, sagittal_case_loss = train_or_eval("train", model, volume_x, volume_y, optimizer, output_loss, LOSS_FACTOR)

            # keep 3 decimal digits like 123.456
            axial_case_loss = round(axial_case_loss, 3)
            coronal_case_loss = round(coronal_case_loss, 3)
            sagittal_case_loss = round(sagittal_case_loss, 3)
            
            logger.log(idx_epoch, "train_axial_case_loss", axial_case_loss)
            logger.log(idx_epoch, "train_coronal_case_loss", coronal_case_loss)
            logger.log(idx_epoch, "train_sagittal_case_loss", sagittal_case_loss)
            # print(f"EPOCH {idx_epoch}, CASE {idx_case}, LOSS {case_loss}")
            
            axial_train_loss += axial_case_loss
            coronal_train_loss += coronal_case_loss
            sagittal_train_loss += sagittal_case_loss
        
        axial_train_loss /= len(train_data_loader)
        coronal_train_loss /= len(train_data_loader)
        sagittal_train_loss /= len(train_data_loader)
        train_loss = (axial_train_loss + coronal_train_loss + sagittal_train_loss) / 3

        logger.log(idx_epoch, "train_axial_loss", axial_train_loss)
        logger.log(idx_epoch, "train_coronal_loss", coronal_train_loss)
        logger.log(idx_epoch, "train_sagittal_loss", sagittal_train_loss)
        logger.log(idx_epoch, "train_loss", train_loss)
    
        # evaluate the model
        if idx_epoch % train_params["val_per_epoch"] == 0:

            model.eval()
            axial_val_loss = 0
            coronal_val_loss = 0
            sagittal_val_loss = 0
            for idx_case, case_data in enumerate(val_data_loader):
                volume_x = case_data["TOFNAC"].to(device)
                volume_y = case_data["CTAC"].to(device)

                axial_case_loss, coronal_case_loss, sagittal_case_loss = train_or_eval("val", model, volume_x, volume_y, optimizer, output_loss, LOSS_FACTOR)

                axial_val_loss += axial_case_loss
                coronal_val_loss += coronal_case_loss
                sagittal_val_loss += sagittal_case_loss

                axial_case_loss = round(axial_case_loss, 3)
                coronal_case_loss = round(coronal_case_loss, 3)
                sagittal_case_loss = round(sagittal_case_loss, 3)
                
                logger.log(idx_epoch, "val_axial_case_loss", axial_case_loss)
                logger.log(idx_epoch, "val_coronal_case_loss", coronal_case_loss)
                logger.log(idx_epoch, "val_sagittal_case_loss", sagittal_case_loss)

            axial_val_loss /= len(val_data_loader)
            coronal_val_loss /= len(val_data_loader)
            sagittal_val_loss /= len(val_data_loader)
            val_loss = (axial_val_loss + coronal_val_loss + sagittal_val_loss) / 3
            
            logger.log(idx_epoch, "val_axial_loss", axial_val_loss)
            logger.log(idx_epoch, "val_coronal_loss", coronal_val_loss)
            logger.log(idx_epoch, "val_sagittal_loss", sagittal_val_loss)
            logger.log(idx_epoch, "val_loss", val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(root_folder, f"best_model_cv{cross_validation}.pth")
                torch.save(model.state_dict(), save_path)
                logger.log(idx_epoch, "best_val_loss", best_val_loss)
                print(f"Save the best model with val_loss: {val_loss} at epoch {idx_epoch}")
                logger.log(idx_epoch, "best_model_epoch", idx_epoch)
                logger.log(idx_epoch, "best_model_val_loss", val_loss)
                wandb_run.log_model(path=save_path, name="model_best_eval", aliases=tag+f"cv{cross_validation}_zscore")
                
                # test the model
                axial_test_loss = 0
                coronal_test_loss = 0
                sagittal_test_loss = 0
                for idx_case, case_data in enumerate(test_data_loader):
                    volume_x = case_data["TOFNAC"].to(device)
                    volume_y = case_data["CTAC"].to(device)
                    
                    axial_case_loss, coronal_case_loss, sagittal_case_loss = train_or_eval("val", model, volume_x, volume_y, optimizer, output_loss, LOSS_FACTOR)

                    axial_test_loss += axial_case_loss
                    coronal_test_loss += coronal_case_loss
                    sagittal_test_loss += sagittal_case_loss

                    axial_case_loss = round(axial_case_loss, 3)
                    coronal_case_loss = round(coronal_case_loss, 3)
                    sagittal_case_loss = round(sagittal_case_loss, 3)
                    
                    logger.log(idx_epoch, "test_axial_case_loss", axial_case_loss)
                    logger.log(idx_epoch, "test_coronal_case_loss", coronal_case_loss)
                    logger.log(idx_epoch, "test_sagittal_case_loss", sagittal_case_loss)

                axial_test_loss /= len(test_data_loader)
                coronal_test_loss /= len(test_data_loader)
                sagittal_test_loss /= len(test_data_loader)
                test_loss = (axial_test_loss + coronal_test_loss + sagittal_test_loss) / 3

                logger.log(idx_epoch, "test_axial_loss", axial_test_loss)
                logger.log(idx_epoch, "test_coronal_loss", coronal_test_loss)
                logger.log(idx_epoch, "test_sagittal_loss", sagittal_test_loss)
                logger.log(idx_epoch, "test_loss", test_loss)
        
        # save the model
        if idx_epoch % train_params["save_per_epoch"] == 0:
            save_path = os.path.join(root_folder, f"model_epoch_{idx_epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logger.log(idx_epoch, "save_model_path", save_path)
            print(f"Save model to {save_path}")
            wandb_run.log_model(path=save_path, name="model_checkpoint", aliases=tag+f"cv{cross_validation}_zscore")

print("Done!")

if __name__ == "__main__":
    main()
