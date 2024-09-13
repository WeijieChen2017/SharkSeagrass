import os

# set the environment variable to use the GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is: ", device)

import argparse
import json
import time
import random
import nibabel as nib
import numpy as np

from UNetUNet_v1_py2_train_util import VQModel, simple_logger, prepare_dataset

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MAX_CT = 1976
MIN_CT = -1024
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET


def two_segment_scale(arr, MIN, MID, MAX, MIQ):
    # Create an empty array to hold the scaled results
    scaled_arr = np.zeros_like(arr, dtype=np.float32)

    # First segment: where arr <= MID
    mask1 = arr <= MID
    scaled_arr[mask1] = (arr[mask1] - MIN) / (MID - MIN) * MIQ

    # Second segment: where arr > MID
    mask2 = arr > MID
    scaled_arr[mask2] = MIQ + (arr[mask2] - MID) / (MAX - MID) * (1 - MIQ)
    
    return scaled_arr

def main():
    # here I will use argparse to parse the arguments
    argparser = argparse.ArgumentParser(description='Prepare dataset for training')
    argparser.add_argument('-c', '--cross_validation', type=int, default=0, help='Index of the cross validation')
    argparser.add_argument('-p', '--part2', type=bool, default=True, help='Whether to run the second part')
    args = argparser.parse_args()
    tag = f"fold{args.cross_validation}"

    random_seed = 42
    # set the random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cross_validation = args.cross_validation
    root_folder = f"B100/five_fold_cross_validation/cv{cross_validation}/"
    data_div_json = "UNetUNet_v1_data_split.json"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)

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
            "cache_rate": 0.5,
        },
        "val": {
            "batch_size": 1,
            "shuffle": False,
            "num_workers_cache": 2,
            "num_workers_loader": 4,
            "cache_rate": 1.0,
        },
        "test": {
            "batch_size": 1,
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
        "ckpt_path": f"B100/five_fold_cross_validation/best_model_cv{cross_validation}.pth",
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

    global_config = {}

    # initialize wandb
    global_config["tag"] = tag
    global_config["input_modality"] = ["TOFNAC", "CTAC"]
    global_config["model_step1_params"] = model_step1_params
    global_config["data_loader_params"] = data_loader_params
    # global_config["train_params"] = train_params
    global_config["cross_validation"] = args.cross_validation

    
    global_config["root_folder"] = root_folder

    # load step 1 model and step 2 model
    model = VQModel(
        ddconfig=model_step1_params["ddconfig"],
        n_embed=model_step1_params["n_embed"],
        embed_dim=model_step1_params["embed_dim"],
    )
    
    model.load_state_dict(torch.load(model_step1_params["ckpt_path"], map_location=torch.device('cpu')), strict=False)
    print("Model step 1 loaded from", model_step1_params["ckpt_path"])
    model.to(device)

    if not args.part2:
        data_folder = "B100/TOFNAC_CTAC_hash/"
        data_split = ["train", "val", "test"]
        # load json
        with open(data_div_json, "r") as f:
            data_div = json.load(f)
        data_div_cv = data_div[f"cv_{cross_validation}"]

        log_file = os.path.join(root_folder, "log.txt")
        with open(log_file, "w") as f:
            print(f"Training division: {data_div_cv['train']}")
            print(f"Validation division: {data_div_cv['val']}")
            print(f"Testing division: {data_div_cv['test']}")

        for split in data_split:
            data_split_folder = os.path.join(root_folder, split)
            if not os.path.exists(data_split_folder):
                os.makedirs(data_split_folder)
            
            split_list = data_div_cv[split]
            split_loss = []
            # now there will be a list of file names
            for casename in split_list:
                print(f"{split} -> Processing {casename}")
                # casenme "FGX078"
                # TOFNAC_path "FGX078_CTAC.nii.gz" have been normalized
                # CTAC_path "FGX078_TOFNAC.nii.gz" have been normalized
                TOFNAC_path = os.path.join(data_folder, f"{casename}_TOFNAC.nii.gz")
                CTAC_path = os.path.join(data_folder, f"{casename}_CTAC.nii.gz")
                # load the data
                TOFNAC_file = nib.load(TOFNAC_path)
                CTAC_file = nib.load(CTAC_path)

                TOFNAC_data = TOFNAC_file.get_fdata()
                CTAC_data = CTAC_file.get_fdata()
                print(f"{split} -> {casename} -> TOFNAC shape: {TOFNAC_data.shape}, CTAC shape: {CTAC_data.shape}")

                len_z = TOFNAC_data.shape[2]
                CTAC_pred = np.zeros_like(CTAC_data)
                for idx_z in range(len_z):
                    if idx_z == 0:
                        slice_1 = TOFNAC_data[:, :, idx_z].reshape(400, 400, 1)
                        slice_2 = TOFNAC_data[:, :, idx_z].reshape(400, 400, 1)
                        slice_3 = TOFNAC_data[:, :, idx_z+1].reshape(400, 400, 1)
                        data_x = np.concatenate([slice_1, slice_2, slice_3], axis=2)
                    elif idx_z == len_z - 1:
                        slice_1 = TOFNAC_data[:, :, idx_z-1].reshape(400, 400, 1)
                        slice_2 = TOFNAC_data[:, :, idx_z].reshape(400, 400, 1)
                        slice_3 = TOFNAC_data[:, :, idx_z].reshape(400, 400, 1)
                        data_x = np.concatenate([slice_1, slice_2, slice_3], axis=2)
                    else:
                        data_x = TOFNAC_data[:, :, idx_z-1:idx_z+2]
                    # data_x is 400x400x3, convert it to 1x3x400x400
                    data_x = np.transpose(data_x, (2, 0, 1))
                    data_x = np.expand_dims(data_x, axis=0)
                    data_x = torch.tensor(data_x, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        pred_y = model(data_x)
                        pred_y = pred_y.cpu().detach().numpy()
                        pred_y = np.squeeze(pred_y, axis=0)
                        CTAC_pred[:, :, idx_z] = pred_y
                
                # save the CTAC_pred
                
                CTAC_pred_path = os.path.join(data_split_folder, f"{casename}_CTAC_pred_cv{cross_validation}.nii.gz")
                CTAC_pred_nii = nib.Nifti1Image(CTAC_pred, CTAC_file.affine, CTAC_file.header)
                nib.save(CTAC_pred_nii, CTAC_pred_path)
                print(f"Save the CTAC_pred to {CTAC_pred_path}")

                # compute the loss
                CTAC_HU = CTAC_data * data_loader_params["norm"]["RANGE_CT"] + data_loader_params["norm"]["MIN_CT"]
                CTAC_pred = CTAC_pred * data_loader_params["norm"]["RANGE_CT"] + data_loader_params["norm"]["MIN_CT"]
                CTAC_mask = CTAC_HU > -500
                MAE = np.mean(np.abs(CTAC_HU[CTAC_mask] - CTAC_pred[CTAC_mask]))
                print(f"{split} -> {casename} -> MAE: {MAE:.4f}")
                split_loss.append(MAE)
                with open(log_file, "a") as f:
                    f.write(f"{split} -> {casename} -> MAE: {MAE:.4f}\n")
            
            split_loss = np.asarray(split_loss)
            split_loss = np.mean(split_loss)
            print(f"{split} -> Average MAE: {split_loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{split} -> Average MAE: {split_loss:.4f}\n")

        print("Done!")
    
    else:
        root_folder = f"B100/TOFNAC_CTACIVV_part2/cv{cross_validation}/"
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)
        data_folder = f"B100/TOFNAC_CTACIVV_part2/"
        data_split = ["test"]
        # load json
        with open(data_div_json, "r") as f:
            data_div = json.load(f)
        
        data_div_cv = data_div[f"part2"]

        log_file = os.path.join(root_folder, "log_part2.txt")
        with open(log_file, "w") as f:
            print(f"Part 2 division: {data_div_cv['test']}")
        
        for split in data_split:
            data_split_folder = os.path.join(root_folder, split)
            if not os.path.exists(data_split_folder):
                os.makedirs(data_split_folder)
            
            split_list = data_div_cv[split]
            split_loss = []
            # now there will be a list of file names
            for casename in split_list:
                print(f"{split} -> Processing {casename}")
                # casenme "FGX078"
                # TOFNAC_path "FGX078_CTAC.nii.gz" is not normalized
                # CTAC_path "FGX078_TOFNAC.nii.gz" is not normalized
                TOFNAC_path = os.path.join(data_folder, f"TOFNAC_{casename}_400.nii.gz")
                CTAC_path = os.path.join(data_folder, f"CTACIVV_{casename}_400.nii.gz")
                # load the data
                TOFNAC_file = nib.load(TOFNAC_path)
                CTAC_file = nib.load(CTAC_path)

                # normalize the data
                TOFNAC_data = TOFNAC_file.get_fdata()
                CTAC_data = CTAC_file.get_fdata()[33:433, 33:433, :]
                TOFNAC_data = two_segment_scale(TOFNAC_data, MIN_PET, MID_PET, MAX_PET, MIQ_PET)

                print(f"{split} -> {casename} -> TOFNAC shape: {TOFNAC_data.shape}, CTAC shape: {CTAC_data.shape}")

                len_z = TOFNAC_data.shape[2]
                CTAC_pred = np.zeros_like(CTAC_data)
                for idx_z in range(len_z):
                    if idx_z == 0:
                        slice_1 = TOFNAC_data[:, :, idx_z].reshape(400, 400, 1)
                        slice_2 = TOFNAC_data[:, :, idx_z].reshape(400, 400, 1)
                        slice_3 = TOFNAC_data[:, :, idx_z+1].reshape(400, 400, 1)
                    elif idx_z == len_z - 1:
                        slice_1 = TOFNAC_data[:, :, idx_z-1].reshape(400, 400, 1)
                        slice_2 = TOFNAC_data[:, :, idx_z].reshape(400, 400, 1)
                        slice_3 = TOFNAC_data[:, :, idx_z].reshape(400, 400, 1)
                    else:
                        slice_1 = TOFNAC_data[:, :, idx_z-1].reshape(400, 400, 1)
                        slice_2 = TOFNAC_data[:, :, idx_z].reshape(400, 400, 1) 
                        slice_3 = TOFNAC_data[:, :, idx_z+1].reshape(400, 400, 1)
                    data_x = np.concatenate([slice_1, slice_2, slice_3], axis=2)
                    # data_x is 400x400x3, convert it to 1x3x400x400
                    data_x = np.transpose(data_x, (2, 0, 1))
                    data_x = np.expand_dims(data_x, axis=0)
                    data_x = torch.tensor(data_x, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        pred_y = model(data_x)
                        pred_y = pred_y.cpu().detach().numpy()
                        pred_y = np.squeeze(pred_y, axis=0)
                        CTAC_pred[:, :, idx_z] = pred_y
                    
                # save the CTAC_pred
                CTAC_pred = CTAC_pred * RANGE_CT + MIN_CT
                CTAC_pred_path = os.path.join(data_split_folder, f"{casename}_CTAC_pred_cv{cross_validation}.nii.gz")
                CTAC_pred_nii = nib.Nifti1Image(CTAC_pred, CTAC_file.affine, CTAC_file.header)
                nib.save(CTAC_pred_nii, CTAC_pred_path)

                # compute the loss
                CTGT_mask = CTAC_data > -500
                MAE = np.mean(np.abs(CTAC_data[CTGT_mask] - CTAC_pred[CTGT_mask]))
                print(f"{split} -> {casename} -> MAE: {MAE:.4f}")
                split_loss.append(MAE)

                with open(log_file, "a") as f:
                    f.write(f"{split} -> {casename} -> MAE: {MAE:.4f}\n")
            
            split_loss = np.asarray(split_loss)
            split_loss = np.mean(split_loss)
            print(f"{split} -> Average MAE: {split_loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{split} -> Average MAE: {split_loss:.4f}\n")
            
        print("Done!")

if __name__ == "__main__":
    main()
