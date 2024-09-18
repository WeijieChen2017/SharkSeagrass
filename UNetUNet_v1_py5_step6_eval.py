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
import numpy as np

from UNetUNet_v1_py5_step2_util import simple_logger, prepare_dataset
from monai.networks.nets import DynUNet
from monai.losses import DeepSupervisionLoss
from monai.inferers import sliding_window_inference

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
        "ckpt_path": f"B100/UNetUNet_best_ckpt/best_model_cv{args.cross_validation}_step2.pth",
        "cube_size": 128,
        "input_modality": ["STEP1", "STEP2"],
    }

    train_params = {
        "num_epoch": 5000, # 50 epochs
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "loss": "MAE",
        "val_per_epoch": 50,
        "save_per_epoch": 100,
        "meaningful_batch_th": 0.05,
        "meaningful_batch_per_epoch": 32,
    }


    global_config = {}

    global_config["tag"] = tag
    # global_config["input_modality"] = ["STEP1", "STEP2"]
    # global_config["model_step1_params"] = model_step1_params
    global_config["model_step2_params"] = model_step2_params
    global_config["data_loader_params"] = data_loader_params
    global_config["train_params"] = train_params
    global_config["cross_validation"] = args.cross_validation
    global_config["cube_size"] = model_step2_params["cube_size"]


    cross_validation = args.cross_validation
    root_folder = f"./B100/UNetUNet_best/cv{cross_validation}_256_step2/"
    data_div_json = "UNetUNet_v1_data_split.json"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)
    global_config["root_folder"] = root_folder

    model = DynUNet(
        spatial_dims=model_step2_params["spatial_dims"],
        in_channels=model_step2_params["in_channels"],
        out_channels=model_step2_params["out_channels"],
        kernel_size=model_step2_params["kernels"],
        strides=model_step2_params["strides"],
        upsample_kernel_size=model_step2_params["strides"][1:],
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
    print("Model step 2 loaded from", model_step2_params["ckpt_path"])
    model.to(device)
    model.eval()


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
            # JQR130_CTAC_pred_cv2.nii.gz
            STEP1_path = os.path.join(f"B100/UNetUNet_best/cv{cross_validation}_256_clip/{casename}_TOFNAC_CTAC_pred_cv{cross_validation}.nii.gz")
            CTAC_path = os.path.join(f"B100/TC256/{casename}_CTAC_256.nii.gz")
            # load the data
            STEP1_file = nib.load(STEP1_path)
            CTAC_file = nib.load(CTAC_path)

            STEP1_data = STEP1_file.get_fdata()
            CTAC_data = CTAC_file.get_fdata()
            print(f"{split} -> {casename} -> TOFNAC shape: {TOFNAC_data.shape}, CTAC shape: {CTAC_data.shape}")

            # now it is using slide_window to process the 3d data
            # synthetic_CT_data_step_1 # 400, 400, z
            # convert to 1, 1, 400, 400, z
            # norm_step1_data = np.expand_dims(np.expand_dims(norm_step1_data, axis=0), axis=0)
            # norm_step1_data = torch.from_numpy(norm_step1_data).float().cpu()

            input_STEP1 = np.expand_dims(np.expand_dims(STEP1_data, axis=0), axis=0)
            input_STEP1 = torch.from_numpy(input_STEP1).float().to(device)
            # the sliding window method takes 
            # sw_device and device arguments for 
           # the window data and the output volume respectively. 
            # print("Processing step 1 in the shape of ", norm_step1_data.shape)
            # print("Please prepare the data and press enter to continue")
            # time.sleep(30)
            # print("Continuing...")
            with torch.no_grad():
                output_diff = sliding_window_inference(
                    inputs = input_STEP1, 
                    roi_size = model_step2_params["cube_size"],
                    sw_batch_size = 4,
                    predictor = model_step_2,
                    overlap=0.25, 
                    mode="gaussian", 
                    sigma_scale=0.125, 
                    padding_mode="constant", 
                    cval=0.0,
                    device=torch.device('cpu'),
                    sw_device=device,
                    buffer_steps=None,
                ) # f(x) -> y-x

            output_STEP2 = input_STEP1 + output_diff # 0 to 1
            output_STEP2 = output_STEP2.squeeze().detach().cpu().numpy()
            output_STEP2 = output_STEP2 * RANGE_CT + MIN_CT
            print(f"{split} -> {casename} -> output_STEP2 shape: {output_STEP2.shape}")

            # save the CTAC_pred
            CTAC_pred_path = os.path.join(data_split_folder, f"{casename}_CTAC_pred_cv{cross_validation}_step2.nii.gz")
            CTAC_pred_nii = nib.Nifti1Image(output_STEP2, CTAC_file.affine, CTAC_file.header)
            nib.save(CTAC_pred_nii, CTAC_pred_path)
            print(f"Save the CTAC_pred to {CTAC_pred_path}")

            # compute the loss
            CTAC_HU = CTAC_data * data_loader_params["norm"]["RANGE_CT"] + data_loader_params["norm"]["MIN_CT"]
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

if __name__ == "__main__":
    main()
