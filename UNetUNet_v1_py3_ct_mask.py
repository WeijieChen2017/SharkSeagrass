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

from scipy.ndimage import binary_fill_holes

MIN_CT = -1024
MAX_CT = 2976
RANGE_CT = MAX_CT - MIN_CT
WRONG_MAX_CT = 1976
WRONG_RANGE_CT = WRONG_MAX_CT - MIN_CT

def correct_CT_range(DLCTAC_data):
    DLCTAC_data = (DLCTAC_data - MIN_CT) / WRONG_RANGE_CT
    DLCTAC_data = DLCTAC_data * RANGE_CT + MIN_CT
    return DLCTAC_data

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
    argparser.add_argument('--acs', type=bool, default=True, help='Whether to run in a/c/s three directions')
    argparser.add_argument('-p', '--part2', type=bool, default=False, help='Whether to run the second part')
    args = argparser.parse_args()
    tag = f"fold{args.cross_validation}_256"

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
    # root_folder = f"B100/UNetUNet_best/cv{cross_validation}_256/"
    root_folder = f"results/cv{cross_validation}_256/"
    data_div_json = "UNetUNet_v1_data_split.json"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)

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
        "img_size" : 256,
        "input_modality" : ["TOFNAC", "CTAC"],
        # "ckpt_path": f"B100/TC256_best_ckpt/best_model_cv{cross_validation}.pth",
        "ckpt_path": root_folder+f"best_model_cv{cross_validation}.pth",
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

    if args.acs:
        data_folder = "B100/TC256/"
        data_split = ["test", "train", "val"]
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
            mask_save_folder = os.path.join(root_folder, split+"_mask")
            if not os.path.exists(mask_save_folder):
                os.makedirs(mask_save_folder)
                
            for casename in split_list:
                print(f"{split} -> Processing {casename}")
                
                CTAC_pred_axial_path = os.path.join(data_split_folder, f"{casename}_CTAC_pred_axial_cv{cross_validation}.nii.gz")
                CTAC_pred_coronal_path = os.path.join(data_split_folder, f"{casename}_CTAC_pred_coronal_cv{cross_validation}.nii.gz")
                CTAC_pred_sagittal_path = os.path.join(data_split_folder, f"{casename}_CTAC_pred_sagittal_cv{cross_validation}.nii.gz")
                CTAC_pred_average_path = os.path.join(data_split_folder, f"{casename}_CTAC_pred_average_cv{cross_validation}.nii.gz")
                CTAC_pred_median_path = os.path.join(data_split_folder, f"{casename}_CTAC_pred_median_cv{cross_validation}.nii.gz")

                CT_pred_axial_file = nib.load(CT_pred_axial_path)
                CT_pred_coronal_file = nib.load(CT_pred_coronal_path)
                CT_pred_sagittal_file = nib.load(CT_pred_sagittal_path)
                CT_pred_average_file = nib.load(CT_pred_average_path)
                CT_pred_median_file = nib.load(CT_pred_median_path)

                CT_pred_axial_data = CT_pred_axial_file.get_fdata()
                CT_pred_coronal_data = CT_pred_coronal_file.get_fdata()
                CT_pred_sagittal_data = CT_pred_sagittal_file.get_fdata()
                CT_pred_average_data = CT_pred_average_file.get_fdata()
                CT_pred_median_data = CT_pred_median_file.get_fdata()
                
                # adjust the range of the CT data
                CT_pred_axial_data = correct_CT_range(CT_pred_axial_data)
                CT_pred_coronal_data = correct_CT_range(CT_pred_coronal_data)
                CT_pred_sagittal_data = correct_CT_range(CT_pred_sagittal_data)
                CT_pred_average_data = correct_CT_range(CT_pred_average_data)
                CT_pred_median_data = correct_CT_range(CT_pred_median_data)

                # save the data
                CTAC_pred_axial_rescale_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_axial_rescale_cv{cross_validation}.nii.gz")
                CTAC_pred_coronal_rescale_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_coronal_rescale_cv{cross_validation}.nii.gz")
                CTAC_pred_sagittal_rescale_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_sagittal_rescale_cv{cross_validation}.nii.gz")
                CTAC_pred_average_rescale_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_average_rescale_cv{cross_validation}.nii.gz")
                CTAC_pred_median_rescale_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_median_rescale_cv{cross_validation}.nii.gz")

                CTAC_pred_axial_rescale_file = nib.Nifti1Image(CT_pred_axial_data, CT_pred_axial_file.affine, CT_pred_axial_file.header)
                CTAC_pred_coronal_rescale_file = nib.Nifti1Image(CT_pred_coronal_data, CT_pred_coronal_file.affine, CT_pred_coronal_file.header)
                CTAC_pred_sagittal_rescale_file = nib.Nifti1Image(CT_pred_sagittal_data, CT_pred_sagittal_file.affine, CT_pred_sagittal_file.header)
                CTAC_pred_average_rescale_file = nib.Nifti1Image(CT_pred_average_data, CT_pred_average_file.affine, CT_pred_average_file.header)
                CTAC_pred_median_rescale_file = nib.Nifti1Image(CT_pred_median_data, CT_pred_median_file.affine, CT_pred_median_file.header)

                nib.save(CTAC_pred_axial_rescale_file, CTAC_pred_axial_rescale_path)
                nib.save(CTAC_pred_coronal_rescale_file, CTAC_pred_coronal_rescale_path)
                nib.save(CTAC_pred_sagittal_rescale_file, CTAC_pred_sagittal_rescale_path)
                nib.save(CTAC_pred_average_rescale_file, CTAC_pred_average_rescale_path)
                nib.save(CTAC_pred_median_rescale_file, CTAC_pred_median_rescale_path)
                print(f"Axial, coronal, sagittal, average, and median masks saved for {casename}")

                # compute the mask
                axial_mask = CT_pred_axial_data > -500
                coronal_mask = CT_pred_coronal_data > -500
                sagittal_mask = CT_pred_sagittal_data > -500
                average_mask = CT_pred_average_data > -500
                median_mask = CT_pred_median_data > -500

                # fill holes
                for z in range(CT_pred_axial_data.shape[2]):
                    axial_mask[:, :, z] = binary_fill_holes(axial_mask[:, :, z])
                    coronal_mask[:, :, z] = binary_fill_holes(coronal_mask[:, :, z])
                    sagittal_mask[:, :, z] = binary_fill_holes(sagittal_mask[:, :, z])
                    average_mask[:, :, z] = binary_fill_holes(average_mask[:, :, z])
                    median_mask[:, :, z] = binary_fill_holes(median_mask[:, :, z])

                # save the mask
                CTAC_pred_axial_mask_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_axial_mask_cv{cross_validation}.nii.gz")
                CTAC_pred_coronal_mask_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_coronal_mask_cv{cross_validation}.nii.gz")
                CTAC_pred_sagittal_mask_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_sagittal_mask_cv{cross_validation}.nii.gz")
                CTAC_pred_average_mask_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_average_mask_cv{cross_validation}.nii.gz")
                CTAC_pred_median_mask_path = os.path.join(mask_save_folder, f"{casename}_CTAC_pred_median_mask_cv{cross_validation}.nii.gz")

                CTAC_pred_axial_mask_file = nib.Nifti1Image(axial_mask, CT_pred_axial_file.affine, CT_pred_axial_file.header)
                CTAC_pred_coronal_mask_file = nib.Nifti1Image(coronal_mask, CT_pred_coronal_file.affine, CT_pred_coronal_file.header)
                CTAC_pred_sagittal_mask_file = nib.Nifti1Image(sagittal_mask, CT_pred_sagittal_file.affine, CT_pred_sagittal_file.header)
                CTAC_pred_average_mask_file = nib.Nifti1Image(average_mask, CT_pred_average_file.affine, CT_pred_average_file.header)
                CTAC_pred_median_mask_file = nib.Nifti1Image(median_mask, CT_pred_median_file.affine, CT_pred_median_file.header)

                nib.save(CTAC_pred_axial_mask_file, CTAC_pred_axial_mask_path)
                nib.save(CTAC_pred_coronal_mask_file, CTAC_pred_coronal_mask_path)
                nib.save(CTAC_pred_sagittal_mask_file, CTAC_pred_sagittal_mask_path)
                nib.save(CTAC_pred_average_mask_file, CTAC_pred_average_mask_path)
                nib.save(CTAC_pred_median_mask_file, CTAC_pred_median_mask_path)

                print(f"Axial, coronal, sagittal, average, and median masks saved for {casename}")

        print("Done!")
    
if __name__ == "__main__":
    main()
