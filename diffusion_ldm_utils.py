import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

from diffusion_ldm_utils_diffusion_model import UNetModel
from diffusion_ldm_utils_vq_model import VQModel

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,
)

from monai.data import CacheDataset, DataLoader
from diffusion_ldm_config import global_config, set_param, get_param

def prepare_dataset(data_div):
    
    cv = get_param("cv")
    print(cv)
    
    # cv = 0, 1, 2, 3, 4
    cv_test = cv
    cv_val = (cv+1)%5
    cv_train = [(cv+2)%5, (cv+3)%5, (cv+4)%5]

    train_list = data_div[f"cv{cv_train[0]}"] + data_div[f"cv{cv_train[1]}"] + data_div[f"cv{cv_train[2]}"]
    val_list = data_div[f"cv{cv_val}"]
    test_list = data_div[f"cv{cv_test}"]

    set_param("train_list", train_list)
    set_param("val_list", val_list)
    set_param("test_list", test_list)

    print(f"train_list:", train_list)
    print(f"val_list:", val_list)
    print(f"test_list:", test_list)

    exit()

    # construct the data path list
    train_path_list = []
    val_path_list = []
    test_path_list = []

    for hashname in train_list:
        train_path_list.append({
            "TOFNAC": f"TC256_v2/{hashname}_TOFNAC_256.nii.gz",
            "CTAC": f"TC256_v2/{hashname}_CTAC_256.nii.gz",
        })

    for hashname in val_list:
        val_path_list.append({
            "TOFNAC": f"TC256_v2/{hashname}_TOFNAC_256.nii.gz",
            "CTAC": f"TC256_v2/{hashname}_CTAC_256.nii.gz",
        })

    for hashname in test_list:
        test_path_list.append({
            "TOFNAC": f"TC256_v2/{hashname}_TOFNAC_256.nii.gz",
            "CTAC": f"TC256_v2/{hashname}_CTAC_256.nii.gz",
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

    input_modality = global_config["model_step1_params"]["input_modality"]
    # input_modality_dict = {
    #     "x": input_modality[0],
    #     "y": input_modality[1],
    # }
    # img_size = global_config["model_step1_params"]["img_size"]
    # in_channel = global_config["model_step1_params"]["ddconfig"]["in_channels"]
    # out_channel = global_config["model_step1_params"]["ddconfig"]["out_ch"]

    # set the data transform
    train_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
            # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["x"], 
            #     roi_size=(img_size, img_size, in_channel), 
            #     random_size=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["y"],
            #     roi_size=(img_size, img_size, out_channel),
            #     random_size=False),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["x"],
            #     channel_dim=-1),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["y"],
            #     channel_dim="none" if out_channel == 1 else -1),

        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
            # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["x"], 
            #     roi_size=(img_size, img_size, in_channel), 
            #     random_size=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["y"],
            #     roi_size=(img_size, img_size, out_channel),
            #     random_size=False),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["x"],
            #     channel_dim=-1),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["y"],
            #     channel_dim="none" if out_channel == 1 else -1),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
            # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["x"], 
            #     roi_size=(img_size, img_size, in_channel), 
            #     random_size=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["y"],
            #     roi_size=(img_size, img_size, out_channel),
            #     random_size=False),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["x"],
            #     channel_dim=-1),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["y"],
            #     channel_dim="none" if out_channel == 1 else -1),
        ]
    )

    

    train_ds = CacheDataset(
        data=train_path_list,
        transform=train_transforms,
        # cache_num=num_train_files,
        cache_rate=global_config["data_loader_params"]["train"]["cache_rate"],
        num_workers=global_config["data_loader_params"]["train"]["num_workers_cache"],
    )

    val_ds = CacheDataset(
        data=val_path_list,
        transform=val_transforms, 
        # cache_num=num_val_files,
        cache_rate=global_config["data_loader_params"]["val"]["cache_rate"],
        num_workers=global_config["data_loader_params"]["val"]["num_workers_cache"],
    )

    test_ds = CacheDataset(
        data=test_path_list,
        transform=test_transforms,
        # cache_num=num_test_files,
        cache_rate=global_config["data_loader_params"]["test"]["cache_rate"],
        num_workers=global_config["data_loader_params"]["test"]["num_workers_cache"],
    )

    train_loader = DataLoader(train_ds, 
                            batch_size=global_config["data_loader_params"]["train"]["batch_size"],
                            shuffle=global_config["data_loader_params"]["train"]["shuffle"],
                            num_workers=global_config["data_loader_params"]["train"]["num_workers_loader"],

    )
    val_loader = DataLoader(val_ds, 
                            batch_size=global_config["data_loader_params"]["val"]["batch_size"],
                            shuffle=global_config["data_loader_params"]["val"]["shuffle"],
                            num_workers=global_config["data_loader_params"]["val"]["num_workers_loader"],
    )

    test_loader = DataLoader(test_ds,
                            batch_size=global_config["data_loader_params"]["test"]["batch_size"],
                            shuffle=global_config["data_loader_params"]["test"]["shuffle"],
                            num_workers=global_config["data_loader_params"]["test"]["num_workers_loader"],
    )

    return train_loader, val_loader, test_loader

def load_diffusion_vq_model_from(ckpt_path, config):
    
    config_params = config['params']
    config_diffusion_unet = config_params['unet_config']['params']
    config_diffusion_vq = config_params['first_stage_config']['params'] # vq-f4-noattn
    # print(config_diffusion_unet.keys())
    # print(config_diffusion_vq.keys())

    # Load the pretrained weights
    pretrained_weights = torch.load(ckpt_path, map_location='cpu')['state_dict']

    # write output to a file named as "diffusion_ldm_config.txt"
    # with open("diffusion_ldm_config.txt", "w") as f:
    #     for key in pretrained_weights.keys():
    #         f.write(key)
    #         f.write("\n")

    # Create a new state dictionary with modified keys
    diffusion_state_dict = {}
    vq_state_dict = {}

    for key, value in pretrained_weights.items():
        if "model.diffusion_model." in key:
            new_key = key.replace("model.diffusion_model.", "")
            diffusion_state_dict[new_key] = value
        if "first_stage_model." in key:
            new_key = key.replace("first_stage_model.", "")
            vq_state_dict[new_key] = value

    # Load the modified state dictionary into the new model
    diffusion_model = UNetModel(**config_diffusion_unet)
    vq_model = VQModel(**config_diffusion_vq)
    # set strict=True for making sure load correct pretrained weights
    diffusion_model.load_state_dict(diffusion_state_dict, strict=True)
    vq_model.load_state_dict(vq_state_dict, strict=True)

    return diffusion_model, vq_model

def make_batch_PET_CT_CT(path):
    image_dict = np.load(path, allow_pickle=True).item()
    PET_img = image_dict['PET_img']
    PET_mask = image_dict['PET_mask']
    CT0_img = image_dict['CT0_img']
    CT1_img = image_dict['CT1_img']
    # they are noramlized to [0,1]
    PET_img = PET_img * 2.0 - 1.0
    CT0_img = CT0_img * 2.0 - 1.0
    CT1_img = CT1_img * 2.0 - 1.0
    # now they are in -1 to 1
    # they are in shape 256,256,3
    PET_img = PET_img.transpose(2,0,1)
    CT0_img = CT0_img.transpose(2,0,1)
    CT1_img = CT1_img.transpose(2,0,1)
    # convert to tensor
    PET_img = torch.from_numpy(PET_img)
    PET_mask = torch.from_numpy(PET_mask)
    CT0_img = torch.from_numpy(CT0_img)
    CT1_img = torch.from_numpy(CT1_img)
    # add new axis to make them in shape 1,3,256,256
    PET_img = PET_img.unsqueeze(0).float()
    PET_mask = PET_mask.unsqueeze(0).unsqueeze(0).float()
    CT0_img = CT0_img.unsqueeze(0).float()
    CT1_img = CT1_img.unsqueeze(0).float()

    return PET_img, PET_mask, CT0_img, CT1_img


def load_image(image, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    batch = {"image": image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch