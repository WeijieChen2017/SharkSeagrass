import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import json
import random

from diffusion_ldm_utils_diffusion_model import UNetModel
from diffusion_ldm_utils_vq_model import VQModel

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,
)

import time
from monai.data import CacheDataset, DataLoader
from diffusion_ldm_config import global_config, set_param, get_param

def printlog(message):
    log_txt_path = get_param("log_txt_path")
    # attach the current time as YYYY-MM-DD HH:MM:SS 
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    message = f"{current_time} {message}"
    print(message)
    with open(log_txt_path, "a") as f:
        f.write(message)
        f.write("\n")


def train_or_eval_or_test_the_batch(batch, batch_size, stage, model, optimizer, device):

    pet = batch["PET"] # 1, z, 256, 256
    ct = batch["CT"] # 1, z, 256, 256
    len_z = pet.shape[1]

    es = get_param("es")

    pet = pet * 2 - 1
    ct = ct * 2 - 1

    # if pet size and ct size are not the same skip this batch
    if pet.shape != ct.shape:
        printlog(f"skip this batch, pet shape: {pet.shape}, ct shape: {ct.shape}")
        return 1.0, 1.0, 1.0

    # 1, z, 256, 256 tensor
    case_loss_first = 0.0
    case_loss_second = 0.0
    case_loss_third = 0.0

    # pad shape
    if len_z % es != 0:
        pad_size = es - len_z % es
        pet = torch.nn.functional.pad(pet, (0, 0, 0, 0, 0, pad_size, 0, 0), mode='constant', value=0)
        ct = torch.nn.functional.pad(ct, (0, 0, 0, 0, 0, pad_size, 0, 0), mode='constant', value=0)

    printlog(f"pet shape {pet.shape}, ct shape {ct.shape}")

    indices_list_first = [i for i in range(1, pet.shape[1]-1)]
    indices_list_second = [i for i in range(1, pet.shape[2]-1)]
    indices_list_third = [i for i in range(1, pet.shape[3]-1)]

    random.shuffle(indices_list_first)
    random.shuffle(indices_list_second)
    random.shuffle(indices_list_third)

    # enumreate first dimension
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, pet.shape[2], pet.shape[3]))
    batch_y = torch.zeros((batch_size, 3, ct.shape[2], ct.shape[3]))
    for indices in indices_list_first:
        slice_x = pet[:, indices-1:indices+2, :, :]
        slice_y = ct[:, indices-1:indices+2, :, :]
        batch_size_count += 1

        batch_x[batch_size_count-1] = slice_x
        batch_y[batch_size_count-1] = slice_y

        if batch_size_count < batch_size and indices != indices_list_first[-1]:
            continue
        else:
            # we get a batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded_batch_y = model.first_stage_model.encode(batch_y)
            if stage == "train":
                optimizer.zero_grad()
                loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                loss.backward()
                optimizer.step()
                case_loss_first += loss.item()
            elif stage == "eval" or stage == "test":
                with torch.no_grad():
                    loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                    case_loss_first += loss.item()
            batch_size_count = 0
        
        case_loss_first = case_loss_first / (len(indices_list_first) // batch_size + 1)
    
    # enumreate second dimension
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, pet.shape[1], pet.shape[3]))
    batch_y = torch.zeros((batch_size, 3, ct.shape[1], ct.shape[3]))

    for indices in indices_list_second:
        slice_x = pet[:, :, indices-1:indices+2, :]
        slice_y = ct[:, :, indices-1:indices+2, :]
        # adjust the index order
        slice_x = slice_x.permute(0, 2, 1, 3)
        slice_y = slice_y.permute(0, 2, 1, 3)
        batch_size_count += 1

        batch_x[batch_size_count-1] = slice_x
        batch_y[batch_size_count-1] = slice_y

        if batch_size_count < batch_size and indices != indices_list_second[-1]:
            continue
        else:
            # we get a batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded_batch_y = model.first_stage_model.encode(batch_y)
            print(batch_x.size(), batch_y.size())
            if stage == "train":
                optimizer.zero_grad()
                loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                loss.backward()
                optimizer.step()
                case_loss_second += loss.item()
            elif stage == "eval" or stage == "test":
                with torch.no_grad():
                    loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                    case_loss_second += loss.item()
            batch_size_count = 0
        
        case_loss_second = case_loss_second / (len(indices_list_second) // batch_size + 1)
    
    # enumreate third dimension
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, pet.shape[1], pet.shape[2]))
    batch_y = torch.zeros((batch_size, 3, ct.shape[1], ct.shape[2]))

    for indices in indices_list_third:
        slice_x = pet[:, :, :, indices-1:indices+2]
        slice_y = ct[:, :, :, indices-1:indices+2]
        # adjust the index order
        slice_x = slice_x.permute(0, 3, 1, 2)
        slice_y = slice_y.permute(0, 3, 1, 2)
        batch_size_count += 1

        batch_x[batch_size_count-1] = slice_x
        batch_y[batch_size_count-1] = slice_y

        if batch_size_count < batch_size and indices != indices_list_third[-1]:
            continue
        else:
            # we get a batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded_batch_y = model.first_stage_model.encode(batch_y)
            if stage == "train":
                optimizer.zero_grad()
                loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                loss.backward()
                optimizer.step()
                case_loss_third += loss.item()
            elif stage == "eval" or stage == "test":
                with torch.no_grad():
                    loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                    case_loss_third += loss.item()
            batch_size_count = 0
        
        case_loss_third = case_loss_third / (len(indices_list_third) // batch_size + 1)

    return case_loss_first, case_loss_second, case_loss_third




def prepare_dataset(data_div):
    
    cv = get_param("cv")
    root = get_param("root")
    
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

    # train_list: ['E4058', 'E4217', 'E4166', 'E4165', 'E4092', 'E4163', 'E4193', 'E4105', 'E4125', 'E4198', 'E4157', 'E4139', 'E4207', 'E4106', 'E4068', 'E4241', 'E4219', 'E4078', 'E4147', 'E4138', 'E4096', 'E4152', 'E4073', 'E4181', 'E4187', 'E4099', 'E4077', 'E4134', 'E4091', 'E4144', 'E4114', 'E4130', 'E4103', 'E4239', 'E4183', 'E4208', 'E4120', 'E4220', 'E4137', 'E4069', 'E4189', 'E4182']
    # val_list: ['E4216', 'E4081', 'E4118', 'E4074', 'E4079', 'E4094', 'E4115', 'E4237', 'E4084', 'E4061', 'E4055', 'E4098', 'E4232']
    # test_list: ['E4128', 'E4172', 'E4238', 'E4158', 'E4129', 'E4155', 'E4143', 'E4197', 'E4185', 'E4131', 'E4162', 'E4066', 'E4124']

    # construct the data path list
    train_path_list = []
    val_path_list = []
    test_path_list = []

    for hashname in train_list:
        train_path_list.append({
            "PET": f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
        })

    for hashname in val_list:
        val_path_list.append({
            "PET": f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
        })

    for hashname in test_list:
        test_path_list.append({
            "PET": f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
        })

    # save the data division file
    data_division_file = os.path.join(root, "data_division.json")
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

    input_modality = ["PET", "CT"]

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
        cache_rate=get_param("data_param")["dataset"]["train"]["cache_rate"],
        num_workers=get_param("data_param")["dataset"]["train"]["num_workers"],
    )

    val_ds = CacheDataset(
        data=val_path_list,
        transform=val_transforms, 
        # cache_num=num_val_files,
        cache_rate=get_param("data_param")["dataset"]["val"]["cache_rate"],
        num_workers=get_param("data_param")["dataset"]["val"]["num_workers"],
    )

    test_ds = CacheDataset(
        data=test_path_list,
        transform=test_transforms,
        # cache_num=num_test_files,
        cache_rate=get_param("data_param")["dataset"]["test"]["cache_rate"],
        num_workers=get_param("data_param")["dataset"]["test"]["num_workers"],
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=1,
        shuffle=True,
        num_workers=get_param("data_param")["dataloader"]["train"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1,
        shuffle=False,
        num_workers=get_param("data_param")["dataloader"]["val"]["num_workers"],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=get_param("data_param")["dataloader"]["test"]["num_workers"],
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