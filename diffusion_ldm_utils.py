import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

from diffusion_ldm_utils_diffusion_model import UNetModel
from diffusion_ldm_utils_vq_model import VQModel



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