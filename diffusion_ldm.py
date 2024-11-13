# 1, load the configuration from diffusion_ldm_config.json
# ----------------------------------------------------
import yaml

with open("diffusion_ldm_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# print(config)
# ----------------------------------------------------

# 2, load the model from the configuration
# ----------------------------------------------------
import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

# from diffusion_ldm_utils_diffusion_model import UNetModel
# from diffusion_ldm_utils_vq_model import VQModel
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from diffusion_ldm_utils import load_diffusion_vq_model_from, make_batch, make_batch_PET_CT_CT

# pip install omegaconf
# pip install pip install pillow
# pip install torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--indir", type=str, default="./inpaint")
parser.add_argument("--outdir", type=str, default="./inpaint_output")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--ckpt_path", type=str, default="model_inpaint.ckpt")
parser.add_argument("--config_path", type=str, default="diffusion_ldm_config.yaml")
parser.add_argument("--test_path", type=str, default="James_data_v3/diffusion_slices/pE4055_E4058_z100_n01.npy")

# load experiment config
opt = parser.parse_args()
print(opt)

# load input
masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
images = [x.replace("_mask.png", ".png") for x in masks]
print(f"Found {len(masks)} inputs.")

# load pretrained model config
config = OmegaConf.load(opt.config_path)

# diffsuion_model, vq_model = load_diffusion_vq_model_from(opt.ckpt_path, config)
# print("Create a diffusion model and a vq model from the pretrained weights {}".format(opt.ckpt_path))

model = instantiate_from_config(config.model)
model.load_state_dict(torch.load(opt.ckpt_path)["state_dict"], strict=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"The current device is {device}")
model = model.to(device)
sampler = DDIMSampler(model)

model.freeze_vq_model()

PET_img, PET_mask, CT0_img, CT1_img = make_batch_PET_CT_CT(opt.test_path)
# print(PET_img.size(), PET_mask.size(), CT0_img.size(), CT1_img.size())
# torch.Size([1, 3, 256, 256]) torch.Size([1, 1, 256, 256]) torch.Size([1, 3, 256, 256]) torch.Size([1, 3, 256, 256])
PET_img = PET_img.to(device)
PET_mask = PET_mask.to(device)
CT0_img = CT0_img.to(device)
CT1_img = CT1_img.to(device)


import datetime
import torch.optim as optim

# Set up directories
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
logdir = f"./logs/{now}"
ckptdir = os.path.join(logdir, "checkpoints")
os.makedirs(ckptdir, exist_ok=True)

# Load configuration
# config = OmegaConf.load("path/to/your_config.yaml")
# train_config = config["model"]["params"]
# base_learning_rate = train_config.base_learning_rate
# linear_start = train_config.params.linear_start
# linear_end = train_config.params.linear_end
# timesteps = train_config.params.timesteps
# image_size = train_config.params.image_size
# channels = train_config.params.channels

base_learning_rate = 1.0e-06
linear_start = 0.0015
linear_end = 0.0205
timesteps = 1000


optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate)

# Learning rate adjustment
def adjust_learning_rate(optimizer, epoch, base_lr):
    # Example: Linear decay from linear_start to linear_end over epochs
    lr = base_lr * (1 - epoch / timesteps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training and validation loop
best_val_loss = float("inf")


ct0_64 = model.cond_stage_model.encode(CT0_img)
pet_64 = model.cond_stage_model.encode(PET_img)
ct1_64 = model.cond_stage_model.encode(CT1_img)
mask_64 = torch.nn.functional.interpolate(PET_mask, size=ct0_64.shape[-2:])
cc = mask_64

c = pet_64
x_T = ct1_64
c = torch.cat((c, cc), dim=1) # channel = 4
shape = (c.shape[1]-1,)+c.shape[2:]

for idz in range(100):
    samples_ddim, _ = sampler.sample(
        S=opt.steps,
        conditioning=c,
        batch_size=c.shape[0],
        shape=shape,
        verbose=False,
        x_T=x_T
    )

    # compute loss between samples_ddim to ct0_64
    loss = torch.nn.functional.mse_loss(samples_ddim, ct0_64)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {idz}, Loss {loss.item()}")

# ----------------------------------------------------

# perform the test

with torch.no_grad():
    with model.ema_scope():
        outpath = os.path.dirname(opt.test_path)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", outpath)
        # c = model.cond_stage_model.encode(CT0_img) # channel = 3
        # c = model.cond_stage_model.encode(PET_img) # channel = 3
        # cc = torch.nn.functional.interpolate(PET_mask, size=c.shape[-2:]) # channel = 1
        # x_T = model.cond_stage_model.encode(CT1_img) # channel = 3
        # cc = PET_mask

        ct0_64 = model.cond_stage_model.encode(CT0_img)
        pet_64 = model.cond_stage_model.encode(PET_img)
        ct1_64 = model.cond_stage_model.encode(CT1_img)
        mask_64 = torch.nn.functional.interpolate(PET_mask, size=ct0_64.shape[-2:])
        
        savename_list = [
            [opt.test_path.replace(".npy", "_ct0_c_e100.npy"), ct0_64, None],
            [opt.test_path.replace(".npy", "_pet_c_e100.npy"), pet_64, None],
            [opt.test_path.replace(".npy", "_ct0_c_ct1_xT_e100.npy"), ct0_64, ct1_64],
            [opt.test_path.replace(".npy", "_pet_c_ct1_xT_e100.npy"), pet_64, ct1_64],
        ]

        cc = mask_64

        for config in savename_list:
            savename = config[0]
            c = config[1]
            x_T = config[2]
            c = torch.cat((c, cc), dim=1) # channel = 4
            shape = (c.shape[1]-1,)+c.shape[2:]
            samples_ddim, _ = sampler.sample(
                S=opt.steps,
                conditioning=c,
                batch_size=c.shape[0],
                shape=shape,
                verbose=False,
                x_T=x_T
            )
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            image = torch.clamp((CT0_img+1.0)/2.0, min=0.0, max=1.0)
            mask = torch.clamp((PET_mask+1.0)/2.0, min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
            inpainted = (1-mask)*image+mask*predicted_image
            inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]
            np.save(savename, inpainted)
            print("The output file is saved to", savename)




# check input size
# image = images[0]
# mask = masks[0]
# batch = make_batch(image, mask, device=torch.device('cpu'))
# print(batch["image"].size(), batch["mask"].size(), batch["masked_image"].size())
# torch.Size([1, 3, 512, 512]) torch.Size([1, 1, 512, 512]) torch.Size([1, 3, 512, 512])

# os.makedirs(opt.outdir, exist_ok=True)
# with torch.no_grad():
#     with model.ema_scope():
#         for image, mask in tqdm(zip(images, masks)):
#             outpath = os.path.join(opt.outdir, os.path.split(image)[1])
#             print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", outpath)
#             batch = make_batch(image, mask, device=device)

#             # encode masked image and concat downsampled mask
#             c = model.cond_stage_model.encode(batch["masked_image"]) # channel = 3
#             cc = torch.nn.functional.interpolate(batch["mask"],
#                                                     size=c.shape[-2:]) # channel = 1
#             c = torch.cat((c, cc), dim=1) # channel = 4

#             shape = (c.shape[1]-1,)+c.shape[2:]
#             samples_ddim, _ = sampler.sample(S=opt.steps,
#                                                 conditioning=c,
#                                                 batch_size=c.shape[0],
#                                                 shape=shape,
#                                                 verbose=False)
#             x_samples_ddim = model.decode_first_stage(samples_ddim)

#             image = torch.clamp((batch["image"]+1.0)/2.0,
#                                 min=0.0, max=1.0)
#             mask = torch.clamp((batch["mask"]+1.0)/2.0,
#                                 min=0.0, max=1.0)
#             predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
#                                             min=0.0, max=1.0)

#             inpainted = (1-mask)*image+mask*predicted_image
#             inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
#             Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
# ----------------------------------------------------