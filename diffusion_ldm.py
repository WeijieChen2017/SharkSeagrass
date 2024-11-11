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

from diffusion_ldm_utils import load_diffusion_vq_model_from, make_batch

# pip install omegaconf
# pip install pip install pillow
# pip install torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--indir", type=str, default="./inpaint")
parser.add_argument("--outdir", type=str, default="./inpaint_output")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--ckpt_path", type=str, default="model_inpaint.ckpt")
parser.add_argument("--config_path", type=str, default="diffusion_ldm_config.yaml")

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



# check input size
# image = images[0]
# mask = masks[0]
# batch = make_batch(image, mask, device=torch.device('cpu'))
# print(batch["image"].size(), batch["mask"].size(), batch["masked_image"].size())
# torch.Size([1, 3, 512, 512]) torch.Size([1, 1, 512, 512]) torch.Size([1, 3, 512, 512])

os.makedirs(opt.outdir, exist_ok=True)
with torch.no_grad():
    with model.ema_scope():
        for image, mask in tqdm(zip(images, masks)):
            outpath = os.path.join(opt.outdir, os.path.split(image)[1])
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", outpath)
            batch = make_batch(image, mask, device=device)

            # encode masked image and concat downsampled mask
            c = model.cond_stage_model.encode(batch["masked_image"]) # channel = 3
            cc = torch.nn.functional.interpolate(batch["mask"],
                                                    size=c.shape[-2:]) # channel = 1
            c = torch.cat((c, cc), dim=1) # channel = 4

            shape = (c.shape[1]-1,)+c.shape[2:]
            samples_ddim, _ = sampler.sample(S=opt.steps,
                                                conditioning=c,
                                                batch_size=c.shape[0],
                                                shape=shape,
                                                verbose=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)

            image = torch.clamp((batch["image"]+1.0)/2.0,
                                min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                            min=0.0, max=1.0)

            inpainted = (1-mask)*image+mask*predicted_image
            inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
            Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
# ----------------------------------------------------