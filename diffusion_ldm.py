# 1, load the configuration from diffusion_ldm_config.json
# ----------------------------------------------------
import yaml

with open("diffusion_ldm_config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)
# ----------------------------------------------------

# 2, load the model from the configuration
# ----------------------------------------------------
import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# pip install omegaconf
# pip install pip install pillow
# pip install torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--indir", type=str, default="./inpaint")
parser.add_argument("--outdir", type=str, default="./inpaint_output")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--ckpt_path", type=str, default="model_inpaint.ckpt")
parser.add_argument("--config_path", type=str, default="diffusion_ldm_config.yaml")

opt = parser.parse_args()

print(opt)
masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
images = [x.replace("_mask.png", ".png") for x in masks]
print(f"Found {len(masks)} inputs.")

config = OmegaConf.load(opt.config_path)
model = instantiate_from_config(config.model)
model.load_state_dict(torch.load(opt.ckpt_path)["state_dict"],
                        strict=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)

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

os.makedirs(opt.outdir, exist_ok=True)
with torch.no_grad():
    with model.ema_scope():
        for image, mask in tqdm(zip(images, masks)):
            outpath = os.path.join(opt.outdir, os.path.split(image)[1])
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", outpath)
            batch = make_batch(image, mask, device=device)

            # encode masked image and concat downsampled mask
            c = model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(batch["mask"],
                                                    size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)

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