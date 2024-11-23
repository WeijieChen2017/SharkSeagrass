import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch
import json

# from diffusion_ldm_utils_diffusion_model import UNetModel
# from diffusion_ldm_utils_vq_model import VQModel
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from diffusion_ldm_utils import load_diffusion_vq_model_from, prepare_dataset
from diffusion_ldm_utils import make_batch, make_batch_PET_CT_CT, load_image
from diffusion_ldm_utils import train_or_eval_or_test_the_batch, printlog

from diffusion_ldm_config import global_config, set_param, get_param

# pip install omegaconf
# pip install pip install pillow
# pip install torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="results/diffusion_ldm_vanilla_scaling_c")
parser.add_argument("--seed", type=int, default=729)
parser.add_argument("--data_div", type=str, default="James_data_v3/cv_list.json")
parser.add_argument("--indir", type=str, default="./semantic_synthesis256")
parser.add_argument("--outdir", type=str, default="./semantic_synthesis256_output")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--ckpt_path", type=str, default="results/diffusion_ldm_vanilla_scaling_c/best.pth")
parser.add_argument("--ldm_config_path", type=str, default="diffusion_ldm_config_semantic_synthesis256.yaml")
parser.add_argument("--experiment_config_path", type=str, default="diffusion_ldm_v1_config.yaml")
parser.add_argument("--test_path", type=str, default="James_data_v3/diffusion_slices/pE4055_E4058_z100_n01.npy")

# load experiment config
opt = parser.parse_args()
print(opt)
root_dir = opt.root
os.makedirs(root_dir, exist_ok=True)
experiment_config_path = opt.experiment_config_path
# use yaml to load the config
experiment_config = OmegaConf.load(experiment_config_path)
print(experiment_config)

# iteratively load the experiment_config using set_param
for key in experiment_config.keys():
    set_param(key, experiment_config[key])

# set random seed
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

set_param("cv", 0)
set_param("root", root_dir)
set_param("seed", opt.seed)
set_param("log_txt_path", os.path.join(root_dir, "log.txt"))



# load data data division
data_div_json = opt.data_div
with open(data_div_json, "r") as f:
    data_div = json.load(f)

# train_loader, val_loader, test_loader = prepare_dataset(data_div)

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(f"The current device is {device}")

# load pretrained model config
config = OmegaConf.load(opt.ldm_config_path)

model = instantiate_from_config(config.model)
model.load_state_dict(torch.load(opt.ckpt_path, map_location="cpu")["state_dict"], strict=False)
model = model.to(device)
sampler = DDIMSampler(model)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f"The current device is {device}")
# model = model.to(device)
# sampler = DDIMSampler(model)

# model.freeze_vq_model()

# embedding_scale_1 = len(config.model.params.unet_config.params.channel_mult) - 1
# embedding_scale_2 = len(config.model.params.first_stage_config.params.ddconfig.ch_mult) - 1
# embedding_scale_1 = 2 ** embedding_scale_1
# embedding_scale_2 = 2 ** embedding_scale_2
# es = embedding_scale_1 * embedding_scale_2
# print("The pixel scaling factor is ", es)
# set_param("es", es)

# import datetime
# import torch.optim as optim

# # Set up directories
# now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# logdir = f"./logs/{now}"
# ckptdir = os.path.join(logdir, "checkpoints")
# os.makedirs(ckptdir, exist_ok=True)

# # Load configuration
# # config = OmegaConf.load("path/to/your_config.yaml")
# # train_config = config["model"]["params"]
# # base_learning_rate = train_config.base_learning_rate
# # linear_start = train_config.params.linear_start
# # linear_end = train_config.params.linear_end
# # timesteps = train_config.params.timesteps
# # image_size = train_config.params.image_size
# # channels = train_config.params.channels

# base_learning_rate = 1.0e-06
# linear_start = 0.0015
# linear_end = 0.0205
# timesteps = 1000


# optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate)
# # loss_fn = torch.nn.MSELoss()

# # Learning rate adjustment
# def adjust_learning_rate(optimizer, epoch, base_lr):
#     # Example: Linear decay from linear_start to linear_end over epochs
#     lr = base_lr * (1 - epoch / timesteps)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# # Training and validation loop
# best_val_loss = float("inf")
# epoch = get_param("train_param")["epoch"]
# for idx_epoch in range(epoch):

#     printlog(f"Epoch [{idx_epoch}]/[{epoch}]")

#     # ===============training stage===============

#     model.train()
#     loss_1st = 0.0
#     loss_2nd = 0.0
#     loss_3rd = 0.0
#     total_case_train = len(train_loader)

#     for idx_case, batch in enumerate(train_loader):
#         cl_1, cl_2, cl_3 = train_or_eval_or_test_the_batch(
#             batch=batch,
#             batch_size=get_param("train_param")["train_stage"]["batch_size"],
#             stage="train",
#             model=model,
#             optimizer=optimizer,
#             device=device,
#         )
#         loss_1st += cl_1
#         loss_2nd += cl_2
#         loss_3rd += cl_3
#         printlog(f"<Train> Epoch [{idx_epoch}]/[{epoch}], Case [{idx_case}]/[{total_case_train}], Loss 1st {cl_1:.6f}, Loss 2nd {cl_2:.6f}, Loss 3rd {cl_3:.6f}")


#     loss_1st /= len(train_loader)
#     loss_2nd /= len(train_loader)
#     loss_3rd /= len(train_loader)
#     avg_loss = (loss_1st + loss_2nd + loss_3rd) / 3
#     printlog(f"<Train> Epoch [{idx_epoch}]/[{epoch}], Loss 1st {loss_1st:.6f}, Loss 2nd {loss_2nd:.6f}, Loss 3rd {loss_3rd:.6f}, Avg Loss {avg_loss:.6f}")

#     # ===============validation stage===============
#     model.eval()
#     loss_1st = 0.0
#     loss_2nd = 0.0
#     loss_3rd = 0.0
#     total_case_val = len(val_loader)

#     for idx_case, batch in enumerate(val_loader):
#         cl_1, cl_2, cl_3 = train_or_eval_or_test_the_batch(
#             batch=batch,
#             batch_size=get_param("train_param")["val_stage"]["batch_size"],
#             stage="val",
#             model=model,
#             device=device,
#         )
#         loss_1st += cl_1
#         loss_2nd += cl_2
#         loss_3rd += cl_3
#         printlog(f"<Val> Epoch [{idx_epoch}]/[{epoch}], Case [{idx_case}]/[{total_case_val}], Loss 1st {cl_1:.6f}, Loss 2nd {cl_2:.6f}, Loss 3rd {cl_3:.6f}")

#     loss_1st /= len(val_loader)
#     loss_2nd /= len(val_loader)
#     loss_3rd /= len(val_loader)
#     avg_loss = (loss_1st + loss_2nd + loss_3rd) / 3
#     printlog(f"<Val> Epoch [{idx_epoch}]/[{epoch}], Loss 1st {loss_1st:.6f}, Loss 2nd {loss_2nd:.6f}, Loss 3rd {loss_3rd:.6f}, Avg Loss {avg_loss:.6f}")
    
#     if avg_loss < best_val_loss:
#         best_val_loss = avg_loss
#         torch.save({
#             "state_dict": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "epoch": idx_epoch,
#             "loss": avg_loss,
#         }, os.path.join(ckptdir, "best.pth"))
#         printlog(f"Best model saved at epoch {idx_epoch}")
    
#     if idx_epoch % get_param("train_param")["save_per_epoch"] == 0:
#         torch.save({
#             "state_dict": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "epoch": idx_epoch,
#             "loss": avg_loss,
#         }, os.path.join(ckptdir, f"epoch_{idx_epoch}.pth"))
#         printlog(f"Model saved at epoch {idx_epoch}")

#     # Learning rate adjustment
#     adjust_learning_rate(optimizer, idx_epoch, base_learning_rate)

# ----------------------------------------------------












PET_img, PET_mask, CT0_img, CT1_img = make_batch_PET_CT_CT(opt.test_path)
# print(PET_img.size(), PET_mask.size(), CT0_img.size(), CT1_img.size())
# torch.Size([1, 3, 256, 256]) torch.Size([1, 1, 256, 256]) torch.Size([1, 3, 256, 256]) torch.Size([1, 3, 256, 256])
PET_img = PET_img.to(device)
# PET_mask = PET_mask.to(device)
CT0_img = CT0_img.to(device)
CT1_img = CT1_img.to(device)

# # ct0_64 = model.first_stage_model.encode(CT0_img)
# # pet_64 = model.first_stage_model.encode(PET_img)
# # ct1_64 = model.first_stage_model.encode(CT1_img)
# # mask_64 = torch.nn.functional.interpolate(PET_mask, size=ct0_64.shape[-2:])
# # cc = mask_64.to(device)

# # c = pet_64
# # x_T = ct1_64
# # # c = torch.cat((c, cc), dim=1) # channel = 4
# # shape = (c.shape[1],)+c.shape[2:]


# # ct0_64 size 64
# # PET_img size 256
# # c will go through cond_stage_model

# # for idz in range(100):
# #     optimizer.zero_grad()
# #     loss, loss_dict = model(
# #         x=ct0_64, 
# #         c=PET_img,
# #         xT=None,
# #     )
# #     # for key in loss_dict.keys():
# #     #     print(key, loss_dict[key], end="")
# #     # print()
# #     loss.backward()
# #     optimizer.step()

# #     print(f"Epoch {idz}, Loss {loss.item()}")

# # # # ----------------------------------------------------

# # # # perform the test

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
        # mask_64 = torch.nn.functional.interpolate(PET_mask, size=ct0_64.shape[-2:])
        
        c = pet_64 / 4
        x_T = ct1_64
        # noise = torch.randn_like(c)
        # c = torch.cat((c, noise), dim=1) # channel = 4
        shape = (c.shape[1],)+c.shape[2:]

        print(f"Before trianing, c is the size {c.shape}, x_T is the size {x_T.shape}")

        samples_ddim, _ = sampler.sample(
            S=opt.steps,
            conditioning=c,
            batch_size=c.shape[0],
            shape=shape,
            verbose=False,
            x_T=x_T
        )
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        # image = torch.clamp((CT0_img+1.0)/2.0, min=0.0, max=1.0)
        # # mask = torch.clamp((PET_mask+1.0)/2.0, min=0.0, max=1.0)
        # predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
        predicted_image = x_samples_ddim.cpu().numpy().transpose(0,2,3,1)[0]
        # inpainted = (1-mask)*image+mask*predicted_image
        # inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]
        savename = root_dir+"xT_test.npy"
        np.save(savename, predicted_image)
        print("The output file is saved to", savename)




# # check input size
# image = images[0]
# mask = masks[0]
# batch = make_batch(image, mask, device=torch.device('cpu'))
# print(batch["image"].size(), batch["mask"].size(), batch["masked_image"].size())
# torch.Size([1, 3, 512, 512]) torch.Size([1, 1, 512, 512]) torch.Size([1, 3, 512, 512])
# images = sorted(glob.glob(os.path.join(opt.indir, "*.jpg")))
# # images = [x.replace("_mask.png", ".png") for x in masks]

# os.makedirs(opt.outdir, exist_ok=True)
# with torch.no_grad():
#     with model.ema_scope():
#         for image in tqdm(images):
#             outpath = os.path.join(opt.outdir, os.path.split(image)[1])
#             print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", outpath)
#             batch = load_image(image, device=device)

#             # encode masked image and concat downsampled mask
#             # c = model.cond_stage_model.encode(batch["image"]) # channel = 3
#             c = model.first_stage_model.encode(batch["image"])
#             savename = "encoded_c.npy"
#             # save the c
#             np.save(savename, c.cpu().numpy())
#             print("encoded_c.npy is saved.")
#             # cc = torch.nn.functional.interpolate(batch["mask"],
#             #                                         size=c.shape[-2:]) # channel = 1
#             # c = torch.cat((c, cc), dim=1) # channel = 4

#             shape = (c.shape[1],)+c.shape[2:]
#             samples_ddim, _ = sampler.sample(
#                 S=opt.steps,
#                 conditioning=c,
#                 batch_size=c.shape[0],
#                 shape=shape,
#                 verbose=False
#             )
#             x_samples_ddim = model.decode_first_stage(samples_ddim)

#             # image = torch.clamp((batch["image"]+1.0)/2.0,
#             #                     min=0.0, max=1.0)
#             # mask = torch.clamp((batch["mask"]+1.0)/2.0,
#             #                     min=0.0, max=1.0)
#             predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
#                                             min=0.0, max=1.0)

#             # inpainted = (1-mask)*image+mask*predicted_image
#             # inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
#             semantic_synthesis = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
#             Image.fromarray(semantic_synthesis.astype(np.uint8)).save(outpath)
# # ----------------------------------------------------