# model selection
# for CNN

# DenseNet
# EfficientNet
# SegResNet
# SENet
# DynUNet
# AttentionUnet
# UNETR
# SwinUNETR
# FullyConnectedNet

# --------------------------------
# PART: data division 
# --------------------------------
import os

root_folder = "James_data_v3/"
fold_cv = 0
save_folder = root_folder + f"James_v3_emb2emb_UNet_v1_cv{fold_cv}_maskTrain/"
json_config =save_folder + "config.json"
import json

config = json.load(open(json_config, "r"))
print("Loaded config from ", json_config)
for key in config:
    print(key, ":", config[key])

fold_cv_train = config["fold_cv_train"]
fold_cv_val = config["fold_cv_val"]
fold_cv_test = config["fold_cv_test"]


data_division = json.load(open(root_folder + "cv_list.json", "r"))

train_list = [data_division[f"cv{fold_cv_x}"] for fold_cv_x in fold_cv_train]
val_list = [data_division[f"cv{fold_cv_x}"] for fold_cv_x in fold_cv_val]
test_list = [data_division[f"cv{fold_cv_x}"] for fold_cv_x in fold_cv_test]

train_list = [item for sublist in train_list for item in sublist]
val_list = [item for sublist in val_list for item in sublist]
test_list = [item for sublist in test_list for item in sublist]

print("train_list: ", train_list)
print("val_list: ", val_list)
print("test_list: ", test_list)

# --------------------------------
# PART: model building
# --------------------------------

import torch
import numpy as np
import random

random_seed = config["random_seed"]
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


vq_norm_factor = config["vq_norm_factor"]
zoom_factor = config["zoom_factor"]
batch_size = config["batch_size"]
dim = config["dim"]
in_channel = config["in_channel"]
out_channel = config["out_channel"]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("The current device is: ", device)

from monai.networks.nets import UNet

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
    num_res_units=6,
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = torch.nn.MSELoss()

vq_weights_path = "f4_vq_weights.npy"
vq_weights = np.load(vq_weights_path)
print(f"Loading vq weights from {vq_weights_path}, shape: {vq_weights.shape}")

# --------------------------------
# PART: start training
# --------------------------------


# load the pretrained weights
pretrain_model_path = save_folder + "best_model.pth"
if os.path.exists(pretrain_model_path):
    model.load_state_dict(torch.load(pretrain_model_path))
    print(f"Pretrained model loaded from {pretrain_model_path}")
else:
    print(f"No pretrained model found at {pretrain_model_path}")

import os
os.makedirs(save_folder, exist_ok=True)

from James_v3_emb2emb_UNet_v1_utils import train_or_eval_or_test

axial_emb_loss = 0.0
coronal_emb_loss = 0.0
sagittal_emb_loss = 0.0

for case_name in train_list:
    axial_loss, axial_pred_output = train_or_eval_or_test(
        model=model, 
        optimizer=None, 
        loss=loss,
        case_name=case_name,
        stage="test",
        anatomical_plane="axial",
        device=device,
        vq_weights=vq_weights,
        config=config)
    axial_emb_loss += axial_loss
    print(f"case_name: {case_name}, axial_loss: {axial_loss}, axial_pred_output: {axial_pred_output.shape}")
    exit()

    coronal_loss, coronal_pred_output = train_or_eval_or_test(
        model=model, 
        optimizer=None, 
        loss=loss,
        case_name=case_name,
        stage="test",
        anatomical_plane="coronal",
        device=device,
        vq_weights=vq_weights,
        config=config)
    coronal_emb_loss += coronal_loss


    sagittal_loss, sagittal_pred_output = train_or_eval_or_test(
        model=model, 
        optimizer=None, 
        loss=loss,
        case_name=case_name,
        stage="test",
        anatomical_plane="sagittal",
        device=device,
        vq_weights=vq_weights,
        config=config)
    sagittal_emb_loss += sagittal_loss
