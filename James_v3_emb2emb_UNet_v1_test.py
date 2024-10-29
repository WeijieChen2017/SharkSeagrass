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

MAX_CT = 2976
MIN_CT = -1024
RANGE_CT = MAX_CT - MIN_CT

import os

root_folder = "James_data_v3/"
fold_cv = 0
save_folder = root_folder + f"James_v3_emb2emb_UNet_v1_cv{fold_cv}_NomaskTrain/"
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

# --------------------------------
# PART: adapter model
# --------------------------------

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

# load the pretrained weights
pretrain_model_path = save_folder + "best_model.pth"
if os.path.exists(pretrain_model_path):
    model.load_state_dict(torch.load(pretrain_model_path))
    print(f"Pretrained model loaded from {pretrain_model_path}")
else:
    print(f"No pretrained model found at {pretrain_model_path}")

# --------------------------------
# PART: decoder model
# --------------------------------

model_params = {
    "vq_weights_path": "f4_vq_weights.npy",
    "VQ_NAME": "f4",
    "n_embed": 8192,
    "embed_dim": 3,
    "img_size" : 256,
    "input_modality" : ["TOFNAC", "CTAC"],
    # "ckpt_path": f"B100/TC256_best_ckpt/best_model_cv{cross_validation}.pth",
    "ckpt_path": f"f4_nnmodel.pth",
    "ddconfig": {
            "double_z": False,
            "z_channels": 3,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
    },
}

from James_v3_token2img_utils import VQModel_decoder

model_decoder = VQModel_decoder(
    ddconfig=model_params["ddconfig"],
    n_embed=model_params["n_embed"],
    embed_dim=model_params["embed_dim"],
)
model_decoder.load_pretrain_weights(model_params["ckpt_path"])
print("Loaded model weights from", model_params["ckpt_path"])
model_decoder.to(device)
model_decoder.eval()

# --------------------------------
# PART: do evaluation
# --------------------------------
import os
import nibabel as nib

from James_v3_emb2emb_UNet_v1_utils import train_or_eval_or_test, VQ_NN_embedings, VQ_NN_embedings_sphere

MAE_all = {
    "axial": {
        "no_VQ": [],
        "VQ_order_one": [],
        "VQ_order_two": [],
        "VQ_sphere": [],
    },
    "coronal": {
        "no_VQ": [],
        "VQ_order_one": [],
        "VQ_order_two": [],
        "VQ_sphere": [],
    },
    "sagittal": {
        "no_VQ": [],
        "VQ_order_one": [],
        "VQ_order_two": [],
        "VQ_sphere": [],
    },
}


for case_name in test_list:

    # load the ground truth
    CTAC_path = root_folder + f"CTACIVV_256_norm/CTACIVV_{case_name}_norm.nii.gz"
    CTAC_file = nib.load(CTAC_path)
    CTAC_data = CTAC_file.get_fdata()
    print(f"CTACIVV_{case_name}_norm.nii.gz loaded, shape: {CTAC_data.shape}, mean: {CTAC_data.mean()}, std: {CTAC_data.std()}")
    gt_x, gt_y, gt_z = CTAC_data.shape

    # load the embeddings if not computed
    axial_no_VQ_path = save_folder + f"axial_no_VQ_{case_name}.npy"
    axial_VQ_order_one_path = save_folder + f"axial_VQ_order_one_{case_name}.npy"
    axial_VQ_order_two_path = save_folder + f"axial_VQ_order_two_{case_name}.npy"
    axial_VQ_sphere_path = save_folder + f"axial_VQ_sphere_{case_name}.npy"

    if os.path.exists(axial_no_VQ_path):
        axial_no_VQ = np.load(axial_no_VQ_path)
        print(f"axial_no_VQ loaded from {axial_no_VQ_path}, shape: {axial_no_VQ.shape}")
    else:

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
        
        # norm method: x_post_quan = x_post_quan / (vq_norm_factor * 2) + 0.5
        # de-norm the embeddings
        axial_no_VQ = (axial_pred_output - 0.5) * (vq_norm_factor * 2)
        np.save(axial_no_VQ_path, axial_no_VQ)
        print(f"axial_no_VQ saved to {axial_no_VQ_path}, shape: {axial_no_VQ.shape}")
    
    # if os.path.exists(axial_VQ_order_one_path):
    #     axial_VQ_order_one = np.load(axial_VQ_order_one_path)
    #     print(f"axial_VQ_order_one loaded from {axial_VQ_order_one_path}, shape: {axial_VQ_order_one.shape}")
    # else:
    #     axial_VQ_order_one = VQ_NN_embedings(vq_weights, axial_no_VQ, dist_order=1)
    #     np.save(axial_VQ_order_one_path, axial_VQ_order_one)
    #     print(f"axial_VQ_order_one saved to {axial_VQ_order_one_path}")

    # if os.path.exists(axial_VQ_order_two_path):
    #     axial_VQ_order_two = np.load(axial_VQ_order_two_path)
    #     print(f"axial_VQ_order_two loaded from {axial_VQ_order_two_path}, shape: {axial_VQ_order_two.shape}")
    # else:
    #     axial_VQ_order_two = VQ_NN_embedings(vq_weights, axial_no_VQ, dist_order=2)
    #     np.save(axial_VQ_order_two_path, axial_VQ_order_two)
    #     print(f"axial_VQ_order_two saved to {axial_VQ_order_two_path}")

    if os.path.exists(axial_VQ_sphere_path):
        axial_VQ_sphere = np.load(axial_VQ_sphere_path)
        print(f"axial_VQ_sphere loaded from {axial_VQ_sphere_path}, shape: {axial_VQ_sphere.shape}")
    else:
        axial_VQ_sphere = VQ_NN_embedings_sphere(vq_weights, axial_no_VQ)
        np.save(axial_VQ_sphere_path, axial_VQ_sphere)
        print(f"axial_VQ_sphere saved to {axial_VQ_sphere_path}")

    len_z = axial_no_VQ.shape[0]
    # recon_axial_no_VQ = np.zeros((gt_x, gt_y, len_z), dtype=np.float32)
    # recon_axial_VQ_order_one = np.zeros((gt_x, gt_y, len_z), dtype=np.float32)
    # recon_axial_VQ_order_two = np.zeros((gt_x, gt_y, len_z), dtype=np.float32)
    recon_axial_VQ_sphere = np.zeros((gt_x, gt_y, len_z), dtype=np.float32)
    for idx_z in range(len_z):
        # recon_axial_no_VQ[:, :, idx_z] = model_decoder(torch.from_numpy(axial_no_VQ[idx_z, :, :, :]).float().unsqueeze(0).to(device)).detach().cpu().numpy()[:, 1, :, :]
        # recon_axial_VQ_order_one[:, :, idx_z] = model_decoder(torch.from_numpy(axial_VQ_order_one[idx_z, :, :, :]).float().unsqueeze(0).to(device)).detach().cpu().numpy()[:, 1, :, :]
        # recon_axial_VQ_order_two[:, :, idx_z] = model_decoder(torch.from_numpy(axial_VQ_order_two[idx_z, :, :, :]).float().unsqueeze(0).to(device)).detach().cpu().numpy()[:, 1, :, :]    
        recon_axial_VQ_sphere[:, :, idx_z] = model_decoder(torch.from_numpy(axial_VQ_sphere[idx_z, :, :, :]).float().unsqueeze(0).to(device)).detach().cpu().numpy()[:, 1, :, :]
    
    # recon_axial_no_VQ = recon_axial_no_VQ[:, :, :gt_z]
    # recon_axial_VQ_order_one = recon_axial_VQ_order_one[:, :, :gt_z]
    # recon_axial_VQ_order_two = recon_axial_VQ_order_two[:, :, :gt_z]
    recon_axial_VQ_sphere = recon_axial_VQ_sphere[:, :, :gt_z]

    # de-norm the reconstructions from -1 -> 1 to 0 -> 1
    # print(f"recon_axial_no_VQ: mean: {recon_axial_no_VQ.mean()}, std: {recon_axial_no_VQ.std()}")
    # print(f"recon_axial_VQ_order_one: mean: {recon_axial_VQ_order_one.mean()}, std: {recon_axial_VQ_order_one.std()}")
    # print(f"recon_axial_VQ_order_two: mean: {recon_axial_VQ_order_two.mean()}, std: {recon_axial_VQ_order_two.std()}")
    print(f"recon_axial_VQ_sphere: mean: {recon_axial_VQ_sphere.mean()}, std: {recon_axial_VQ_sphere.std()}")
    # recon_axial_no_VQ = (recon_axial_no_VQ + 1) / 2
    # recon_axial_VQ_order_one = (recon_axial_VQ_order_one + 1) / 2
    # recon_axial_VQ_order_two = (recon_axial_VQ_order_two + 1) / 2
    recon_axial_VQ_sphere = (recon_axial_VQ_sphere + 1) / 2

    # compute the MAE
    # # MAE_no_VQ = np.mean(np.abs(CTAC_data - recon_axial_no_VQ)) * RANGE_CT
    # # MAE_VQ_order_one = np.mean(np.abs(CTAC_data - recon_axial_VQ_order_one)) * RANGE_CT
    # # MAE_VQ_order_two = np.mean(np.abs(CTAC_data - recon_axial_VQ_order_two)) * RANGE_CT
    MAE_VQ_sphere = np.mean(np.abs(CTAC_data - recon_axial_VQ_sphere)) * RANGE_CT
    print(f"MAE_VQ_sphere: {MAE_VQ_sphere}")
    # print(f"MAE_no_VQ: {MAE_no_VQ}, MAE_VQ_order_one: {MAE_VQ_order_one}, MAE_VQ_order_two: {MAE_VQ_order_two}")
    # MAE_all["axial"]["no_VQ"].append(MAE_no_VQ)
    # MAE_all["axial"]["VQ_order_one"].append(MAE_VQ_order_one)
    # MAE_all["axial"]["VQ_order_two"].append(MAE_VQ_order_two)
    MAE_all["axial"]["VQ_sphere"].append(MAE_VQ_sphere)

    # save the reconstructions
    # denorm_recon_axial_no_VQ = recon_axial_no_VQ * RANGE_CT + MIN_CT
    # denorm_recon_axial_VQ_order_one = recon_axial_VQ_order_one * RANGE_CT + MIN_CT
    # denorm_recon_axial_VQ_order_two = recon_axial_VQ_order_two * RANGE_CT + MIN_CT
    denorm_recon_axial_VQ_sphere = recon_axial_VQ_sphere * RANGE_CT + MIN_CT

    # denorm_recon_axial_no_VQ_nii = nib.Nifti1Image(denorm_recon_axial_no_VQ, CTAC_file.affine, CTAC_file.header)
    # denorm_recon_axial_VQ_order_one_nii = nib.Nifti1Image(denorm_recon_axial_VQ_order_one, CTAC_file.affine, CTAC_file.header)
    # denorm_recon_axial_VQ_order_two_nii = nib.Nifti1Image(denorm_recon_axial_VQ_order_two, CTAC_file.affine, CTAC_file.header)
    denorm_recon_axial_VQ_sphere_nii = nib.Nifti1Image(denorm_recon_axial_VQ_sphere, CTAC_file.affine, CTAC_file.header)

    # denorm_recon_axial_no_VQ_path = save_folder + f"denorm_recon_axial_no_VQ_{case_name}.nii.gz"
    # denorm_recon_axial_VQ_order_one_path = save_folder + f"denorm_recon_axial_VQ_order_one_{case_name}.nii.gz"
    # denorm_recon_axial_VQ_order_two_path = save_folder + f"denorm_recon_axial_VQ_order_two_{case_name}.nii.gz"
    denorm_recon_axial_VQ_sphere_path = save_folder + f"denorm_recon_axial_VQ_sphere_{case_name}.nii.gz"    

    # nib.save(denorm_recon_axial_no_VQ_nii, denorm_recon_axial_no_VQ_path)
    # nib.save(denorm_recon_axial_VQ_order_one_nii, denorm_recon_axial_VQ_order_one_path)
    # nib.save(denorm_recon_axial_VQ_order_two_nii, denorm_recon_axial_VQ_order_two_path)
    nib.save(denorm_recon_axial_VQ_sphere_nii, denorm_recon_axial_VQ_sphere_path)
    # print(f"denorm_recon_axial_no_VQ saved to {denorm_recon_axial_no_VQ_path}")
    # print(f"denorm_recon_axial_VQ_order_one saved to {denorm_recon_axial_VQ_order_one_path}")
    # print(f"denorm_recon_axial_VQ_order_two saved to {denorm_recon_axial_VQ_order_two_path}")
    print(f"denorm_recon_axial_VQ_sphere saved to {denorm_recon_axial_VQ_sphere_path}")

