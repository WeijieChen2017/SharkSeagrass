import os
import nibabel as nib

# import wandb
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device is: ", device)

from James_v3_token2img_utils import VQModel_decoder

import argparse
import json
import time
import random
import numpy as np

# from UNetUNet_v1_py2_train_acs_util import VQModel, simple_logger, prepare_dataset

data_params = {
    "zoom": 4,
}

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

root_folder = f"James_data_v3/"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
print("The root folder is: ", root_folder)

model = VQModel_decoder(
    ddconfig=model_params["ddconfig"],
    n_embed=model_params["n_embed"],
    embed_dim=model_params["embed_dim"],
)
model.load_pretrain_weights(model_params["ckpt_path"])
print("Loaded model weights from", model_params["ckpt_path"])
model.to(device)
model.eval()

# load vq_weights
vq_weights = np.load(model_params["vq_weights_path"])
print(f"Loading vq weights from {model_params['vq_weights_path']}, shape: {vq_weights.shape}")

target_case_name_list = ["E4055"]
############################################################################################################

for idx_case, case_name in enumerate(target_case_name_list):
    path_x_axial = root_folder + f"index/{case_name}_x_axial_ind.npy"
    path_y_axial = root_folder + f"index/{case_name}_y_axial_ind.npy"
    path_x_PET = root_folder + f"TOFNAC_256_norm/TOFNAC_{case_name}_norm.nii.gz"
    path_y_CT = root_folder + f"CTACIVV_256_norm/CTACIVV_{case_name}_norm.nii.gz"

    file_x_axial = np.load(path_x_axial)
    file_y_axial = np.load(path_y_axial)

    file_x_PET = nib.load(path_x_PET)
    file_x_PET_data = file_x_PET.get_fdata()
    file_y_CT = nib.load(path_y_CT)
    file_y_CT_data = file_y_CT.get_fdata()

    print(f"Processing {case_name}:[{idx_case+1}]/[{len(target_case_name_list)}]")
    print(f"file_x_axial shape: {file_x_axial.shape}, file_y_axial shape: {file_y_axial.shape}")
    print(f"file_x_PET shape: {file_x_PET_data.shape}")
    print(f"file_y_CT shape: {file_y_CT_data.shape}")

    len_z = file_x_axial.shape[0] # padded
    len_x_len_y = file_x_axial.shape[1] # x*y
    len_x = int(np.sqrt(len_x_len_y))
    len_y = len_x

    recon_x_axial = np.zeros((file_x_PET_data.shape[0], file_x_PET_data.shape[1], len_z), dtype=np.float32)
    recon_y_axial = np.zeros((file_y_CT_data.shape[0], file_y_CT_data.shape[1], len_z), dtype=np.float32)

    for i in range(len_z):
        
        print(f"Processing {case_name}:[{idx_case+1}]/[{len(target_case_name_list)}] -> {i+1}/{len_z}")
        x_axial_ind = file_x_axial[i, :]
        y_axial_ind = file_y_axial[i, :]

        x_axial_post_quan = vq_weights[x_axial_ind.astype(int)].reshape(len_x, len_y, 3)
        y_axial_post_quan = vq_weights[y_axial_ind.astype(int)].reshape(len_x, len_y, 3)

        x_axial_post_quan = torch.from_numpy(x_axial_post_quan).float().to(device)
        x_axial_post_quan = x_axial_post_quan.unsqueeze(0)
        x_axial_post_quan = x_axial_post_quan.permute(0, 3, 1, 2)

        y_axial_post_quan = torch.from_numpy(y_axial_post_quan).float().to(device)
        y_axial_post_quan = y_axial_post_quan.unsqueeze(0)
        y_axial_post_quan = y_axial_post_quan.permute(0, 3, 1, 2)

        # print("x_axial_post_quan shape: ", x_axial_post_quan.shape)
        # print("y_axial_post_quan shape: ", y_axial_post_quan.shape)

        recon_x_axial_slice = model(x_axial_post_quan).detach().cpu().numpy().squeeze()
        recon_y_axial_slice = model(y_axial_post_quan).detach().cpu().numpy().squeeze()

        # print("recon_x_axial_slice shape: ", recon_x_axial_slice.shape)
        # print("recon_y_axial_slice shape: ", recon_y_axial_slice.shape)
        
        recon_x_axial[:, :, i] = recon_x_axial_slice[1, :, :]
        recon_y_axial[:, :, i] = recon_y_axial_slice[1, :, :]

    # cut the padded part in z axis
    len_z_nonpad = file_y_CT_data.shape[2]
    recon_x_axial = recon_x_axial[:, :, :len_z_nonpad]
    recon_y_axial = recon_y_axial[:, :, :len_z_nonpad]

    # convert from -1 -> 1 to 0 -> 1
    recon_x_axial = (recon_x_axial + 1) / 2
    recon_y_axial = (recon_y_axial + 1) / 2

    # save the reconstructed images
    recon_x_axial_nii = nib.Nifti1Image(recon_x_axial, affine=file_x_PET.affine, header=file_x_PET.header)
    recon_y_axial_nii = nib.Nifti1Image(recon_y_axial, affine=file_y_CT.affine, header=file_y_CT.header)
    recon_x_axial_path = root_folder + f"{case_name}_vqrecon_x_axial.nii.gz"
    recon_y_axial_path = root_folder + f"{case_name}_vqrecon_y_axial.nii.gz"
    nib.save(recon_x_axial_nii, recon_x_axial_path)
    nib.save(recon_y_axial_nii, recon_y_axial_path)




    # temp_img_size = (256, 256, 720)
    # direction = "axial"
    # len_axial = temp_img_size[2]
    # pred_volume = np.zeros(temp_img_size, dtype=np.float32)
    # for i in range(len_axial):
    #     index = len_axial - i - 1
    #     pred_ind_path = f"{root_folder}/{case_name}_{direction}_pred_ind{index:03d}.npy"
    #     pred_ind = np.load(pred_ind_path)            
    #     pred_ind = pred_ind.reshape((32, 32))
    #     # load each 
    #     pred_post_quan = vq_weights[pred_ind.astype(int)].reshape(32, 32, 4)
    #     # convert to tensor
    #     pred_post_quan = torch.from_numpy(pred_post_quan).float().to(device)
    #     pred_post_quan = pred_post_quan.unsqueeze(0)
    #     # change dim from 1, 32, 32, 4 to 1, 4, 32, 32
    #     pred_post_quan = pred_post_quan.permute(0, 3, 1, 2)
    #     pred_img = model(pred_post_quan)
    #     pred_img = pred_img.detach().cpu().numpy().squeeze()
    #     pred_img_denorm = (pred_img[1, :, :] + 1.0) / 2 # 0 -> 1
    #     pred_img_denorm = pred_img_denorm * 5000 - 1024
    #     pred_volume[:, :, index] = pred_img_denorm
    #     print("Processing", case_name, direction, index)
    
    # nii_filepath = "TC256_v2/NKQ091_CTAC_256_corrected.nii.gz"
    # nii_file = nib.load(nii_filepath)
    # pred_nii = nib.Nifti1Image(pred_volume, affine=nii_file.affine, header=nii_file.header)
    # pred_nii_path = f"{root_folder}/{case_name}_pred.nii.gz"
    # nib.save(pred_nii, pred_nii_path)
    # exit()
