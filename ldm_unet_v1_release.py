gpu_list = ','.join(str(x) for x in [1])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from ldm_unet_v1_release_util import VQModel # step 1 model
from monai.networks.nets import DynUNet # step 2 model

root_folder = "./B100/ldm_unet_v1_release"
data_target_folder = "./B100/nifti_tet/"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
print("The root folder is: ", root_folder)
log_file = os.path.join(root_folder, "log.txt")

model_step1_params = {
    "VQ_NAME": "f4-noattn",
    "n_embed": 8192,
    "embed_dim": 3,
    "ckpt_path": root_folder+"model_step1.pth",
    "ddconfig": {
        "attn_type": "none",
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
    }
}

model_step2_params = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "kernels": [[3, 3], [3, 3], [3, 3], [3, 3]],
    "strides": [[1, 1], [2, 2], [2, 2], [2, 2]],
    "filters": (64, 128, 256, 512),
    "dropout": 0.0,
    "norm_name": ('INSTANCE', {'affine': True}),
    "act_name": ('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
    "deep_supervision": True,
    "deep_supr_num": 1,
    "res_block": True,
    "trans_bias": False,
    "ckpt_path": root_folder+"model_step2.pth",
}

model_step_1 = VQModel(
    ddconfig=model_step1_params["ddconfig"],
    n_embed=model_step1_params["n_embed"],
    embed_dim=model_step1_params["embed_dim"],
    ckpt_path=model_step1_params["ckpt_path"],
    ignore_keys=[],
    image_key="image",
)

model_step_2 = DynUNet(
    spatial_dims=model_step2_params["spatial_dims"],
    in_channels=model_step2_params["in_channels"],
    out_channels=model_step2_params["out_channels"],
    kernel_size=model_step2_params["kernels"],
    strides=model_step2_params["strides"],
    upsample_kernel_size=model_step2_params["strides"][1:],
    filters=model_step2_params["filters"],
    dropout=model_step2_params["dropout"],
    norm_name=model_step2_params["norm_name"],
    act_name=model_step2_params["act_name"],
    deep_supervision=model_step2_params["deep_supervision"],
    deep_supr_num=model_step2_params["deep_supr_num"],
    res_block=model_step2_params["res_block"],
    trans_bias=model_step2_params["trans_bias"],
)

model_step_2_pretrained_dict = torch.load(model_step2_params["ckpt_path"])
model_step_2.load_state_dict(model_step_2_pretrained_dict)

print("Model step 1 loaded from", model_step1_params["ckpt_path"])
print("Model step 2 loaded from", model_step2_params["ckpt_path"])

