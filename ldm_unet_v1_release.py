gpu_list = ','.join(str(x) for x in [1])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
import os
import glob
import numpy as np
import nibabel as nib

from ldm_unet_v1_release_util import VQModel # step 1 model
from monai.networks.nets import DynUNet # step 2 model
from ldm_unet_v1_release_util import two_segment_scale

def main():
    # here I will use argparse to parse the arguments
    parser = argparse.ArgumentParser(description='Synthetic CT from TOFNAC PET Model')
    parser.add_argument('--root_folder', type=str, default="./B100/ldm_unet_v1_release/", help='The root folder to save the model and log file')
    parser.add_argument('--data_target_folder', type=str, default="./B100/nifti_test/", help='The folder to save the PET files')
    args = parser.parse_args()

    root_folder = args.root_folder
    data_target_folder = args.data_target_folder
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)
    log_file = os.path.join(root_folder, "log.txt")

    MID_PET = 5000
    MIQ_PET = 0.9
    MAX_PET = 20000
    MAX_CT = 3976
    MIN_CT = -1024
    MIN_PET = 0
    RANGE_CT = MAX_CT - MIN_CT
    RANGE_PET = MAX_PET - MIN_PET

    model_step1_params = {
        "VQ_NAME": "f4-noattn",
        "n_embed": 8192,
        "embed_dim": 3,
        "ckpt_path": root_folder+"model_step_1.pth",
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
        "ckpt_path": root_folder+"model_step_2.pth",
    }

    # load step 1 model and step 2 model
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

    model_step_2_pretrained_dict = torch.load(model_step2_params["ckpt_path"], map_location="cpu")
    model_step_2.load_state_dict(model_step_2_pretrained_dict)

    print("Model step 1 loaded from", model_step1_params["ckpt_path"])
    print("Model step 2 loaded from", model_step2_params["ckpt_path"])

    model_step_1.to(device)
    model_step_2.to(device)

    # process the PET files

    PET_file_list = sorted(glob.glob(data_target_folder + "*TOFNAC*.nii.gz"))
    print(f"Detected {len(PET_file_list)} PET files in {data_target_folder}")

    for idx_PET, PET_file_path in enumerate(PET_file_list):
        CT_file_path = PET_file_path.replace("TOFNAC", "CTACIVV")
        # check whether the CT file exists
        if os.path.exists(CT_file_path):
            to_COMPUTE_LOSS = True
            print(f"[{idx_PET}]/[{len(PET_file_list)}] Processing {PET_file_path} with CT {CT_file_path}")
        else:
            to_COMPUTE_LOSS = False
            print(f"[{idx_PET}]/[{len(PET_file_list)}] Processing {PET_file_path} without CT")
        
        # load the PET file
        PET_file = nib.load(PET_file_path)
        PET_data = PET_file.get_fdata()

        len_z = PET_data.shape[2]
        norm_PET_data = np.clip(PET_data, MIN_PET, MAX_PET)
        norm_PET_data = two_segment_scale(norm_PET_data, MIN_PET, MID_PET, MAX_PET, MIQ_PET) # (arr, MIN, MID, MAX, MIQ)
        synthetic_CT_data = np.zeros_like(norm_PET_data)

        for idz in range(len_z):
            
            print(f"Processing [{idx_PET}]/[{len(PET_file_list)}] slice {idz}/{len_z}")
            
            if idz == 0:
                index_list = [idz, idz, idz+1]
            elif idz == len_z-1:
                index_list = [idz-1, idz, idz]
            else:
                index_list = [idz-1, idz, idz+1]
            
            PET_slice = np.squeeze(norm_PET_data[:, :, index_list])
            # (400, 400, 3) -> (3, 400, 400)
            PET_slice = np.transpose(PET_slice, (2, 0, 1))
            PET_slice = np.expand_dims(PET_slice, axis=0)
            PET_slice = torch.from_numpy(PET_slice).float().to(device)

            output_step_1 = model_step_1(PET_slice)
            output_step_2 = model_step_2(output_step_1)
            synthetic_CT_slice = output_step_1 + output_step_2

            synthetic_CT_slice = synthetic_CT_slice.detach().cpu().numpy()
            synthetic_CT_slice = np.squeeze(np.clip(synthetic_CT_slice, 0, 1))
            synthetic_CT_slice = synthetic_CT_slice * RANGE_CT + MIN_CT
            synthetic_CT_data[:, :, idz] = synthetic_CT_slice
        
        # save the synthetic CT data
        synthetic_CT_file = nib.Nifti1Image(synthetic_CT_data, affine=PET_file.affine, header=PET_file.header)
        synthetic_CT_path = PET_file_path.replace("TOFNAC", "SYNTHCT")
        nib.save(synthetic_CT_file, synthetic_CT_path)
        print("Saved to", synthetic_CT_path)

        if to_COMPUTE_LOSS:
            CT_file = nib.load(CT_file_path)
            CT_data = CT_file.get_fdata()
            mask_CT = CT_data > -MIN_CT
            masked_loss = np.mean(np.abs(synthetic_CT_data[mask_CT] - CT_data[mask_CT]))
            print(f"Masked Loss: {masked_loss}")
            with open(log_file, "a") as f:
                f.write(f"{PET_file_path} Masked Loss: {masked_loss}\n")
        else:
            with open(log_file, "a") as f:
                f.write(f"{PET_file_path} No CT file found\n")

if __name__ == "__main__":
    main()
