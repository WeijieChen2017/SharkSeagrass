# gpu_list = ','.join(str(x) for x in [1])
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
# print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
# import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
import os
import json
import torch
import glob
import time
import numpy as np
import nibabel as nib

from ldm_unet_v1_release_util import VQModel # step 1 model
from monai.networks.nets import DynUNet # step 2 model
from ldm_unet_v1_release_util import two_segment_scale

from monai.inferers import sliding_window_inference

def main():
    # here I will use argparse to parse the arguments
    parser = argparse.ArgumentParser(description='Synthetic CT from TOFNAC PET Model')
    parser.add_argument('--root_folder', type=str, default="./B100/ldm_unet_v1_release_3d/", help='The root folder to save the model and log file')
    parser.add_argument('--data_div_json', type=str, default="./B100/step1step2_0822_vanila.json", help='The folder to save the data division files')
    parser.add_argument('--mode', type=str, default="d3f64", help='The mode of the model, train or test')
    args = parser.parse_args()

    root_folder = args.root_folder
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print("The root folder is: ", root_folder)
    log_file = os.path.join(root_folder, "log.txt")
    with open(log_file, "w") as f:
        f.write("\n")
    device = torch.device('cuda:1')

    MID_PET = 5000
    MIQ_PET = 0.9
    MAX_PET = 20000
    MAX_CT = 1976
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

    if args.mode == "d4f32":
        kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        filters = (32, 64, 128, 256)
        # device = torch.device("cuda:1")
    elif args.mode == "d3f64":
        kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2]]
        filters = (64, 128, 256)
        # device = torch.device("cuda:0")

    model_step2_params = {
        "cube_size": 128,
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "kernels": kernels,
        "strides": strides,
        "filters": filters,
        "upsample_kernel_size": strides[1:],
        "dropout": 0.0,
        "norm_name": ('INSTANCE', {'affine': True}),
        "act_name": ('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
        "deep_supervision": True,
        "deep_supr_num": 1,
        "res_block": True,
        "trans_bias": False,
        "ckpt_path":f"./B100/dynunet3d_v2_step2_pretrain_{args.mode}_continue_wloss_iceEnc_res/best_model.pth",
    }

    # load step 1 model and step 2 model
    # model_step_1 = VQModel(
    #     ddconfig=model_step1_params["ddconfig"],
    #     n_embed=model_step1_params["n_embed"],
    #     embed_dim=model_step1_params["embed_dim"],
    #     ckpt_path=model_step1_params["ckpt_path"],
    #     ignore_keys=[],
    #     image_key="image",
    # )

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

    # print("Model step 1 loaded from", model_step1_params["ckpt_path"])
    print("Model step 2 loaded from", model_step2_params["ckpt_path"])

    # model_step_1.to(device)
    model_step_2.to(device)

    # model_step_1.eval()
    model_step_2.eval()

    # process the PET files

    with open(args.data_div_json, "r") as f:
        data_div = json.load(f)
    train_list = data_div["train"]
    val_list = data_div["val"]
    test_list = data_div["test"]

    num_train = len(train_list)
    num_val = len(val_list)
    num_test = len(test_list)

    print(f"Detected {num_train} training files")
    print(f"Detected {num_val} validation files")
    print(f"Detected {num_test} testing files")

    for data_list in [train_list, val_list, test_list]:

        for idx_PET, pair in enumerate(data_list):
            
            tik = time.time()

            step1_path = pair["STEP1"]
            step2_path = pair["STEP2"]
            # check whether the CT file exists
            if os.path.exists(step2_path):
                to_COMPUTE_LOSS = True
                print(f"[{idx_PET+1}]/[{len(data_list)}] Processing {step1_path} with CT {step2_path}")
            else:
                to_COMPUTE_LOSS = False
                print(f"[{idx_PET+1}]/[{len(data_list)}] Processing {step1_path} without CT")
            
            # load the PET file
            step1_file = nib.load(step1_path)
            step1_data = step1_file.get_fdata()

            step1_data = np.clip(step1_data, MIN_CT, MAX_CT)
            # step1_data = two_segment_scale(step1_data, MIN_PET, MID_PET, MAX_PET, MIQ_PET) # (arr, MIN, MID, MAX, MIQ)
            step1_data = (step1_data - MIN_CT) / RANGE_CT # 0 to 1
            norm_step1_data = step1_data * 2 - 1 # -1 to 1
            
            # now it is using slide_window to process the 3d data
            # synthetic_CT_data_step_1 # 400, 400, z
            # convert to 1, 1, 400, 400, z
            norm_step1_data = np.expand_dims(np.expand_dims(norm_step1_data, axis=0), axis=0)
            norm_step1_data = torch.from_numpy(norm_step1_data).float().to(device)
            synthetic_step2_data = sliding_window_inference(
                inputs = norm_step1_data, 
                roi_size = model_step2_params["cube_size"],
                sw_batch_size = 1,
                predictor = model_step_2,
                overlap=0.25, 
                mode="gaussian", 
                sigma_scale=0.125, 
                padding_mode="constant", 
                cval=0.0,
                device=device,
            ) # f(x) -> y-x
            synthetic_CT_data = norm_step1_data + synthetic_step2_data # -1 to 1
            synthetic_CT_data = (synthetic_CT_data + 1) / 2 # 0 to 1
            synthetic_CT_data = synthetic_CT_data.squeeze().detach().cpu().numpy() # 400, 400, z
            synthetic_CT_data = synthetic_CT_data * RANGE_CT + MIN_CT # MIN_CT to MAX_CT
            
            # save the synthetic CT data
            synthetic_CT_file = nib.Nifti1Image(synthetic_CT_data, affine=step1_file.affine, header=step1_file.header)
            synthetic_CT_path = step1_path.replace("STEP1", "STEP3")
            nib.save(synthetic_CT_file, synthetic_CT_path)
            print("Saved to", synthetic_CT_path)

            if to_COMPUTE_LOSS:
                CT_file = nib.load(step2_path)
                CT_data = CT_file.get_fdata() # 467, 467, z
                CT_data = CT_data[33:433, 33:433, :] # 400, 400, z
                mask_CT = CT_data > -MIN_CT
                masked_loss = np.mean(np.abs(synthetic_CT_data[mask_CT] - CT_data[mask_CT]))
                print(f"Masked Loss: {masked_loss}")
                with open(log_file, "a") as f:
                    f.write(f"{step1_path} Masked Loss: {masked_loss}\n")
            else:
                with open(log_file, "a") as f:
                    f.write(f"{step1_path} No CT file found\n")

            # save the step 1
            # synthetic_CT_file_step_1 = nib.Nifti1Image(synthetic_CT_data_step_1, affine=step1_file.affine, header=step1_file.header)
            # synthetic_CT_path_step_1 = step1_file_path.replace("TOFNAC", "SYNTHCT_STEP1")
            # nib.save(synthetic_CT_file_step_1, synthetic_CT_path_step_1)
            # print("Saved to", synthetic_CT_path_step_1)

            # # compute the loss between the synthetic CT and the real CT
            # if to_COMPUTE_LOSS:
            #     masked_loss = np.mean(np.abs(synthetic_CT_data_step_1[mask_CT] - CT_data[mask_CT]))
            #     print(f"Masked Loss after step 1: {masked_loss}")
            #     with open(log_file, "a") as f:
            #         f.write(f"{step1_file_path} Masked Loss after step 1: {masked_loss}\n")
            # else:
            #     with open(log_file, "a") as f:
            #         f.write(f"{step1_file_path} No CT file found\n")

            tok = time.time()
            print(f"Time elapsed: {tok-tik:.2f} seconds")
            with open(log_file, "a") as f:
                f.write(f"Time elapsed: {tok-tik:.2f} seconds\n")

if __name__ == "__main__":
    main()
