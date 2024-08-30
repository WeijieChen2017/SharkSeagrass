SLICE_FOLDER = "./B100/f4noattn_step1/"
TOFNAC_FOLDER = "./B100/TOFNAC_resample/"
CTACIVV_FOLDER = "./B100/CTACIVV_resample/"
STEP1_VOLUME_FOLDER = "./B100/f4noattn_step1_volume/"
RECON_SLICE_FOLDER = "./B100/f4noattn_step1_recon/"
TOKEN_SLICE_FOLDER = "./B100/f4noattn_step1_token/"
VQ_NAME = "f4-noattn"


MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MAX_CT = 3976
MIN_CT = -1024
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET


import numpy as np
import nibabel as nib
import torch
import glob
import yaml
import os


from ldm_unet_v1_utils import VQModel

os.makedirs(STEP1_VOLUME_FOLDER, exist_ok=True)

def two_segment_scale(arr, MIN, MID, MAX, MIQ):
    # Create an empty array to hold the scaled results
    scaled_arr = np.zeros_like(arr, dtype=np.float32)

    # First segment: where arr <= MID
    mask1 = arr <= MID
    scaled_arr[mask1] = (arr[mask1] - MIN) / (MID - MIN) * MIQ

    # Second segment: where arr > MID
    mask2 = arr > MID
    scaled_arr[mask2] = MIQ + (arr[mask2] - MID) / (MAX - MID) * (1 - MIQ)
    
    return scaled_arr


def reverse_two_segment_scale(arr, MIN, MID, MAX, MIQ):
    # Create an empty array to hold the reverse scaled results
    reverse_scaled_arr = np.zeros_like(arr, dtype=np.float32)

    # First segment: where arr <= MIQ
    mask1 = arr <= MIQ
    reverse_scaled_arr[mask1] = arr[mask1] * (MID - MIN) / MIQ + MIN

    # Second segment: where arr > MIQ
    mask2 = arr > MIQ
    reverse_scaled_arr[mask2] = MID + (arr[mask2] - MIQ) * (MAX - MID) / (1 - MIQ)
    
    return reverse_scaled_arr


TOFNAC_list = sorted(glob.glob(TOFNAC_FOLDER+"*.nii.gz"))
print("Found", len(TOFNAC_list), "TOFNAC files")

config_yaml_path = f"ldm_models/first_stage_models/vq-{VQ_NAME}/config.yaml"
with open(config_yaml_path, 'r') as file:
    config = yaml.safe_load(file)

print(config)

ckpt_path = f"vq_{VQ_NAME}.ckpt"

dd_config = config['model']['params']['ddconfig']
loss_config = config['model']['params']['lossconfig']

model = VQModel(ddconfig=dd_config,
                lossconfig=loss_config,
                n_embed=config['model']['params']['n_embed'],
                embed_dim=config['model']['params']['embed_dim'],
                ckpt_path=ckpt_path,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                batch_resize_range=None,
                scheduler_config=None,
                lr_g_factor=1.0,
                remap=None,
                sane_index_shape=False, # tell vector quantizer to return indices as bhw
)

print("<" * 50)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("The current device is", device)
model.to(device)

def load_slices(slice_folder, modality_tag, TOFNAC_tag, indices):
    loaded_slices = []
    for idz in indices:
        synCT_path = os.path.join(slice_folder, f"{modality_tag}_{TOFNAC_tag}_z{idz}.npy")
        if os.path.exists(synCT_path):
            synCT_slice = np.load(synCT_path)
            loaded_slices.append(synCT_slice)
        else:
            print(">>> File not found:", synCT_path)
    return np.asarray(loaded_slices)
    

for idx, TOFNAC_path in enumerate(TOFNAC_list):
    TOFNAC_tag = TOFNAC_path.split('/')[-1].split('.')[0][-5:]
    print(f"Processing [{idx+1}]/[{len(TOFNAC_list)}] {TOFNAC_path} TOFNAC tag is {TOFNAC_tag}")
    TOFNAC_file = nib.load(TOFNAC_path)
    TOFNAC_data = TOFNAC_file.get_fdata()
    len_z = TOFNAC_data.shape[2]
    synCT_data = np.zeros_like(TOFNAC_data)
    gt_data = np.zeros_like(TOFNAC_data)
    indices_diff_count = 0

    for idz in range(len_z):
        if idz == 0:
            index_list = [0, 0, 1]
        elif idz == len_z - 1:
            index_list = [len_z - 2, len_z - 1, len_z - 1]
        else:
            index_list = [idz - 1, idz, idz + 1]

        synCT_step1_slices = load_slices(SLICE_FOLDER, "STEP1", TOFNAC_tag, index_list)
        synCT_step2_slices = load_slices(SLICE_FOLDER, "STEP2", TOFNAC_tag, index_list)
        
        synCT_step1_slices = np.clip(synCT_step1_slices, 0, 1)
        synCT_step2_slices = np.clip(synCT_step2_slices, 0, 1)

        # from 0 to 1 to -1 to 1
        synCT_step1_slices = synCT_step1_slices * 2 - 1
        synCT_step2_slices = synCT_step2_slices * 2 - 1
        # print(">>> synCT_step1_slices shape:", synCT_step1_slices.shape)
        # print(">>> synCT_step2_slices shape:", synCT_step2_slices.shape)
        # >>> synCT_step1_slices shape: (3, 1, 400, 400)
        # >>> synCT_step2_slices shape: (3, 400, 400)
        synCT_step1_slices = np.squeeze(synCT_step1_slices)

        # from 400, 400, 3 to 1, 3, 400, 400
        synCT_step1_slices = np.transpose(synCT_step1_slices, (2, 0, 1))
        synCT_step2_slices = np.transpose(synCT_step2_slices, (2, 0, 1))
        synCT_step1_slices = np.expand_dims(synCT_step1_slices, axis=0)
        synCT_step2_slices = np.expand_dims(synCT_step2_slices, axis=0)

        # convert to torch tensor
        synCT_step1_slices = torch.from_numpy(synCT_step1_slices).float().to(device)
        synCT_step2_slices = torch.from_numpy(synCT_step2_slices).float().to(device)

        # forward pass
        recon_synCT_step1_slices, _, ind_synCT_step1_slices = model(synCT_step1_slices, return_pred_indices=True)
        recon_synCT_step2_slices, _, ind_synCT_step2_slices = model(synCT_step2_slices, return_pred_indices=True)

        # detach the tensor and convert to numpy
        recon_synCT_step1_slices = recon_synCT_step1_slices.detach().cpu().numpy()
        recon_synCT_step2_slices = recon_synCT_step2_slices.detach().cpu().numpy()
        ind_synCT_step1_slices = ind_synCT_step1_slices.detach().cpu().numpy()
        ind_synCT_step2_slices = ind_synCT_step2_slices.detach().cpu().numpy()

        # save the recon
        recon_synCT_step1_slices = np.squeeze(recon_synCT_step1_slices)
        recon_synCT_step2_slices = np.squeeze(recon_synCT_step2_slices)
        recon_synCT_step1_slices_name = os.path.join(RECON_SLICE_FOLDER, f"RECON_STEP1_{TOFNAC_tag}_z{idz}.npy")
        recon_synCT_step2_slices_name = os.path.join(RECON_SLICE_FOLDER, f"RECON_STEP2_{TOFNAC_tag}_z{idz}.npy")
        np.save(recon_synCT_step1_slices_name, recon_synCT_step1_slices)
        np.save(recon_synCT_step2_slices_name, recon_synCT_step2_slices)
        print(">>> Saved to", recon_synCT_step1_slices_name)
        print(">>> Saved to", recon_synCT_step2_slices_name)

        # save the indices
        ind_synCT_step1_slices = np.squeeze(ind_synCT_step1_slices)
        ind_synCT_step2_slices = np.squeeze(ind_synCT_step2_slices)
        ind_synCT_step1_slices_name = os.path.join(TOKEN_SLICE_FOLDER, f"TOKEN_STEP1_{TOFNAC_tag}_z{idz}.npy")
        ind_synCT_step2_slices_name = os.path.join(TOKEN_SLICE_FOLDER, f"TOKEN_STEP2_{TOFNAC_tag}_z{idz}.npy")
        np.save(ind_synCT_step1_slices_name, ind_synCT_step1_slices)
        np.save(ind_synCT_step2_slices_name, ind_synCT_step2_slices)
        print(">>> Saved to", ind_synCT_step1_slices_name)
        print(">>> Saved to", ind_synCT_step2_slices_name)

        # compute how many indices are different
        diff_count = np.sum(ind_synCT_step1_slices != ind_synCT_step2_slices)
        print(f"Diff count: {diff_count}")
        with open("diff_count.txt", "a") as f:
            f.write(f"{TOFNAC_tag} z{idz} Diff count: {diff_count}\n")
        indices_diff_count += diff_count

    indices_diff_count /= len_z
    print(f"Average diff count: {indices_diff_count}")
    with open("diff_count.txt", "a") as f:
        f.write(f"{TOFNAC_tag} Average diff count: {indices_diff_count}\n")
        

        