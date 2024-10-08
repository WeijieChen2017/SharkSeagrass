import json
import glob
import numpy as np
import os

import nibabel as nib
import matplotlib.pyplot as plt

data_div_to_plot = "train"



def plot_case_from_view_cut(x_data, y_data, z_data, save_name, num_cut, cut_view, index_list):

    y_mask = y_data > -500
    mask_ratio = np.sum(y_mask) / y_mask.size
    mae = np.mean(np.abs(y_data[y_mask] - z_data[y_mask]))
    unmask_mae = np.mean(np.abs(y_data - z_data))
    
    # build index list for cut
    if cut_view == "axial": 
        len_axis = x_data.shape[2]
        row_len_factor = 1.5
        start_plot_axis_q = 25
        end_plot_axis_q = 75  
    elif cut_view == "sagittal":
        len_axis = x_data.shape[0]
        row_len_factor = 3
        start_plot_axis_q = 30
        end_plot_axis_q = 70  
    elif cut_view == "coronal":
        len_axis = x_data.shape[1]
        row_len_factor = 3
        start_plot_axis_q = 35
        end_plot_axis_q = 65
    else:
        raise ValueError("cut_view must be either axial, sagittal, or coronal")
    if index_list is None:
        start_plot_axis_index = int(start_plot_axis_q / 100 * len_axis)
        end_plot_axis_index = int(end_plot_axis_q / 100 * len_axis)
        cut_index_list = np.linspace(start_plot_axis_index, end_plot_axis_index, num_cut, dtype=int)
        # cut_index_list = [len_axis // (num_cut + 1) * (i + 1) for i in range(num_cut)]
    else:
        cut_index_list = index_list
    
    n_col = 6
    n_row = len(cut_index_list)

    fig = plt.figure(figsize=(12, n_row*row_len_factor), dpi=300)
    # super title
    fig.suptitle(f"Test case: {case_name} in {cut_view} view MAE = {mae:.2f} HU", fontsize=16)
    for idx_cut in range(num_cut):

        if cut_view == "axial":
            x_img = x_data[:, :, cut_index_list[idx_cut]]
            y_img = y_data[:, :, cut_index_list[idx_cut]]
            z_img = z_data[:, :, cut_index_list[idx_cut]]
            x_img = np.rot90(x_img, 3)
            y_img = np.rot90(y_img, 3)
            z_img = np.rot90(z_img, 3)
        elif cut_view == "sagittal":
            x_img = x_data[cut_index_list[idx_cut], :, :]
            y_img = y_data[cut_index_list[idx_cut], :, :]
            z_img = z_data[cut_index_list[idx_cut], :, :]
            x_img = np.rot90(x_img, 1)
            y_img = np.rot90(y_img, 1)
            z_img = np.rot90(z_img, 1)
            # horizontal flip
            x_img = np.fliplr(x_img)
            y_img = np.fliplr(y_img)
            z_img = np.fliplr(z_img)
        elif cut_view == "coronal":
            x_img = x_data[:, cut_index_list[idx_cut], :]
            y_img = y_data[:, cut_index_list[idx_cut], :]
            z_img = z_data[:, cut_index_list[idx_cut], :]
            x_img = np.rot90(x_img, 1)
            y_img = np.rot90(y_img, 1)
            z_img = np.rot90(z_img, 1)
        else:
            raise ValueError("cut_view must be either axial, sagittal, or coronal")
        
        # norm to 0-1
        x_img = (x_img - MIN_PET) / (MAX_PET - MIN_PET)
        y_img = (y_img - MIN_CT) / (MAX_CT - MIN_CT)
        z_img = (z_img - MIN_CT) / (MAX_CT - MIN_CT)

        plt.subplot(n_row, n_col, idx_cut * n_col + 1)
        plt.imshow(x_img, cmap="gray", vmin=0, vmax=1.)
        plt.title("TOFNAC") if idx_cut == 0 else None
        plt.axis("off")

        plt.subplot(n_row, n_col, idx_cut * n_col + 2)
        plt.imshow(y_img, cmap="gray", vmin=0, vmax=1.)
        plt.title("CTAC") if idx_cut == 0 else None
        plt.axis("off")

        plt.subplot(n_row, n_col, idx_cut * n_col + 3)
        plt.imshow(z_img, cmap="gray", vmin=0, vmax=1.)
        plt.title("PRED") if idx_cut == 0 else None
        plt.axis("off")

        plt.subplot(n_row, n_col, idx_cut * n_col + 4)
        plt.hist(x_img.flatten(), bins=100)
        plt.title("TOFNAC") if idx_cut == 0 else None
        plt.yscale("log")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, idx_cut * n_col + 5)
        plt.hist(y_img.flatten(), bins=100)
        plt.title("CTAC") if idx_cut == 0 else None
        plt.yscale("log")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, idx_cut * n_col + 6)
        plt.hist(z_img.flatten(), bins=100)
        plt.title("PRED") if idx_cut == 0 else None
        plt.yscale("log")
        plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    
    return mae, unmask_mae, mask_ratio

def out_to_file(string):
    print(string)
    with open(log_file, "a") as f:
        f.write(string + "\n")


data_div_json = "./B100/step1step2_0822_vanila.json"
with open(data_div_json, "r") as f:
    data_div = json.load(f)

save_folder = f"./B100/plot_{data_div_to_plot}_case_UNetUNet/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
log_file = f"{save_folder}log.txt"

train_list = data_div["train"]
val_list = data_div["val"]
test_list = data_div["test"]

num_train = len(train_list)
num_val = len(val_list)
num_test = len(test_list)

out_to_file(f"num_train: {num_train}")
out_to_file(f"num_val: {num_val}")
out_to_file(f"num_test: {num_test}")

axial_cut = 8
sagittal_cut = 4
coronal_cut = 4

MAX_PET = 3000
MIN_PET = 0
MAX_CT = 1976
MIN_CT = -1024

mae_list = []
unmask_mae_list = []
mask_ratio_list = []
# plot the test case

if data_div_to_plot == "train":
    target_list = train_list
elif data_div_to_plot == "val":
    target_list = val_list
elif data_div_to_plot == "test":
    target_list = test_list

for test_pair in target_list:
    out_to_file("")
    x_path = test_pair["STEP1"] # "STEP1": "./B100/f4noattn_step1_volume/STEP1_E4078.nii.gz",
    x_path = x_path.replace("f4noattn_step1_volume_vanila", "TOFNAC_resample")
    x_path = x_path.replace("STEP1", "PET_TOFNAC")
    y_path = test_pair["STEP2"] # "STEP2": "./B100/f4noattn_step2_volume/STEP2_E4078.nii.gz",
    z_path = test_pair["STEP1"].replace("STEP1", "STEP3_d3f64")
    case_name = y_path[-12:-7]
    out_to_file(f"Processing test case: {case_name}")
    out_to_file(f">>> TOFNAC_path: {x_path}")
    out_to_file(f">>> CTAC_path: {y_path}")
    out_to_file(f">>> PRED_path: {z_path}")

    x_data = nib.load(x_path).get_fdata()
    y_data = nib.load(y_path).get_fdata()
    z_data = nib.load(z_path).get_fdata()

    out_to_file(f">>> TOFNAC_shape: {x_data.shape}, CTAC_shape: {y_data.shape}, PRED_shape: {z_data.shape}")


    # for axial:
    save_name = f"{save_folder}{case_name}_axial_cut_{axial_cut}.png"
    mae, unmask_mae, mask_ratio = plot_case_from_view_cut(x_data, y_data, z_data, save_name, axial_cut, "axial", None)
    out_to_file(f">>> Saving to {save_name}")

    # for sagittal:
    save_name = f"{save_folder}{case_name}_sagittal_cut_{sagittal_cut}.png"
    _ = plot_case_from_view_cut(x_data, y_data, z_data, save_name, sagittal_cut, "sagittal", None)
    out_to_file(f">>> Saving to {save_name}")

    # for coronal:
    save_name = f"{save_folder}{case_name}_coronal_cut_{coronal_cut}.png"
    _ = plot_case_from_view_cut(x_data, y_data, z_data, save_name, coronal_cut, "coronal", None)
    out_to_file(f">>> Saving to {save_name}")

    out_to_file(f">>> MAE: {mae:.2f} HU, unmask MAE: {unmask_mae:.2f} HU, mask ratio: {mask_ratio*100:.2f}%")
    mae_list.append(mae)
    unmask_mae_list.append(unmask_mae)
    mask_ratio_list.append(mask_ratio)

out_to_file("")
out_to_file(f"The average MAE of the {data_div_to_plot} cases is {np.mean(mae_list):.2f} HU")
out_to_file(f"The average unmask MAE of the {data_div_to_plot} cases is {np.mean(unmask_mae_list):.2f} HU")
out_to_file(f"The average mask ratio of the {data_div_to_plot} cases is {np.mean(mask_ratio_list)*100:.2f}%")
