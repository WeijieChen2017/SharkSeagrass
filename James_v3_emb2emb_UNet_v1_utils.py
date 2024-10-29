# tag_list = [
#     'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
#     'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
#     'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
#     'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
#     'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
#     'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
#     'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
#     'E4139', 'E4143', 'E4144', 'E4147', 'E4152',
#     'E4155', 'E4157', 'E4158', 'E4162', 'E4163',
#     'E4165', 'E4166', 'E4172', 'E4181', 'E4182',
#     'E4183', 'E4185', 'E4187', 'E4189', 'E4193',
#     'E4197', 'E4198', 'E4207', 'E4208', 'E4216',
#     'E4217', 'E4219', 'E4220', 'E4232', 'E4237',
#     'E4238', 'E4239', 'E4241',
# ]

# root_folder = "James_data_v3/"

# # randomly divide the tag_list into 5 parts from cv0 to cv4
# import random

# random.seed(0)

# random.shuffle(tag_list)
# total_len = len(tag_list)
# cv_len = total_len // 5
# cv_list = []
# for i in range(5):
#     if i == 4:
#         cv_list.append(tag_list[i * cv_len:])
#     else:
#         cv_list.append(tag_list[i * cv_len: (i + 1) * cv_len])

# print("cv_list: ", cv_list)

# json_dict = {
#     "cv0": cv_list[0],
#     "cv1": cv_list[1],
#     "cv2": cv_list[2],
#     "cv3": cv_list[3],
#     "cv4": cv_list[4],
# }

# import json

# with open(root_folder + "cv_list.json", "w") as f:
#     json.dump(json_dict, f)

# print("cv_list.json saved.")

# {
#     "cv0": 
#     [
#         "E4128", "E4172", "E4238", "E4158", "E4129",
#         "E4155", "E4143", "E4197", "E4185", "E4131",
#         "E4162", "E4066", "E4124"
#     ],
#     "cv1":
#     [
#         "E4216", "E4081", "E4118", "E4074", "E4079",
#         "E4094", "E4115", "E4237", "E4084", "E4061",
#         "E4055", "E4098", "E4232"
#     ],
#     "cv2":
#     [
#         "E4058", "E4217", "E4166", "E4165", "E4092",
#         "E4163", "E4193", "E4105", "E4125", "E4198",
#         "E4157", "E4139", "E4207"
#     ],
#     "cv3":
#     [
#         "E4106", "E4068", "E4241", "E4219", "E4078",
#         "E4147", "E4138", "E4096", "E4152", "E4073",
#         "E4181", "E4187", "E4099"
#     ],
#     "cv4":
#     [
#         "E4077", "E4134", "E4091", "E4144", "E4114",
#         "E4130", "E4103", "E4239", "E4183", "E4208",
#         "E4120", "E4220", "E4137", "E4069", "E4189",
#         "E4182"
#     ]
# }

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom


def train_or_eval_or_test(
        model, # the adapter model
        optimizer, # the optimizer
        loss, # the loss function
        case_name, # the case_name to find in folders
        stage, # "train", "eval", "test"
        anatomical_plane, # "axial", "coronal", "sagittal"
        device, # cpu or cuda
        vq_weights, # the vq weights for each embedding
        config, # the config file
    ):

    # input 256, 256, 468
    # activated axis is sagittal, coronal, axial
    # 468 64 64
    # axial (64, 64, 3) (64, 64, 3)
    # axial (64, 64, 3) (64, 64, 3)
    # 256 117 64
    # coronal (64, 117, 3) (64, 117, 3)
    # coronal (64, 117, 3) (64, 117, 3)
    # 256 117 64
    # sagittal (64, 117, 3) (64, 117, 3)
    # sagittal (64, 117, 3) (64, 117, 3)
    # axial slice is 256, 256, 1 -> 1/4 -> 64, 64, 1 -> 64, 64
    # coronal slice is 256, 1, 468 -> 1/4 -> 64, 1, 117 -> 64, 117
    # sagittal slice is 1, 256, 468 -> 1/4 -> 1, 64, 117 -> 64, 117

    root_folder = config["root_folder"]
    batch_size = config["batch_size"]
    vq_norm_factor = config["vq_norm_factor"]
    zoom_factor = config["zoom_factor"]
    is_mask_train = config["apply_mask_train"]
    is_mask_eval = config["apply_mask_eval"]

    if stage == "train":
        model.train()
    else:
        model.eval()

    case_loss = []
    cnt_batch = 0

    mask_path = root_folder + f"mask/mask_body_contour_{case_name}.nii.gz"
    mask_file = nib.load(mask_path)
    mask_data = mask_file.get_fdata()
    len_z = mask_data.shape[2]
    # mask_data = mask_data > 0
    if len_z % zoom_factor != 0:
        # pad it to the nearest multiple of 4 at the end
        # print(f"Padding the z-axis to the nearest multiple of {len_factor}")
        pad_len = zoom_factor - len_z % zoom_factor
        mask_data = np.pad(mask_data, ((0, 0), (0, 0), (0, pad_len)), mode="constant", constant_values=0)


    if anatomical_plane == "axial":
        anatomical_zoom_factor = (1/zoom_factor, 1/zoom_factor, 1)
        anatomical_mask = zoom(mask_data, anatomical_zoom_factor, order=0)  # order=1 for bilinear interpolation
        anatomical_mask = np.squeeze(anatomical_mask)
        anatomical_mask = np.transpose(anatomical_mask, (2, 0, 1))
    elif anatomical_plane == "coronal":
        anatomical_zoom_factor = (1/zoom_factor, 1, 1/zoom_factor)
        anatomical_mask = zoom(mask_data, anatomical_zoom_factor, order=0)  # order=1 for bilinear interpolation
        anatomical_mask = np.squeeze(anatomical_mask)
        anatomical_mask = np.transpose(anatomical_mask, (1, 2, 0))
    elif anatomical_plane == "sagittal":
        anatomical_zoom_factor = (1, 1/zoom_factor, 1/zoom_factor)
        anatomical_mask = zoom(mask_data, anatomical_zoom_factor, order=0)  # order=1 for bilinear interpolation
        anatomical_mask = np.squeeze(anatomical_mask)
        anatomical_mask = np.transpose(anatomical_mask, (0, 2, 1))

    path_x = root_folder + f"index/{case_name}_x_{anatomical_plane}_ind.npy"
    path_y = root_folder + f"index/{case_name}_y_{anatomical_plane}_ind.npy"

    ind_data_x = np.load(path_x)
    ind_data_y = np.load(path_y)

    len_z = ind_data_y.shape[0] # padded
    len_x_len_y = ind_data_y.shape[1] # x*y
    len_x = 64
    len_y = int(len_x_len_y // len_x)

    # Initialize tensors to hold the batch results
    x_batch = []
    y_batch = []
    mask_batch = []

    for i in range(len_z):
        
        slice_x = ind_data_x[i, :].reshape(len_x, len_y)
        slice_y = ind_data_y[i, :].reshape(len_x, len_y)

        x_post_quan = vq_weights[slice_x]
        y_post_quan = vq_weights[slice_y]

        x_post_quan = x_post_quan / (vq_norm_factor * 2) + 0.5 # [-1, 1] -> [0, 1] for ReLU activation
        y_post_quan = y_post_quan / (vq_norm_factor * 2) + 0.5 # [-1, 1] -> [0, 1] for ReLU activation
        slice_mask = anatomical_mask[i, :, :]
        # duplicate the slice_mask to 3 channels at last axis
        slice_mask = np.stack((slice_mask, slice_mask, slice_mask), axis=-1)

        # slice_x = np.rot90(slice_x)
        # slice_y = np.rot90(slice_y)
        # slice_mask = np.rot90(slice_mask)
        # print(slice_x.shape, slice_y.shape, slice_mask.shape)
        # print(slice_mask.strides)

        x_post_quan = torch.from_numpy(x_post_quan).float().to(device)
        x_post_quan = x_post_quan.unsqueeze(0)
        x_post_quan = x_post_quan.permute(0, 3, 1, 2)

        y_post_quan = torch.from_numpy(y_post_quan).float().to(device)
        y_post_quan = y_post_quan.unsqueeze(0)
        y_post_quan = y_post_quan.permute(0, 3, 2, 1)

        tensor_mask = torch.from_numpy(slice_mask).float().to(device)
        tensor_mask = tensor_mask.unsqueeze(0)
        tensor_mask = tensor_mask.permute(0, 3, 1, 2)

        # if the last dim is not divided by 4, pad it to the nearest multiple of 4
        if x_post_quan.shape[-1] % zoom_factor != 0:
            pad_len = zoom_factor - x_post_quan.shape[2] % zoom_factor
            x_post_quan = torch.nn.functional.pad(x_post_quan, (0, pad_len, 0, pad_len), mode="constant", value=0)
            y_post_quan = torch.nn.functional.pad(y_post_quan, (0, pad_len, 0, pad_len), mode="constant", value=0)
            tensor_mask = torch.nn.functional.pad(tensor_mask, (0, pad_len, 0, pad_len), mode="constant", value=0)

        print(x_post_quan.shape, y_post_quan.shape, tensor_mask.shape)

        cnt_batch += 1
        x_batch.append(x_post_quan)
        y_batch.append(y_post_quan)
        mask_batch.append(tensor_mask)

        if cnt_batch == batch_size or i == len_z - 1:
            x_batch = torch.cat(x_batch, dim=0)
            y_batch = torch.cat(y_batch, dim=0)
            mask_batch = torch.cat(mask_batch, dim=0)

            if stage == "train":
                optimizer.zero_grad()
                y_hat = model(x_batch)
                if is_mask_train:
                    loss_term = ((loss(y_hat, y_batch) * mask_batch).mean())
                else:
                    loss_term = loss(y_hat, y_batch).mean()
                loss_term.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    y_hat = model(x_batch)
                    if is_mask_eval:
                        loss_term = ((loss(y_hat, y_batch) * mask_batch).mean())
                    else:
                        loss_term = loss(y_hat, y_batch).mean()
                
            case_loss.append(loss_term.item())
            cnt_batch = 0
            x_batch = []
            y_batch = []
            mask_batch = []


    case_loss = np.asarray(case_loss)
    return np.mean(case_loss)

