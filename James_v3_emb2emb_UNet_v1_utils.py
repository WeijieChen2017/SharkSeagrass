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


import numpy as np
import torch


def train_or_eval_or_test(
        model, # the adapter model
        optimizer, # the optimizer
        loss, # the loss function
        case_name, # the case_name to find in folders
        stage, # "train", "eval", "test"
        ana_planes, # "axial", "coronal", "sagittal"
        device, # cpu or cuda
        vq_weights, # the vq weights for each embedding
        config, # the config file
    ):

    root_folder = config["root_folder"]
    batch_size = config["batch_size"]


    if stage == "train":
        model.train()
    else:
        model.eval()

    if ana_planes == "axial":
        path_x_axial = root_folder + f"index/{case_name}_x_axial_ind.npy"
        path_y_axial = root_folder + f"index/{case_name}_y_axial_ind.npy"
        
        file_x_axial = np.load(path_x_axial)
        file_y_axial = np.load(path_y_axial)

        len_z = file_x_axial.shape[0] # padded
        len_x_len_y = file_x_axial.shape[1] # x*y
        len_x = int(np.sqrt(len_x_len_y))
        len_y = len_x

        case_loss = []
        cnt_batch = 0

        # Initialize tensors to hold the batch results
        x_axial_batch = []
        y_axial_batch = []

        for i in range(len_z):
            
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
            
            cnt_batch += 1
            x_axial_batch.append(x_axial_post_quan)
            y_axial_batch.append(y_axial_post_quan)

            if cnt_batch == batch_size or i == len_z - 1:
                x_axial_batch = torch.cat(x_axial_batch, dim=0)
                y_axial_batch = torch.cat(y_axial_batch, dim=0)

                if stage == "train":
                    optimizer.zero_grad()
                    y_hat = model(x_axial_batch)
                    loss_val = loss(y_hat, y_axial_batch)
                    loss_val.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        y_hat = model(x_axial_batch)
                        loss_val = loss(y_hat, y_axial_batch)
                    
                case_loss.append(loss_val.item())
                cnt_batch = 0
                x_axial_batch = []
                y_axial_batch = []

    case_loss = np.asarray(case_loss)
    return np.mean(case_loss)

