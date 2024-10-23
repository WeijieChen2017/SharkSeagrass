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

root_folder = "James_data_v3/"
fold_cv = 0
fold_cv_train = [0, 1, 2]
fold_cv_val = [3]
fold_cv_test = [4]

import json

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

import torch
import numpy as np

dim = 64
in_channel = 3
out_channel = 3
batch_size = 8
n_epoch = 200
n_epoch_eval = 10
n_epoch_save = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from monai.networks.nets import UNet

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=6,
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = torch.nn.MSELoss()

vq_weights_path = "f4_vq_weights.npy"
vq_weights = np.load(vq_weights_path)
print(f"Loading vq weights from {vq_weights_path}, shape: {vq_weights.shape}")

def train_or_eval_or_test(model, case_name, stage, ana_planes):
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

            if stage == "train":
                optimizer.zero_grad()
                y_hat = model(x_axial_post_quan)
                loss_val = loss(y_hat, y_axial_post_quan)
                loss_val.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    y_hat = model(x_axial_post_quan)
                    loss_val = loss(y_hat, y_axial_post_quan)
                
            case_loss.append(loss_val.item())

    case_loss = np.asarray(case_loss)
    return np.mean(case_loss)

save_folder = root_folder + f"James_v3_emb2emb_UNet_v1_cv{fold_cv}/"
best_eval_loss = 1e10
for idx_epoch in range(n_epoch):
    print(f"Epoch: {idx_epoch+1}/{n_epoch}")
    train_loss = 0.0
    val_loss = 0.0
    test_loss = 0.0

    for case_name in train_list:
        current_train_loss = train_or_eval_or_test(model, case_name, "train", "axial")
        print(f"Epoch [Train]: {idx_epoch+1}/{n_epoch}, case_name: {case_name}, train_loss: {current_train_loss}")
        train_loss += current_train_loss
    train_loss /= len(train_list)
    print(f"Epoch [Train]: {idx_epoch+1}/{n_epoch}, train_loss: {train_loss}")

    if (idx_epoch+1) % n_epoch_eval == 0:
        for case_name in val_list:
            current_val_loss = train_or_eval_or_test(model, case_name, "eval", "axial")
            print(f"Epoch [Eval]: {idx_epoch+1}/{n_epoch}, case_name: {case_name}, val_loss: {current_val_loss}")
            val_loss += current_val_loss
        val_loss /= len(val_list)
        print(f"Epoch [Eval]: {idx_epoch+1}/{n_epoch}, val_loss: {val_loss}")
        if val_loss < best_eval_loss:
            best_eval_loss = val_loss
            torch.save(model.state_dict(), save_folder + "best_model.pth")
            print(f"Best model saved at {save_folder}best_model.pth")
    
            for case_name in test_list:
                current_test_loss = train_or_eval_or_test(model, case_name, "test", "axial")
                print(f"Epoch [Test]: {idx_epoch+1}/{n_epoch}, case_name: {case_name}, test_loss: {current_test_loss}")
                test_loss += current_test_loss
            test_loss /= len(test_list)
            print(f"Epoch [Test]: {idx_epoch+1}/{n_epoch}, test_loss: {test_loss}")

    if (idx_epoch+1) % n_epoch_save == 0:
        torch.save(model.state_dict(), save_folder + f"model_{idx_epoch+1}.pth")
        print(f"Model saved at {save_folder}model_{idx_epoch+1}.pth")

    