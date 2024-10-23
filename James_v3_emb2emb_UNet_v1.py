
# --------------------------------
# PART: data division 
# --------------------------------
config = {}

root_folder = "James_data_v3/"
fold_cv = 0
fold_cv_train = [0, 1, 2]
fold_cv_val = [3]
fold_cv_test = [4]

config["root_folder"] = root_folder
config["fold_cv"] = fold_cv
config["fold_cv_train"] = fold_cv_train
config["fold_cv_val"] = fold_cv_val
config["fold_cv_test"] = fold_cv_test


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

# --------------------------------
# PART: model building
# --------------------------------

import torch
import numpy as np

batch_size = 64
dim = 64
in_channel = 3
out_channel = 3
batch_size = 8
n_epoch = 2000
n_epoch_eval = 20
n_epoch_save = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The current device is: ", device)

config["batch_size"] = batch_size
config["dim"] = dim
config["in_channel"] = in_channel
config["out_channel"] = out_channel
config["n_epoch"] = n_epoch
config["n_epoch_eval"] = n_epoch_eval
config["n_epoch_save"] = n_epoch_save
config["device"] = device

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

# --------------------------------
# PART: start training
# --------------------------------

save_folder = root_folder + f"James_v3_emb2emb_UNet_v1_cv{fold_cv}/"
import os
os.makedirs(save_folder, exist_ok=True)
best_eval_loss = 1e10

from James_v3_emb2emb_UNet_v1_utils import train_or_eval_or_test

for idx_epoch in range(n_epoch):
    print(f"Epoch: {idx_epoch+1}/{n_epoch}")
    train_loss = 0.0
    val_loss = 0.0
    test_loss = 0.0

    for case_name in train_list:
        current_train_loss = train_or_eval_or_test(model, optimizer, loss, case_name, "train", "axial", device, vq_weights, config)
        print(f"Epoch [Train]: {idx_epoch+1}/{n_epoch}, case_name: {case_name}, train_loss: {current_train_loss}")
        train_loss += current_train_loss
    train_loss /= len(train_list)
    print(f"Epoch [Train]: {idx_epoch+1}/{n_epoch}, train_loss: {train_loss}")

    if (idx_epoch+1) % n_epoch_eval == 0:
        for case_name in val_list:
            current_val_loss = train_or_eval_or_test(model, optimizer, loss, case_name, "eval", "axial", device, vq_weights, config)
            print(f"Epoch [Eval]: {idx_epoch+1}/{n_epoch}, case_name: {case_name}, val_loss: {current_val_loss}")
            val_loss += current_val_loss
        val_loss /= len(val_list)
        print(f"Epoch [Eval]: {idx_epoch+1}/{n_epoch}, val_loss: {val_loss}")
        if val_loss < best_eval_loss:
            best_eval_loss = val_loss
            torch.save(model.state_dict(), save_folder + "best_model.pth")
            print(f"Best model saved at {save_folder}best_model.pth")
    
            for case_name in test_list:
                current_test_loss = train_or_eval_or_test(model, optimizer, loss, case_name, "test", "axial", device, vq_weights, config)
                print(f"Epoch [Test]: {idx_epoch+1}/{n_epoch}, case_name: {case_name}, test_loss: {current_test_loss}")
                test_loss += current_test_loss
            test_loss /= len(test_list)
            print(f"Epoch [Test]: {idx_epoch+1}/{n_epoch}, test_loss: {test_loss}")

    if (idx_epoch+1) % n_epoch_save == 0:
        torch.save(model.state_dict(), save_folder + f"model_{idx_epoch+1}.pth")
        print(f"Model saved at {save_folder}model_{idx_epoch+1}.pth")

    