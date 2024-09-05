import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandFlipd, 
    RandRotated,
    Transposed,
    NormalizeIntensityd,
)
from monai.data import CacheDataset, DataLoader
from SAM_Med3D_util import Sam3D, ImageEncoderViT3D, partial

input_modality = ["PET", "CT"]
img_size = 400
cube_size = 128
in_channels = 1
out_channels = 1
batch_size = 4
num_epoch = 100000
debug_file_num = 0
save_per_epoch = 1000
eval_per_epoch = 100
plot_per_epoch = 100
CT_NORM = 5000
cache_rate = 0.1
device = torch.device("cuda:0")
root_folder = "./B100/SAM-Med3D_v1"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
print("The root folder is: ", root_folder)
log_file = os.path.join(root_folder, "log.txt")

encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]
prompt_embed_dim = 384
image_size = 128
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size

model = Sam3D(
    image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
)

model.load_pretrain(path="sam_med3d_turbo.pth")
model.to(device)




# set the data transform
train_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
        NormalizeIntensityd(keys="PET"),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(cube_size, cube_size, cube_size),
                         random_center=True, random_size=False),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
        NormalizeIntensityd(keys="PET"),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(cube_size, cube_size, cube_size),
                         random_center=True, random_size=False),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
        NormalizeIntensityd(keys="PET"),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(cube_size, cube_size, cube_size),
                         random_center=True, random_size=False),
    ]
)

data_division_file = "./B100/B100_0822.json"
with open(data_division_file, "r") as f:
    data_division = json.load(f)

if debug_file_num > 0:
    train_list = data_division["train"][:debug_file_num]
    val_list = data_division["val"][:debug_file_num]
    test_list = data_division["test"][:debug_file_num]
else:
    train_list = data_division["train"]
    val_list = data_division["val"]
    test_list = data_division["test"]

num_train_files = len(train_list)
num_val_files = len(val_list)
num_test_files = len(test_list)

print("The number of train files is: ", num_train_files)
print("The number of val files is: ", num_val_files)
print("The number of test files is: ", num_test_files)
print()

# save the data division file
data_division_file = os.path.join(root_folder, "data_division.json")

train_ds = CacheDataset(
    data=train_list,
    transform=train_transforms,
    cache_num=num_train_files,
    cache_rate=cache_rate,
    num_workers=4,
)

val_ds = CacheDataset(
    data=val_list,
    transform=val_transforms, 
    cache_num=num_val_files,
    cache_rate=cache_rate,
    num_workers=4,
)

test_ds = CacheDataset(
    data=test_list,
    transform=test_transforms,
    cache_num=num_test_files,
    cache_rate=cache_rate,
    num_workers=4,
)



train_loader = DataLoader(train_ds, 
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=4,

)
val_loader = DataLoader(val_ds, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=4,
)

test_loader = DataLoader(test_ds,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
)

# set the optimizer and loss
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_function = torch.nn.L1Loss()
output_loss = torch.nn.L1Loss()
# ds_loss = DeepSupervisionLoss(
#     loss = output_loss,
#     weight_mode = "exp",
#     weights = None,
# )

best_val_loss = 1e10
n_train_batches = len(train_loader)
n_val_batches = len(val_loader)
n_test_batches = len(test_loader)

def plot_results(inputs, labels, outputs, idx_epoch):
    # plot the results
    n_block = 8
    if inputs.shape[0] < n_block:
        n_block = inputs.shape[0]
    plt.figure(figsize=(12, n_block*1.5), dpi=300)

    n_row = n_block
    n_col = 6

    for i in range(n_block):
        # first three and hist
        plt.subplot(n_row, n_col, i * n_col + 1)
        img_PET = np.rot90(inputs[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, 0, 1))
        plt.imshow(img_PET, cmap="gray")
        # plt.title("input PET")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 2)
        img_CT = np.rot90(labels[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, 0, 1))
        plt.imshow(img_CT, cmap="gray")
        # plt.title("label CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 3)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        img_pred = np.rot90(outputs[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, 0, 1))
        plt.imshow(img_pred, cmap="gray")
        # plt.title("output CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 4)
        # img_PET = np.clip(img_PET, 0, 1)
        plt.hist(img_PET.flatten(), bins=100)
        # plt.title("input PET")
        plt.yscale("log")
        plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 5)
        # img_CT = np.clip(img_CT, 0, 1)
        plt.hist(img_CT.flatten(), bins=100)
        # plt.title("label CT")
        plt.yscale("log")
        plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 6)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist(img_pred.flatten(), bins=100)
        # plt.title("output CT")
        plt.yscale("log")
        plt.axis("off")
        plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(root_folder, f"epoch_{idx_epoch}.png"))
    plt.close()



# start the training
for idx_epoch in range(num_epoch):

    # train the model
    model.train()
    train_loss = 0
    for idx_batch, batch_data in enumerate(train_loader):
        inputs = batch_data["PET"].to(device)
        labels = batch_data["CT"].to(device)
        # print("inputs.shape: ", inputs.shape, "labels.shape: ", labels.shape)
        # inputs.shape:  torch.Size([1, 1, 128, 128, 128]) labels.shape:  torch.Size([1, 1, 128, 128, 128])
        # outputs.shape:  torch.Size([1, 1, 128, 128, 128])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = output_loss(outputs, labels)
        # print("outputs.shape: ", outputs.shape)
        # loss = loss_function(outputs, labels)
        # loss = ds_loss(torch.unbind(outputs, 1), labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {idx_epoch}, batch [{idx_batch}]/[{n_train_batches}], loss: {loss.item()*CT_NORM:.4f}")
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {idx_epoch}, train_loss: {train_loss*CT_NORM:.4f}")
    # log the results
    with open(log_file, "a") as f:
        f.write(f"Epoch {idx_epoch}, train_loss: {train_loss*CT_NORM:.4f}\n")

    if idx_epoch % plot_per_epoch == 0:
        plot_results(inputs, labels, outputs, idx_epoch)

    # evaluate the model
    if idx_epoch % eval_per_epoch == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for idx_batch, batch_data in enumerate(val_loader):
                inputs = batch_data["PET"].to(device)
                labels = batch_data["CT"].to(device)
                outputs = model(inputs)
                loss = output_loss(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Epoch {idx_epoch}, val_loss: {val_loss*CT_NORM:.4f}")
            with open(log_file, "a") as f:
                f.write(f"Epoch {idx_epoch}, val_loss: {val_loss*CT_NORM:.4f}\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(root_folder, "best_model.pth"))
                print(f"Save the best model with val_loss: {val_loss*CT_NORM:.4f} at epoch {idx_epoch}")
                with open(log_file, "a") as f:
                    f.write(f"Save the best model with val_loss: {val_loss*CT_NORM:.4f} at epoch {idx_epoch}\n")
                
                # test the model
                with torch.no_grad():
                    test_loss = 0
                    for idx_batch, batch_data in enumerate(test_loader):
                        inputs = batch_data["PET"].to(device)
                        labels = batch_data["CT"].to(device)
                        outputs = model(inputs)
                        loss = output_loss(outputs, labels)
                        test_loss += loss.item()
                    test_loss /= len(test_loader)
                    print(f"Epoch {idx_epoch}, test_loss: {test_loss*CT_NORM:.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"Epoch {idx_epoch}, test_loss: {test_loss*CT_NORM:.4f}\n")

    # save the model
    if idx_epoch % save_per_epoch == 0:
        save_path = os.path.join(root_folder, f"model_epoch_{idx_epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Save model to {save_path}")

    

     