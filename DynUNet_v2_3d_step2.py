import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    RandSpatialCropd,
    # RandSpatialCropSamplesd,
    # RandFlipd, 
    # RandRotated,
    # Transposed,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
)
from monai.data import CacheDataset, DataLoader
from monai.losses import DeepSupervisionLoss

input_modality = ["STEP1", "STEP2"]
img_size = 400
cube_size = 128
in_channels = 1
out_channels = 1
batch_size = 2
num_epoch = 10000
debug_file_num = 0
save_per_epoch = 100
eval_per_epoch = 10
plot_per_epoch = 10
CT_NORM = 5000
cache_rate = 1.0
root_folder = "./B100/dynunet3d_v2_step2_intensity"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
print("The root folder is: ", root_folder)
log_file = os.path.join(root_folder, "log.txt")

kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2]]

model = DynUNet(
    spatial_dims=3,
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernels,
    strides=strides,
    upsample_kernel_size=strides[1:],
    filters=(64, 128, 256),
    dropout=0.,
    norm_name=('INSTANCE', {'affine': True}), 
    act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
    deep_supervision=True,
    deep_supr_num=1,
    res_block=False,
    trans_bias=False,
)


# set the data transform
train_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        # Transposed(keys=input_modality, indices=(2, 0, 1)),
        # EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
        # EnsureChannelFirstd(keys="PET", channel_dim=-1),
        EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(cube_size, cube_size, cube_size),
                         random_center=True, random_size=False),
        RandGaussianNoised(keys=input_modality, prob=0.1, mean=0.0, std=0.1),
        RandGaussianSharpend(keys=input_modality, prob=0.1),
        RandGaussianSmoothd(keys=input_modality, prob=0.1),
        # RandSpatialCropd(keys="PET",
        #                  roi_size=(cube_size, cube_size, cube_size),
        #                  random_center=True, random_size=False),
        # RandSpatialCropd(keys="CT",
        #                  roi_size=(cube_size, cube_size, cube_size),
        #                  random_center=True, random_size=False),
        # RandSpatialCropSamplesd(keys="PET",
        #                         roi_size=(img_size, img_size, in_channels),
        #                         num_samples=batch_size,
        #                         random_size=False, random_center=False),
        # RandSpatialCropSamplesd(keys="CT",
        #                         roi_size=(img_size, img_size, out_channels),
        #                         num_samples=batch_size,
        #                         random_size=False, random_center=False),
        # RandSpatialCropSamplesd(keys=input_modality,
        #                         roi_size=(img_size, img_size, in_channels),
        #                         num_samples=num_samples,
        #                         random_size=False, random_center=True),
        # RandFlipd(keys=input_modality, prob=0.5, spatial_axis=0),
        # RandFlipd(keys=input_modality, prob=0.5, spatial_axis=1),
        # RandFlipd(keys=input_modality, prob=0.5, spatial_axis=2),
        # RandRotated(keys=input_modality, prob=0.5, range_x=30),
        
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        # EnsureChannelFirstd(keys="PET", channel_dim=-1),
        EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
        # RandSpatialCropSamplesd(keys="PET",
        #                         roi_size=(img_size, img_size, in_channels),
        #                         num_samples=batch_size,
        #                         random_size=False, random_center=False),
        # RandSpatialCropSamplesd(keys="CT",
        #                         roi_size=(img_size, img_size, out_channels),
                                # num_samples=batch_size,
                                # random_size=False, random_center=False),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(cube_size, cube_size, cube_size),
                         random_center=True, random_size=False),
        # RandSpatialCropd(keys="PET",
        #                  roi_size=(cube_size, cube_size, cube_size),
        #                  random_center=True, random_size=False),
        # RandSpatialCropd(keys="CT",
        #                  roi_size=(cube_size, cube_size, cube_size),
        #                  random_center=True, random_size=False),
        # RandSpatialCropSamplesd(keys=input_modality,
        #                         roi_size=(img_size, img_size, in_channels),
        #                         num_samples=num_samples,
        #                         random_size=False, random_center=True),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        # EnsureChannelFirstd(keys="PET", channel_dim=-1),
        EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(cube_size, cube_size, cube_size),
                         random_center=True, random_size=False),
        # RandSpatialCropd(keys="PET",
        #                  roi_size=(cube_size, cube_size, cube_size),
        #                  random_center=True, random_size=False),
        # RandSpatialCropd(keys="CT",
        #                  roi_size=(cube_size, cube_size, cube_size),
        #                  random_center=True, random_size=False),
    ]
)

data_division_file = "./step1step2_0822.json"
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

device = torch.device("cuda:0")
model.to(device)

# set the optimizer and loss
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_function = torch.nn.L1Loss()
# loss: main loss instance, e.g DiceLoss().
# weight_mode: {``"same"``, ``"exp"``, ``"two"``}
#     Specifies the weights calculation for each image level. Defaults to ``"exp"``.
#     - ``"same"``: all weights are equal to 1.
#     - ``"exp"``: exponentially decreasing weights by a power of 2: 1, 0.5, 0.25, 0.125, etc .
#     - ``"two"``: equal smaller weights for lower levels: 1, 0.5, 0.5, 0.5, 0.5, etc
# weights: a list of weights to apply to each deeply supervised sub-loss, if provided, this will be used
#     regardless of the weight_mode
output_loss = torch.nn.L1Loss()
ds_loss = DeepSupervisionLoss(
    loss = output_loss,
    weight_mode = "exp",
    weights = None,
)
# def forward(self, input: Union[None, torch.Tensor, list[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:

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
        img_pred = np.rot90(outputs[i, 0, :, :, :, cube_size // 2].detach().cpu().numpy())
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
        inputs = batch_data["STEP1"].to(device)
        labels = batch_data["STEP2"].to(device)
        # print("inputs.shape: ", inputs.shape, "labels.shape: ", labels.shape)
        # inputs.shape:  torch.Size([5, 1, 96, 96, 96]) labels.shape:  torch.Size([5, 1, 96, 96, 96])
        # outputs.shape:  torch.Size([5, 2, 1, 96, 96, 96])
        optimizer.zero_grad()
        outputs = model(inputs)
        # print("outputs.shape: ", outputs.shape)
        # loss = loss_function(outputs, labels)
        loss = ds_loss(torch.unbind(outputs, 1), labels)
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
                inputs = batch_data["STEP1"].to(device)
                labels = batch_data["STEP2"].to(device)
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
                        inputs = batch_data["STEP1"].to(device)
                        labels = batch_data["STEP2"].to(device)
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

    

     