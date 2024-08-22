import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from monai.networks.nets import DynUNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, RandSpatialCropd, RandFlipd, RandRotated
from monai.data import CacheDataset, DataLoader



input_modality = ["PET", "CT"]
img_size = 400
in_channels = 5
batch_size = 16
num_epoch = 10000
save_per_epoch = 100
eval_per_epoch = 50
plot_per_epoch = 50
root_folder = "./B100/dynunet_v1"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
print("The root folder is: ", root_folder)


model = DynUNet(
    spatial_dims=2,
    in_channels=5,
    out_channels=1,
    kernel_size=(3, 3, 3),
    strides=(2, 2, 2),
    upsample_kernel_size=(3, 3, 3),
    filters=(32, 64, 128, 256),
    dropout=0.1,
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
        EnsureChannelFirstd(keys=input_modality),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(in_channels, img_size, img_size),
                         random_center=True, random_size=False),
        # RandSpatialCropSamplesd(keys=input_modality,
        #                         roi_size=(img_size, img_size, in_channels),
        #                         num_samples=num_samples,
        #                         random_size=False, random_center=True),
        RandFlipd(keys=input_modality, prob=0.5, spatial_axis=0),
        RandFlipd(keys=input_modality, prob=0.5, spatial_axis=1),
        RandFlipd(keys=input_modality, prob=0.5, spatial_axis=2),
        RandRotated(keys=input_modality, prob=0.5, range_x=15, range_y=15, range_z=15),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        EnsureChannelFirstd(keys=input_modality),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(in_channels, img_size, img_size),
                         random_center=True, random_size=False),
        # RandSpatialCropSamplesd(keys=input_modality,
        #                         roi_size=(img_size, img_size, in_channels),
        #                         num_samples=num_samples,
        #                         random_size=False, random_center=True),
    ]
)

data_division_file = "./B100/B100_0822.json"
with open(data_division_file, "r") as f:
    data_division = json.load(f)

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
    cache_rate=1.0,
    num_workers=4,
)

val_ds = CacheDataset(
    data=val_list,
    transform=val_transforms, 
    cache_num=num_val_files,
    cache_rate=1.0,
    num_workers=4,
)

train_loader = DataLoader(train_ds, 
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=4,
                        )
val_loader = DataLoader(val_ds, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=4,
                        )

device = torch.device("cuda:0")
model.to(device)

# set the optimizer and loss
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_function = torch.nn.L1Loss()

# start the training
for idx_epoch in range(num_epoch):

    # train the model
    model.train()
    for idx_batch, batch_data in enumerate(train_loader):
        inputs = batch_data["PET"].to(device)
        labels = batch_data["CT"].to(device)
        # [16, 1, 5, 400, 400]
        # remove the second dimension
        inputs = inputs.squeeze(1)
        labels = labels.squeeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {idx_epoch}, batch {idx_batch}, loss: {loss.item():.4f}")

    # evaluate the model
    if idx_epoch % eval_per_epoch == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for idx_batch, batch_data in enumerate(val_loader):
                inputs = batch_data["PET"].to(device)
                labels = batch_data["CT"].to(device)
                inputs = inputs.squeeze(1)
                labels = labels.squeeze(1)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Epoch {idx_epoch}, val_loss: {val_loss:.4f}")

    # save the model
    if idx_epoch % save_per_epoch == 0:
        save_path = os.path.join(root_folder, f"model_epoch_{idx_epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Save model to {save_path}")

    # plot the results
    if idx_epoch % plot_per_epoch == 0:
        plt.figure(figsize=(10, 5), dpi=300)

        # first three and hist
        plt.subplot(4, 6, 1)
        img_PET = np.rot90(inputs[0, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_PET, cmap="gray")
        plt.title("input PET")
        plt.axis("off")

        plt.subplot(4, 6, 2)
        img_CT = np.rot90(labels[0, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_CT, cmap="gray")
        plt.title("label CT")
        plt.axis("off")

        plt.subplot(4, 6, 3)
        img_pred = np.rot90(outputs[0, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_pred, cmap="gray")
        plt.title("output CT")
        plt.axis("off")

        plt.subplot(4, 6, 7)
        img_PET = np.clip(img_PET, 0, 1)
        plt.hist(img_PET.flatten(), bins=100)
        plt.title("input PET")
        plt.xlim(0, 1)

        plt.subplot(4, 6, 8)
        img_CT = np.clip(img_CT, 0, 1)
        plt.hist(img_CT.flatten(), bins=100)
        plt.title("label CT")
        plt.xlim(0, 1)

        plt.subplot(4, 6, 9)
        img_pred = np.clip(img_pred, 0, 1)
        plt.hist(img_pred.flatten(), bins=100)
        plt.title("output CT")
        plt.xlim(0, 1)

        # second three and hist
        plt.subplot(4, 6, 4)
        img_PET = np.rot90(inputs[1, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_PET, cmap="gray")
        plt.title("input PET")
        plt.axis("off")

        plt.subplot(4, 6, 5)
        img_CT = np.rot90(labels[1, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_CT, cmap="gray")
        plt.title("label CT")
        plt.axis("off")

        plt.subplot(4, 6, 6)
        img_pred = np.rot90(outputs[1, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_pred, cmap="gray")
        plt.title("output CT")
        plt.axis("off")

        plt.subplot(4, 6, 10)
        img_PET = np.clip(img_PET, 0, 1)
        plt.hist(img_PET.flatten(), bins=100)
        plt.title("input PET")
        plt.xlim(0, 1)

        plt.subplot(4, 6, 11)
        img_CT = np.clip(img_CT, 0, 1)
        plt.hist(img_CT.flatten(), bins=100)
        plt.title("label CT")
        plt.xlim(0, 1)

        plt.subplot(4, 6, 12)
        img_pred = np.clip(img_pred, 0, 1)
        plt.hist(img_pred.flatten(), bins=100)
        plt.title("output CT")
        plt.xlim(0, 1)

        # third three and hist
        plt.subplot(4, 6, 13)
        img_PET = np.rot90(inputs[2, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_PET, cmap="gray")
        plt.title("input PET")
        plt.axis("off")

        plt.subplot(4, 6, 14)
        img_CT = np.rot90(labels[2, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_CT, cmap="gray")
        plt.title("label CT")
        plt.axis("off")

        plt.subplot(4, 6, 15)
        img_pred = np.rot90(outputs[2, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_pred, cmap="gray")
        plt.title("output CT")
        plt.axis("off")

        plt.subplot(4, 6, 19)
        img_PET = np.clip(img_PET, 0, 1)
        plt.hist(img_PET.flatten(), bins=100)
        plt.title("input PET")
        plt.xlim(0, 1)

        plt.subplot(4, 6, 20)
        img_CT = np.clip(img_CT, 0, 1)
        plt.hist(img_CT.flatten(), bins=100)
        plt.title("label CT")
        plt.xlim(0, 1)

        plt.subplot(4, 6, 21)
        img_pred = np.clip(img_pred, 0, 1)
        plt.hist(img_pred.flatten(), bins=100)
        plt.title("output CT")
        plt.xlim(0, 1)

        # forth three and hist
        plt.subplot(4, 6, 16)
        img_PET = np.rot90(inputs[3, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_PET, cmap="gray")
        plt.title("input PET")
        plt.axis("off")

        plt.subplot(4, 6, 17)
        img_CT = np.rot90(labels[3, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_CT, cmap="gray")
        plt.title("label CT")
        plt.axis("off")

        plt.subplot(4, 6, 18)
        img_pred = np.rot90(outputs[3, in_channels // 2, :, :].cpu().numpy())
        plt.imshow(img_pred, cmap="gray")
        plt.title("output CT")
        plt.axis("off")

        plt.subplot(4, 6, 22)
        img_PET = np.clip(img_PET, 0, 1)
        plt.hist(img_PET.flatten(), bins=100)
        plt.title("input PET")
        plt.xlim(0, 1)

        plt.subplot(4, 6, 23)
        img_CT = np.clip(img_CT, 0, 1)
        plt.hist(img_CT.flatten(), bins=100)
        plt.title("label CT")
        plt.xlim(0, 1)

        plt.subplot(4, 6, 24)
        img_pred = np.clip(img_pred, 0, 1)
        plt.hist(img_pred.flatten(), bins=100)
        plt.title("output CT")
        plt.xlim(0, 1)


    