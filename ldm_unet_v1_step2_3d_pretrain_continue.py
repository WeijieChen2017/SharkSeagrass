import os
import json
import torch
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    RandGaussianSmoothd,
    ScaleIntensityRanged,
)
from monai.data import CacheDataset, DataLoader
from monai.losses import DeepSupervisionLoss

# mode = "d4f32"
mode = "d3f64"


input_modality = ["STEP1", "STEP2"]
img_size = 400
cube_size = 128
in_channels = 1
out_channels = 1
num_epoch = 10000
debug_file_num = 0
save_per_epoch = 10
eval_per_epoch = 10
plot_per_epoch = 1
CT_NORM = 5000
CT_MIN = -1024
CT_MAX = 3976
train_case = 0
val_case = 0
test_case = 0
meaningful_batch_th = 0.157 * 0.3
root_folder = f"./B100/dynunet3d_v2_step2_pretrain_{mode}_continue/"
pretrain_folder = f"./B100/dynunet3d_v2_step2_pretrain_{mode}/"
# dataset_folder = "tsv1_ct/"
# data_division_file = "tsv1_ct_over128.json"
data_division_file = "./B100/step1step2_0822_vanila.json"
if "tsv1_ct" in data_division_file:
    cache_rate = 0.125
    eval_per_epoch = 10
elif "step1step2" in data_division_file:
    cache_rate = 0.5
    eval_per_epoch = 1
else:
    cache_rate = 0.05

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
print("The root folder is: ", root_folder)
log_file = os.path.join(root_folder, "log.txt")

if mode == "d4f32":
    kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    filters = (32, 64, 128, 256)
    device = torch.device("cuda:1")
    batch_size = 3
    train_case = 0
elif mode == "d3f64":
    kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2]]
    filters = (64, 128, 256)
    device = torch.device("cuda:0")
    batch_size = 2
    train_case = 650

model = DynUNet(
    spatial_dims=3,
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernels,
    strides=strides,
    upsample_kernel_size=strides[1:],
    filters=filters,
    dropout=0.,
    norm_name=('INSTANCE', {'affine': True}), 
    act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
    deep_supervision=True,
    deep_supr_num=1,
    res_block=False,
    trans_bias=False,
)

pretrain_path = os.path.join(pretrain_folder, "best_model.pth")
model.load_state_dict(torch.load(pretrain_path))
print("Load the pretrain model from: ", pretrain_path)

# set the data transform
train_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        # Transposed(keys=input_modality, indices=(2, 0, 1)),
        # EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
        # EnsureChannelFirstd(keys="PET", channel_dim=-1),
        EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
        ScaleIntensityRanged(keys=input_modality, a_min=CT_MIN, a_max=CT_MAX, b_min=-1.0, b_max=1.0, clip=True),
        # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(cube_size, cube_size, cube_size),
                         random_center=True, random_size=False),
        # RandGaussianSmoothd(keys="STEP1", prob=1.,
        #                     sigma_x=(0.5, 2.5), sigma_y=(0.5, 2.5), sigma_z=(0.5, 2.5)),
        # RandGaussianSharpend(keys="STEP1", prob=1.),
        # RandGaussianNoised(keys="STEP1", prob=1., mean=0.0, std=0.1),
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
        ScaleIntensityRanged(keys=input_modality, a_min=CT_MIN, a_max=CT_MAX, b_min=-1.0, b_max=1.0, clip=True),
        # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
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
        # RandGaussianSmoothd(keys="STEP1", prob=1.,
        #                     sigma_x=(0.5, 2.5), sigma_y=(0.5, 2.5), sigma_z=(0.5, 2.5)),
        # RandGaussianSharpend(keys="STEP1", prob=1.),
        # RandGaussianNoised(keys="STEP1", prob=1., mean=0.0, std=0.1),
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
        ScaleIntensityRanged(keys=input_modality, a_min=CT_MIN, a_max=CT_MAX, b_min=-1.0, b_max=1.0, clip=True),
        # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
        RandSpatialCropd(keys=input_modality,
                         roi_size=(cube_size, cube_size, cube_size),
                         random_center=True, random_size=False),
        # RandGaussianSmoothd(keys="STEP1", prob=1.,
        #                     sigma_x=(0.5, 2.5), sigma_y=(0.5, 2.5), sigma_z=(0.5, 2.5)),
        # RandGaussianSharpend(keys="STEP1", prob=1.),
        # RandGaussianNoised(keys="STEP1", prob=1., mean=0.0, std=0.01),
        # RandSpatialCropd(keys="PET",
        #                  roi_size=(cube_size, cube_size, cube_size),
        #                  random_center=True, random_size=False),
        # RandSpatialCropd(keys="CT",
        #                  roi_size=(cube_size, cube_size, cube_size),
        #                  random_center=True, random_size=False),
    ]
)

# data_division_file = "./step1step2_0822.json"
# with open(data_division_file, "r") as f:
#     data_division = json.load(f)

# if debug_file_num > 0:
#     train_list = data_division["train"][:debug_file_num]
#     val_list = data_division["val"][:debug_file_num]
#     test_list = data_division["test"][:debug_file_num]
# else:
#     train_list = data_division["train"]
#     val_list = data_division["val"]
#     test_list = data_division["test"]

# num_train_files = len(train_list)
# num_val_files = len(val_list)
# num_test_files = len(test_list)
# print(f"The data search path is: ", dataset_folder+"*.nii.gz")
# dataset_folder = "tsv1_ct"
# datalist = sorted(glob.glob(dataset_folder+"/*.nii.gz"))
# random.shuffle(datalist)
# print(f"{len(datalist)} files are found in the dataset folder")
# data_pairs = [{"STEP1": item, "STEP2": item,} for item in datalist]

# load TotalSegmentator data
# with open(data_division_file, "r") as f:
#     data_pairs = json.load(f)

# train_list = data_pairs[:int(len(data_pairs)*train_ratio)]
# val_list = data_pairs[int(len(data_pairs)*train_ratio):int(len(data_pairs)*(train_ratio+val_ratio))]
# test_list = data_pairs[int(len(data_pairs)*(train_ratio+val_ratio)):]

with open(data_division_file, "r") as f:
    data_division = json.load(f)

train_list = data_division["train"]
val_list = data_division["val"]
test_list = data_division["test"]

# load TOFNAC data
if train_case > 0:
    train_list = train_list[:train_case]
if val_case > 0:
    val_list = val_list[:val_case]
if test_case > 0:
    test_list = test_list[:test_case]

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
    num_workers=6,
)

val_ds = CacheDataset(
    data=val_list,
    transform=val_transforms, 
    cache_num=num_val_files,
    cache_rate=cache_rate,
    num_workers=2,
)

test_ds = CacheDataset(
    data=test_list,
    transform=test_transforms,
    cache_num=num_test_files,
    cache_rate=cache_rate,
    num_workers=2,
)



train_loader = DataLoader(train_ds, 
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=4,
                        persistent_workers=True,
                        pin_memory=True,

)
val_loader = DataLoader(val_ds, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=4,
                        persistent_workers=True,
                        pin_memory=True,
)

test_loader = DataLoader(test_ds,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        persistent_workers=True,
                        pin_memory=True,
)

def check_batch_cube_size(batch_data, check_size):
    # given a batch, N, C, D, H, W, check wither D, H, W are the same as check_size
    for key in batch_data.keys():
        if key == "label":
            continue
        if batch_data[key].shape[2] != check_size or batch_data[key].shape[3] != check_size or batch_data[key].shape[4] != check_size:
            return False
    return True

def check_whether_full_batch(batch_data):
    # given a batch, check whether the batch is full
    for key in batch_data.keys():
        if batch_data[key].shape[0] != batch_size:
            return False
    return True

def check_whether_batch_meaningful(batch_data):
    # given a batch, check whether the batch is meaningful
    for key in batch_data.keys():
        # across all the axis,
        cube_mean = torch.abs(torch.mean(batch_data[key]))
        if cube_mean < meaningful_batch_th:
            return cube_mean, False
    return cube_mean, True

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
    plt.figure(figsize=(12, n_block*1.2), dpi=300)

    n_row = n_block
    n_col = 10

    for i in range(n_block):

        img_PET = np.rot90(inputs[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1))
        img_PET = (img_PET + 1) / 2

        img_CT = np.rot90(labels[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1))
        img_CT = (img_CT + 1) / 2

        img_pred = np.rot90(outputs[i, 0, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1))
        img_pred = (img_pred + 1) / 2

        yhat_x = img_pred - img_PET # -1 to 1
        yhat_x = (yhat_x + 1) / 2 # 0 to 1

        y_x = img_CT - img_PET
        y_x = (y_x + 1) / 2

        # first three and hist
        plt.subplot(n_row, n_col, i * n_col + 1)
        plt.imshow(img_PET, cmap="gray", vmin=0, vmax=0.5) # x
        # plt.title("input PET")
        if i == 0:
            plt.title("input STEP1")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 2)
        plt.imshow(img_CT, cmap="gray", vmin=0, vmax=0.5) # y
        # plt.title("label CT")
        if i == 0:
            plt.title("input STEP2")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 3)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(yhat_x, cmap="bwr", vmin=0.45, vmax=0.55) # yhat = f(x) + x, img_pred = f(x) = yhat - x
        # plt.title("output CT")
        plt.colorbar()
        if i == 0:
            plt.title("f(x)=yhat-x")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 4)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(y_x, cmap="bwr", vmin=0.45, vmax=0.55) # y = x + (y - x), (y - x) = y - x
        plt.colorbar()
        if i == 0:
            plt.title("gt=y-x")
        # plt.title("output CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 5)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(img_pred, cmap="gray", vmin=0, vmax=0.5) # yhat
        if i == 0:
            plt.title("yhat")
        # plt.title("output CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 6)
        # img_PET = np.clip(img_PET, 0, 1)
        plt.hist(img_PET.flatten(), bins=100)
        # plt.title("input PET")
        if i == 0:
            plt.title("input STEP1")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 7)
        # img_CT = np.clip(img_CT, 0, 1)
        plt.hist(img_CT.flatten(), bins=100)
        # plt.title("label CT")
        if i == 0:
            plt.title("input STEP2")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 8)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_pred-img_PET).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("f(x)=yhat-x")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(-1, 1)

        plt.subplot(n_row, n_col, i * n_col + 9)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_CT-img_PET).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("gt=y-x")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(-1, 1)

        plt.subplot(n_row, n_col, i * n_col + 10)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_pred).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("yhat")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(root_folder, f"epoch_{idx_epoch}.png"))
    plt.close()



# start the training
for idx_epoch in range(num_epoch):

    # train the model
    model.train()
    train_loss = 0
    valid_batch = 0
    for idx_batch, batch_data in enumerate(train_loader):
        if check_batch_cube_size(batch_data, cube_size) is False:
            print("The batch size is not correct")
            continue

        if check_whether_full_batch(batch_data) is False:
            print("The batch is not full")
            continue
        
        cube_mean, is_meaningful = check_whether_batch_meaningful(batch_data)
        if is_meaningful is False:
            print("The batch is not meaningful")
            print("The cube_mean is: ", cube_mean)

        valid_batch += 1
        inputs = batch_data["STEP1"].to(device)
        labels = batch_data["STEP2"].to(device)

        # for inputs and labels, clip the values to CT_MIN and CT_MAX
        # inputs = torch.clamp(inputs, CT_MIN, CT_MAX)
        # labels = torch.clamp(labels, CT_MIN, CT_MAX)
        # then normalize the values to 0 and 1
        # inputs = (inputs - CT_MIN) / CT_NORM
        # labels = (labels - CT_MIN) / CT_NORM
        # # 0 to 1 to -1 to 1
        # inputs = inputs * 2 - 1
        # labels = labels * 2 - 1
        res_inputs = torch.repeat_interleave(inputs, 2, dim=1).unsqueeze(2)
        # print("inputs.shape: ", inputs.shape, "labels.shape: ", labels.shape)
        # print("res_inputs.shape: ", res_inputs.shape)
        # print("inputs.shape: ", inputs.shape, "labels.shape: ", labels.shape)
        # inputs.shape:  torch.Size([2, 1, 128, 128, 128]) labels.shape:  torch.Size([2, 1, 128, 128, 128])
        # res_inputs.shape:  torch.Size([2, 2, 1, 128, 128, 128])
        # outputs.shape:  torch.Size([2, 2, 1, 128, 128, 128])
        optimizer.zero_grad()
        outputs = model(inputs)
        # print("outputs.shape: ", outputs.shape)
        outputs = outputs + res_inputs
        # loss = loss_function(outputs, labels)
        loss = ds_loss(torch.unbind(outputs, 1), labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {idx_epoch}, batch [{idx_batch}]/[{n_train_batches}], loss: {loss.item()*CT_NORM:.4f}")
        train_loss += loss.item()
    train_loss /= valid_batch
    print(f"Epoch {idx_epoch}, train_loss: {train_loss*CT_NORM:.4f}")
    # log the results
    with open(log_file, "a") as f:
        f.write(f"Epoch {idx_epoch}, train_loss: {train_loss*CT_NORM:.4f}\n")

    if idx_epoch % plot_per_epoch == 0:
        plot_results(inputs, labels, outputs, idx_epoch)

    # evaluate the model
    if idx_epoch % eval_per_epoch == 0:
        model.eval()
        valid_batch = 0
        with torch.no_grad():
            val_loss = 0
            for idx_batch, batch_data in enumerate(val_loader):
                if check_batch_cube_size(batch_data, cube_size) is False:
                    print("The batch size is not correct")
                    continue

                cube_mean, is_meaningful = check_whether_batch_meaningful(batch_data)
                if is_meaningful is False:
                    print("The batch is not meaningful")
                    print("The cube_mean is: ", cube_mean)

                valid_batch += 1
                inputs = batch_data["STEP1"].to(device)
                labels = batch_data["STEP2"].to(device)
                # inputs = torch.clamp(inputs, CT_MIN, CT_MAX)
                # labels = torch.clamp(labels, CT_MIN, CT_MAX)
                # inputs = (inputs - CT_MIN) / CT_NORM
                # labels = (labels - CT_MIN) / CT_NORM
                # inputs = inputs * 2 - 1
                # labels = labels * 2 - 1
                outputs = model(inputs)+inputs
                loss = output_loss(outputs, labels)
                val_loss += loss.item()
            val_loss /= valid_batch
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
                    valid_batch = 0
                    for idx_batch, batch_data in enumerate(test_loader):
                        if check_batch_cube_size(batch_data, cube_size) is False:
                            print("The batch size is not correct")
                            continue
                        
                        cube_mean, is_meaningful = check_whether_batch_meaningful(batch_data)
                        if is_meaningful is False:
                            print("The batch is not meaningful")
                            print("The cube_mean is: ", cube_mean)

                        valid_batch += 1
                        inputs = batch_data["STEP1"].to(device)
                        labels = batch_data["STEP2"].to(device)
                        # inputs = torch.clamp(inputs, CT_MIN, CT_MAX)
                        # labels = torch.clamp(labels, CT_MIN, CT_MAX)
                        # inputs = (inputs - CT_MIN) / CT_NORM
                        # labels = (labels - CT_MIN) / CT_NORM
                        # inputs = inputs * 2 - 1
                        # labels = labels * 2 - 1
                        outputs = model(inputs) + inputs
                        loss = output_loss(outputs, labels)
                        test_loss += loss.item()
                    test_loss /= valid_batch
                    print(f"Epoch {idx_epoch}, test_loss: {test_loss*CT_NORM:.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"Epoch {idx_epoch}, test_loss: {test_loss*CT_NORM:.4f}\n")

    # save the model
    if idx_epoch % save_per_epoch == 0:
        save_path = os.path.join(root_folder, f"model_epoch_{idx_epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Save model to {save_path}")

    

     