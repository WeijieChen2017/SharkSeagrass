
import json
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from einops import rearrange
from einops.layers.torch import Rearrange
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm


from monai.data import (
    DataLoader,
    CacheDataset,
)

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    RandSpatialCropd,
    # random flip and rotate
    RandFlipd,
    RandRotated,
)




class UNet3D_encoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()
        
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # input - down1 ------------- up1 -- output
        #         |                   |
        #         down2 ------------- up2
        #         |                   |
        #         down3 ------------- up3
        # 1 -> (32, 64, 128, 256) -> 1

        self.depth = len(self.channels)
        self.down_blocks = nn.ModuleList()


        for i in range(self.depth):
            self.down_blocks.append(
                ResidualUnit(3, self.in_channels, self.channels[i], self.strides[i],
                    kernel_size=self.kernel_size, subunits=self.num_res_units,
                    act=self.act, norm=self.norm, dropout=self.dropout,
                    bias=self.bias, adn_ordering=self.adn_ordering)
            )
            self.in_channels = self.channels[i]

        # flatten from (B, C, H, W, D) to (B, C, H*W*D), C is the self.channels[2]
        self.flatten = nn.Sequential(
            Rearrange('b c h w d -> b (h w d) c'),
        )

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.depth):
            x = self.down_blocks[i](x)
        
        x = self.flatten(x)

        return x
    

class UNet3D_decoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        hwd: Union[Tuple, str] = 8,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:

        super().__init__()

        self.dimensions = spatial_dims
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.hwd = hwd
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # input - down1 ------------- up1 -- output
        #         |                   |
        #         down2 ------------- up2
        #         |                   |
        #         down3 ------------- up3
        # 1 -> (32, 64, 128, 256) -> 1
        
        # take the cubic root of the second element of the tuple
        self.unflatten = nn.Sequential(
            Rearrange('b (h w d) c -> b c h w d', h=self.hwd, w=self.hwd, d=self.hwd),
        )

        self.depth = len(self.channels)
        self.up = nn.ModuleList()
        for i in range(self.depth - 1):
            self.up.append(
                nn.Sequential(
                    Convolution(3, self.channels[i], self.channels[i+1], self.strides[i], self.up_kernel_size,
                        act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, conv_only=False,
                        is_transposed=True, adn_ordering=self.adn_ordering),
                    ResidualUnit(3, self.channels[i+1], self.channels[i+1], 1, self.kernel_size, self.num_res_units,
                        act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias, last_conv_only=False,
                        adn_ordering=self.adn_ordering)
                )
            )
        self.out = nn.Conv3d(self.channels[-1], self.out_channels, kernel_size=1, stride=1, padding=0)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unflatten(x)
        for i in range(self.depth - 1):
            x = self.up[i](x)
        x = self.out(x)
        return x



def build_dataloader_train_val(batch_size: int, global_config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    pix_dim = global_config["pix_dim"]
    volume_size = global_config["volume_size"]
    cache_ratio_train = global_config["cache_ratio_train"]
    cache_ratio_val = global_config["cache_ratio_val"]
    num_workers_train_cache_dataset = global_config["num_workers_train_cache_dataset"]
    num_workers_val_cache_dataset = global_config["num_workers_val_cache_dataset"]
    num_workers_train_dataloader = global_config["num_workers_train_dataloader"]
    num_workers_val_dataloader = global_config["num_workers_val_dataloader"]
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(global_config.pix_dim, global_config.pix_dim, pix_dim),
                mode=("bilinear"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=2976,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # CropForegroundd(keys=["image"], source_key="image"),
            # random crop to the target size
            RandSpatialCropd(keys=["image"], roi_size=(volume_size, volume_size, volume_size), random_center=True, random_size=False),
            # add random flip and rotate
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandRotated(keys=["image"], prob=0.5, range_x=15, range_y=15, range_z=15),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(pix_dim, pix_dim, pix_dim),
                mode=("bilinear"),
            ),
            ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=2976, b_min=0.0, b_max=1.0, clip=True),
            # CropForegroundd(keys=["image"], source_key="image"),
            RandSpatialCropd(keys=["image"], roi_size=(volume_size, volume_size, volume_size), random_center=True, random_size=False),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        ]
    )

    # load data_chunks.json and specif chunk_0 to chunk_4 for training, chunk_5 to chunk_7 for validation, chunk_8 and chunk_9 for testing
    with open("data_chunks.json", "r") as f:
        data_chunk = json.load(f)

    train_files = []
    val_files = []
    test_files = []

    for i in range(5):
        train_files.extend(data_chunk[f"chunk_{i}"])
    for i in range(5, 8):
        val_files.extend(data_chunk[f"chunk_{i}"])
    for i in range(8, 10):
        test_files.extend(data_chunk[f"chunk_{i}"])

    num_train_files = len(train_files)
    num_val_files = len(val_files)
    num_test_files = len(test_files)

    print("Train files are ", len(train_files))
    print("Val files are ", len(val_files))
    print("Test files are ", len(test_files))


    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=num_train_files,
        cache_rate=cache_ratio_train, # 600 * 0.1 = 60
        num_workers=num_workers_train_cache_dataset,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms, 
        cache_num=num_val_files,
        cache_rate=cache_ratio_val, # 360 * 0.05 = 18
        num_workers=num_workers_val_cache_dataset,)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers_train_dataloader)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers_val_dataloader)

    return train_loader, val_loader

class simple_logger():
    def __init__(self, log_file_path, global_config):
        self.log_file_path = log_file_path
        self.log_dict = dict()
        self.IS_LOGGER_WANDB = global_config["IS_LOGGER_WANDB"]
        self.wandb_run = global_config["wandb_run"]
    
    def log(self, global_epoch, key, msg):
        if key not in self.log_dict.keys():
            self.log_dict[key] = dict()
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dict[key] = {
            "time": current_time,
            "global_epoch": global_epoch,
            "msg": msg
        }
        log_str = f"{current_time} Global epoch: {global_epoch}, {key}, {msg}\n"
        with open(self.log_file_path, "a") as f:
            f.write(log_str)
        print(log_str)

        # log to wandb if msg is number
        if self.IS_LOGGER_WANDB and isinstance(msg, (int, float)):
            self.wandb_run.log({key: msg})

def plot_and_save_x_xrec(x, xrec, num_per_direction=1, savename=None, wandb_name="val_snapshots", global_config=None):
    wandb_run = global_config["wandb_run"]
    numpy_x = x[0, :, :, :, :].cpu().numpy().squeeze()
    numpy_xrec = xrec[0, :, :, :, :].cpu().numpy().squeeze()
    x_clip = np.clip(numpy_x, 0, 1)
    rec_clip = np.clip(numpy_xrec, 0, 1)
    fig_width = num_per_direction * 3
    fig_height = 4
    fig, axs = plt.subplots(3, fig_width, figsize=(fig_width, fig_height), dpi=100)
    # for axial
    for i in range(num_per_direction):
        img_x = x_clip[x_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        img_rec = rec_clip[rec_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        axs[0, 3*i].imshow(img_x, cmap="gray")
        axs[0, 3*i].set_title(f"A x {x_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i].axis("off")
        axs[1, 3*i].imshow(img_rec, cmap="gray")
        axs[1, 3*i].set_title(f"A xrec {rec_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i].axis("off")
        axs[2, 3*i].imshow(img_x - img_rec, cmap="bwr")
        axs[2, 3*i].set_title(f"A diff {rec_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i].axis("off")
    # for sagittal
    for i in range(num_per_direction):
        img_x = x_clip[:, :, x_clip.shape[2]//(num_per_direction+1)*(i+1)]
        img_rec = rec_clip[:, :, rec_clip.shape[2]//(num_per_direction+1)*(i+1)]
        axs[0, 3*i+1].imshow(img_x, cmap="gray")
        axs[0, 3*i+1].set_title(f"S x {x_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i+1].axis("off")
        axs[1, 3*i+1].imshow(img_rec, cmap="gray")
        axs[1, 3*i+1].set_title(f"S xrec {rec_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i+1].axis("off")
        axs[2, 3*i+1].imshow(img_x - img_rec, cmap="bwr")
        axs[2, 3*i+1].set_title(f"S diff {rec_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i+1].axis("off")

    # for coronal
    for i in range(num_per_direction):
        img_x = x_clip[:, x_clip.shape[1]//(num_per_direction+1)*(i+1), :]
        img_rec = rec_clip[:, rec_clip.shape[1]//(num_per_direction+1)*(i+1), :]
        axs[0, 3*i+2].imshow(img_x, cmap="gray")
        axs[0, 3*i+2].set_title(f"C x {x_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i+2].axis("off")
        axs[1, 3*i+2].imshow(img_rec, cmap="gray")
        axs[1, 3*i+2].set_title(f"C xrec {rec_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i+2].axis("off")
        axs[2, 3*i+2].imshow(img_x - img_rec, cmap="bwr")
        axs[2, 3*i+2].set_title(f"C diff {rec_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i+2].axis("off")

    plt.tight_layout()
    plt.savefig(savename)
    wandb_run.log({wandb_name: fig})
    plt.close()
    print(f"Save the plot to {savename}")

def compute_average_gradient(model_part):
    total_grad = 0.0
    num_params = 0
    for param in model_part.parameters():
        if param.grad is not None:
            total_grad += param.grad.abs().mean().item()
            num_params += 1
    return total_grad / num_params if num_params > 0 else 0.0

# Effective Number of Classes
def effective_number_of_classes(probabilities):
    return 1 / np.sum(probabilities ** 2)