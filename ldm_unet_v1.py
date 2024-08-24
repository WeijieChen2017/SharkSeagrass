gpu_list = ','.join(str(x) for x in [1])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
# import torch.nn.functional as F

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
import pytorch_lightning as pl

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
  def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                dropout, temb_channels=512):
      super().__init__()
      self.in_channels = in_channels
      out_channels = in_channels if out_channels is None else out_channels
      self.out_channels = out_channels
      self.use_conv_shortcut = conv_shortcut

      self.norm1 = Normalize(in_channels)
      self.conv1 = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
      if temb_channels > 0:
          self.temb_proj = torch.nn.Linear(temb_channels,
                                            out_channels)
      self.norm2 = Normalize(out_channels)
      self.dropout = torch.nn.Dropout(dropout)
      self.conv2 = torch.nn.Conv2d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
      if self.in_channels != self.out_channels:
          if self.use_conv_shortcut:
              self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1)
          else:
              self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0)

  def forward(self, x, temb):
      h = x
      h = self.norm1(h)
      h = nonlinearity(h)
      h = self.conv1(h)

      if temb is not None:
          h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

      h = self.norm2(h)
      h = nonlinearity(h)
      h = self.dropout(h)
      h = self.conv2(h)

      if self.in_channels != self.out_channels:
          if self.use_conv_shortcut:
              x = self.conv_shortcut(x)
          else:
              x = self.nin_shortcut(x)

      return x+h
  

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
    

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_
        
def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)
    

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()

        self.h_maps = []  # Store intermediate feature maps

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None
        intermediate_maps = []

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            intermediate_maps.append(h) # Store feature maps
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, intermediate_maps



class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)


        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, enc_intermediate_maps):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # Concatenate with corresponding encoder feature map
            h = torch.cat([h, enc_intermediate_maps[i_level]], dim=1)
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h



class VQModel(pl.LightningModule):
# class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)
        
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.out_conv = torch.nn.Conv2d(ddconfig["out_ch"], 1, 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        h, in_maps = self.encoder(x)
        # print(f"Shape checking 2, h: {h.shape}")
        h = self.quant_conv(h)
        # print(f"Shape checking 3, h: {h.shape} after quant_conv")
        # print(f"Shape checking 4, quant: {quant.shape} after quantize")
        return h, in_maps

    def decode(self, quant, in_maps):
        quant = self.post_quant_conv(quant)
        # print(f"Shape checking 5, quant: {quant.shape} after post_quant_conv")
        dec = self.decoder(quant, in_maps)
        # print(f"Shape checking 6, dec: {dec.shape}")
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        
        # print(f"Shape checking 1, input: {input.shape}")

        quant, in_maps = self.encode(input)
        dec = self.decode(quant, in_maps)
        out = self.out_conv(dec)
        return out


VQ_NAME = "f4-noattn"

# load the configuration yaml files
import os
import yaml

config_yaml_path = f"ldm_models/first_stage_models/vq-{VQ_NAME}/config.yaml"
with open(config_yaml_path, 'r') as file:
    config = yaml.safe_load(file)

print(config)

ckpt_path = f"vq_{VQ_NAME}.ckpt"

dd_config = config['model']['params']['ddconfig']
loss_config = config['model']['params']['lossconfig']

model = VQModel(ddconfig=dd_config,
                n_embed=config['model']['params']['n_embed'],
                embed_dim=config['model']['params']['embed_dim'],
                ckpt_path=ckpt_path,
                ignore_keys=[],
                image_key="image",
).to(device)


import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
)
from monai.data import CacheDataset, DataLoader

input_modality = ["PET", "CT"]
img_size = 400
cube_size = 64
in_channels = 3
out_channels = 1
batch_size = 16
num_epoch = 10000
save_per_epoch = 10
eval_per_epoch = 1
plot_per_epoch = 1
CT_NORM = 5000
root_folder = "./B100/ldm_unet_v1"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
print("The root folder is: ", root_folder)
log_file = os.path.join(root_folder, "log.txt")


# set the data transform
train_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        EnsureChannelFirstd(keys="PET", channel_dim=-1),
        EnsureChannelFirstd(keys="CT", channel_dim='no_channel'),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        EnsureChannelFirstd(keys="PET", channel_dim=-1),
        EnsureChannelFirstd(keys="CT", channel_dim='no_channel'),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=input_modality, image_only=True),
        EnsureChannelFirstd(keys="PET", channel_dim=-1),
        EnsureChannelFirstd(keys="CT", channel_dim='no_channel'),
    ]
)

data_division_file = "./B100/B100_0822_2d3c.json"
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
    cache_rate=0.1,
    num_workers=4,
)

val_ds = CacheDataset(
    data=val_list,
    transform=val_transforms, 
    cache_num=num_val_files,
    cache_rate=0.1,
    num_workers=4,
)

test_ds = CacheDataset(
    data=test_list,
    transform=test_transforms,
    cache_num=num_test_files,
    cache_rate=0.1,
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

best_val_loss = 1e10
n_train_batches = len(train_loader)
n_val_batches = len(val_loader)
n_test_batches = len(test_loader)


def plot_results(inputs, labels, outputs, idx_epoch):
    # plot the results
    n_block = 8
    plt.figure(figsize=(12, 12), dpi=300)

    n_row = n_block
    n_col = 6

    for i in range(n_block):
        # first three and hist
        plt.subplot(n_row, n_col, i * n_col + 1)
        img_PET = np.rot90(inputs[i, in_channels // 2, :, :].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, 0, 1))
        plt.imshow(img_PET, cmap="gray")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 2)
        img_CT = np.rot90(labels[i, 0, :, :].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, 0, 1))
        plt.imshow(img_CT, cmap="gray")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 3)
        img_pred = np.rot90(outputs[i, 0, 0, :, :].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, 0, 1))
        plt.imshow(img_pred, cmap="gray")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 4)
        plt.hist(img_PET.flatten(), bins=100)
        plt.yscale("log")
        plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 5)
        plt.hist(img_CT.flatten(), bins=100)
        plt.yscale("log")
        plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 6)
        plt.hist(img_pred.flatten(), bins=100)
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
        # inputs.shape:  torch.Size([16, 3, 400, 400]) labels.shape:  torch.Size([16, 1, 400, 400])
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = output_loss(outputs, labels)
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

    
