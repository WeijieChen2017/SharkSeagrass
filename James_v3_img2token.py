import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
import pytorch_lightning as pl
from ldm.util import instantiate_from_config

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

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
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
        return h



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

    def forward(self, z):
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
            for i_block in range(self.num_res_blocks+1):
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
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

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

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


VQ_NAME = "f4"

# load the configuration yaml files

import yaml

config_yaml_path = f"models/first_stage_models/vq-{VQ_NAME}/config.yaml"
with open(config_yaml_path, 'r') as file:
    config = yaml.safe_load(file)

print(config)

ckpt_path = "vq_{VQ_NAME}.ckpt"

dd_config = config['model']['params']['ddconfig']
loss_config = config['model']['params']['lossconfig']

model = VQModel(ddconfig=dd_config,
                lossconfig=loss_config,
                n_embed=config['model']['params']['n_embed'],
                embed_dim=config['model']['params']['embed_dim'],
                ckpt_path=ckpt_path,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                batch_resize_range=None,
                scheduler_config=None,
                lr_g_factor=1.0,
                remap=None,
                sane_index_shape=False, # tell vector quantizer to return indices as bhw
)

# load ckpt_path and show all keys
sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
keys = list(sd.keys())

for k in keys:
    print(k)

print("<" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The current device is", device)
model.to(device)

def plot_images(savename, CTr_img, PET_img, return_CTr, return_PET, ind_CTr, ind_PET):

    # 1st plot for recon
    fig = plt.figure(figsize=(12, 16), dpi=100)

    plt.subplot(4, 3, 1)
    img = np.rot90(CTr_img[1, :, :])
    # img = CTr_img[1, :, :]
    plt.imshow(img, cmap='gray')
    plt.title('CTr')
    plt.axis('off')

    plt.subplot(4, 3, 2)
    img_rCTr = np.rot90(return_CTr[0, 1, :, :])
    # img_rCTr = return_CTr[0, 1, :, :]
    # clip img from -1 to 1
    img_rCTr = np.clip(img_rCTr, -1, 1)
    plt.imshow(img_rCTr, cmap='gray')
    n_unique_CTr = torch.unique(ind_CTr).shape[0]
    plt.title(f"CTr_recon via {n_unique_CTr} embedding")
    plt.axis('off')

    plt.subplot(4, 3, 3)
    img = img_rCTr - np.rot90(CTr_img[1, :, :])
    # img = img_rCTr - CTr_img[1, :, :]
    plt.imshow(img, cmap='bwr')
    plt.title('diff_CTr')
    plt.axis('off')

    plt.subplot(4, 3, 7)
    img = np.rot90(PET_img[1, :, :])
    # img = PET_img[1, :, :]
    plt.imshow(img, cmap='gray')
    plt.title('PET')
    plt.axis('off')

    plt.subplot(4, 3, 8)
    img_rPET = np.rot90(return_PET[0, 1, :, :])
    # img_rPET = return_PET[0, 1, :, :]
    # clip img from -1 to 1
    img_rPET = np.clip(img_rPET, -1, 1)
    plt.imshow(img_rPET, cmap='gray')
    n_unique_PET = torch.unique(ind_PET).shape[0]
    plt.title(f"PET_recon via {n_unique_PET} embedding")
    plt.axis('off')

    plt.subplot(4, 3, 9)
    img = img_rPET - np.rot90(PET_img[1, :, :])
    # img = img_rPET - PET_img[1, :, :]
    plt.imshow(img, cmap='bwr')
    plt.title('diff_PET')
    plt.axis('off')

    plt.subplot(4, 3, 4)
    img = np.rot90(CTr_img[1, :, :])
    # img = CTr_img[1, :, :]
    plt.hist(img.flatten(), bins=100)
    plt.xlim(-1, 1)
    plt.title('CTr')
    plt.yscale('log')

    plt.subplot(4, 3, 5)
    img_rCTr = np.rot90(return_CTr[0, 1, :, :])
    # img_rCTr = return_CTr[0, 1, :, :]
    # clip img from -1 to 1
    img_rCTr = np.clip(img_rCTr, -1, 1)
    plt.hist(img_rCTr.flatten(), bins=100)
    plt.xlim(-1, 1)
    n_unique_CTr = torch.unique(ind_CTr).shape[0]
    plt.title(f"CTr_recon via {n_unique_CTr} embedding")
    plt.yscale('log')

    plt.subplot(4, 3, 6)
    img = img_rCTr - np.rot90(CTr_img[1, :, :])
    # img = img_rCTr - CTr_img[1, :, :]
    plt.hist(img.flatten(), bins=100)
    plt.xlim(-1, 1)
    plt.title('diff_CTr')
    plt.yscale('log')

    plt.subplot(4, 3, 10)
    img = np.rot90(PET_img[1, :, :])
    # img = PET_img[1, :, :]
    plt.hist(img.flatten(), bins=100)
    plt.xlim(-1, 1)
    plt.title('PET')
    plt.yscale('log')

    plt.subplot(4, 3, 11)
    img_rPET = np.rot90(return_PET[0, 1, :, :])
    # img_rPET = return_PET[0, 1, :, :]
    # clip img from -1 to 1
    img_rPET = np.clip(img_rPET, -1, 1)
    plt.hist(img_rPET.flatten(), bins=100)
    plt.xlim(-1, 1)
    n_unique_PET = torch.unique(ind_PET).shape[0]
    plt.title(f"PET_recon via {n_unique_PET} embedding")
    plt.yscale('log')

    plt.subplot(4, 3, 12)
    img = img_rPET - np.rot90(PET_img[1, :, :])
    # img = img_rPET - PET_img[1, :, :]
    plt.hist(img.flatten(), bins=100)
    plt.xlim(-1, 1)
    plt.title('diff_PET')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def two_segment_scale(arr, MIN, MID, MAX, MIQ):
    # Create an empty array to hold the scaled results
    scaled_arr = np.zeros_like(arr, dtype=np.float32)

    # First segment: where arr <= MID
    mask1 = arr <= MID
    scaled_arr[mask1] = (arr[mask1] - MIN) / (MID - MIN) * MIQ

    # Second segment: where arr > MID
    mask2 = arr > MID
    scaled_arr[mask2] = MIQ + (arr[mask2] - MID) / (MAX - MID) * (1 - MIQ)
    
    return scaled_arr

def reverse_two_segment_scale(arr, MIN, MID, MAX, MIQ):
    # Create an empty array to hold the reverse scaled results
    reverse_scaled_arr = np.zeros_like(arr, dtype=np.float32)

    # First segment: where arr <= MIQ
    mask1 = arr <= MIQ
    reverse_scaled_arr[mask1] = arr[mask1] * (MID - MIN) / MIQ + MIN

    # Second segment: where arr > MIQ
    mask2 = arr > MIQ
    reverse_scaled_arr[mask2] = MID + (arr[mask2] - MIQ) * (MAX - MID) / (1 - MIQ)
    
    return reverse_scaled_arr


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

tag_list = ['E4055', 'E4058', 'E4061', 'E4066', 'E4068',
            'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
            'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
            'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
            'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
            'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
            'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
            'E4139', 'E4143', 'E4144', 'E4147', 'E4152',
            'E4155', 'E4157', 'E4158', 'E4162', 'E4163',
            'E4165', 'E4166', 'E4172', 'E4181', 'E4182',
            'E4183', 'E4185', 'E4187', 'E4189', 'E4193',
            'E4197', 'E4198', 'E4207', 'E4208', 'E4216',
            'E4217', 'E4219', 'E4220', 'E4232', 'E4237',
            'E4238', 'E4239', 'E4241'
]
total_file_list = len(tag_list)

MAX_CT = 2976
MIN_CT = -1024

MID_PET = 5000
MIQ_PET = 0.9
MAX_PET = 20000
MIN_PET = 0
RANGE_CT = MAX_CT - MIN_CT
RANGE_PET = MAX_PET - MIN_PET

root_folder = "James_data_v3/"

n_cut = 8
zoom_factors = [256/512, 256/512, 1]

done_axial = False
done_coronal = False
done_sagittal = False

for idx_tag, name_tag in enumerate(tag_list):

    print()
    print(f"Processing {name_tag}:[{idx_tag}]/[{len(total_file_list)}]")

    CTAC_path = root_folder + f"CTACIVV_256_norm/CTACIVV_{name_tag}_norm.nii.gz"
    TOFNAC_path = root_folder + f"TOFNAC_256_norm/TOFNAC_{name_tag}_norm.nii.gz"

    CTAC_file = nib.load(CTAC_path)
    TOFNAC_file = nib.load(TOFNAC_path)

    CTAC_data = CTAC_file.get_fdata()
    TOFNAC_data = TOFNAC_file.get_fdata()

    # # clip to [0, 1]
    # CTAC_data = np.clip(CTAC_data, MIN_CT, MAX_CT)
    # TOFNAC_data = np.clip(TOFNAC_data, MIN_PET, MAX_PET)

    # normalize the CT and PET data
    CTAC_data = (CTAC_data - 0.5) * 2 # [0 -> 1] to [-1 -> 1]
    TOFNAC_data = (TOFNAC_data - 0.5) * 2

    # convert the img to be channel first, from 256, 256, 3 to 3, 256, 256
    # CT_res_data = np.moveaxis(CT_res_data, -1, 0)
    # PET_data = np.moveaxis(PET_data, -1, 0)
    # print(f"CT_res_data shape: {CT_res_data.shape}, PET_data shape: {PET_data.shape}")

    len_z = TOFNAC_data.shape[2]
    
    if "4" in VQ_NAME:
        len_factor = 4
    elif "8" in VQ_NAME:
        len_factor = 8
    elif "16" in VQ_NAME:
        len_factor = 16
    else:
        ValueError("VQ_NAME should be 4, 8, or 16")

    
    if len_z % len_factor != 0:
        # pad it to the nearest multiple of 4 at the end
        print(f"Padding the z-axis to the nearest multiple of {len_factor}")
        pad_len = len_factor - len_z % len_factor
        TOFNAC_data = np.pad(TOFNAC_data, ((0, 0), (0, 0), (0, pad_len)), mode="constant", constant_values=0)
        CTAC_data = np.pad(CTAC_data, ((0, 0), (0, 0), (0, pad_len)), mode="constant", constant_values=0)

    print(f"{name_tag} -> TOFNAC shape: {TOFNAC_data.shape}, CTAC shape: {CTAC_data.shape}")
    
    len_x, len_y, len_z = TOFNAC_data.shape
    x_axial_mae_list = []
    x_coronal_mae_list = []
    x_sagittal_mae_list = []
    y_axial_mae_list = []   
    y_coronal_mae_list = []
    y_sagittal_mae_list = []
    x_axial_ind_list = []
    x_coronal_ind_list = []
    x_sagittal_ind_list = []
    y_axial_ind_list = []
    y_coronal_ind_list = []
    y_sagittal_ind_list = []

    x_axial_recon = np.zeros((len_x, len_y, len_z))
    x_coronal_recon = np.zeros((len_x, len_y, len_z))
    x_sagittal_recon = np.zeros((len_x, len_y, len_z))

    y_axial_recon = np.zeros((len_x, len_y, len_z))
    y_coronal_recon = np.zeros((len_x, len_y, len_z))
    y_sagittal_recon = np.zeros((len_x, len_y, len_z))

    if not done_axial:
        # for axial
        for idx_z in range(TOFNAC_data.shape[2]):
            if idx_z == 0:
                slice_1 = TOFNAC_data[:, :, idx_z]
                slice_2 = TOFNAC_data[:, :, idx_z]
                slice_3 = TOFNAC_data[:, :, idx_z+1]
                slice_1 = np.expand_dims(slice_1, axis=2)
                slice_2 = np.expand_dims(slice_2, axis=2)
                slice_3 = np.expand_dims(slice_3, axis=2)
                data_x = np.concatenate([slice_1, slice_2, slice_3], axis=2)

                slice_1 = CTAC_data[:, :, idx_z]
                slice_2 = CTAC_data[:, :, idx_z]
                slice_3 = CTAC_data[:, :, idx_z+1]
                slice_1 = np.expand_dims(slice_1, axis=2)
                slice_2 = np.expand_dims(slice_2, axis=2)
                slice_3 = np.expand_dims(slice_3, axis=2)
                data_y = np.concatenate([slice_1, slice_2, slice_3], axis=2)
            elif idx_z == TOFNAC_data.shape[2] - 1:
                slice_1 = TOFNAC_data[:, :, idx_z-1]
                slice_2 = TOFNAC_data[:, :, idx_z]
                slice_3 = TOFNAC_data[:, :, idx_z]
                slice_1 = np.expand_dims(slice_1, axis=2)
                slice_2 = np.expand_dims(slice_2, axis=2)
                slice_3 = np.expand_dims(slice_3, axis=2)
                data_x = np.concatenate([slice_1, slice_2, slice_3], axis=2)

                slice_1 = CTAC_data[:, :, idx_z-1]
                slice_2 = CTAC_data[:, :, idx_z]
                slice_3 = CTAC_data[:, :, idx_z]
                slice_1 = np.expand_dims(slice_1, axis=2)
                slice_2 = np.expand_dims(slice_2, axis=2)
                slice_3 = np.expand_dims(slice_3, axis=2)
                data_y = np.concatenate([slice_1, slice_2, slice_3], axis=2)
            else:
                data_x = TOFNAC_data[:, :, idx_z-1:idx_z+2]
                data_y = CTAC_data[:, :, idx_z-1:idx_z+2]
            # data_x is 400x400x3, convert it to 1x3x256x256
            data_x = np.transpose(data_x, (2, 0, 1))
            data_x = np.expand_dims(data_x, axis=0)
            data_x = torch.tensor(data_x, dtype=torch.float32).to(device)

            # data_y is 400x400x3, convert it to 1x3x256x256
            data_y = np.transpose(data_y, (2, 0, 1))
            data_y = np.expand_dims(data_y, axis=0)
            data_y = torch.tensor(data_y, dtype=torch.float32).to(device)
            with torch.no_grad():
                return_x, _, ind_x = model(data_x, return_pred_indices=True)
                return_y, _, ind_y = model(data_y, return_pred_indices=True)
            
            x_axial_ind_list.append(ind_x.detach().cpu().numpy())
            y_axial_ind_list.append(ind_y.detach().cpu().numpy())
            recon_x = return_x.detach().cpu().numpy()
            recon_y = return_y.detach().cpu().numpy()
            # move the channel dim to the last
            recon_x = np.squeeze(recon_x)[1, :, :]
            recon_y = np.squeeze(recon_y)[1, :, :]
            gt_x = TOFNAC_data[:, :, idx_z]
            gt_y = CTAC_data[:, :, idx_z]
            # [-1 to 1] -> [0 to 1]
            recon_x = np.clip(recon_x, -1, 1)
            recon_y = np.clip(recon_y, -1, 1)
            recon_x = (recon_x + 1) / 2
            recon_y = (recon_y + 1) / 2
            gt_x = (gt_x + 1) / 2
            gt_y = (gt_y + 1) / 2
            # reverse the scale
            recon_x = reverse_two_segment_scale(recon_x, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
            gt_x = reverse_two_segment_scale(gt_x, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
            recon_y = recon_y * RANGE_CT + MIN_CT
            gt_y = gt_y * RANGE_CT + MIN_CT
            # save the recon
            x_axial_recon[:, :, idx_z] = recon_x
            y_axial_recon[:, :, idx_z] = recon_y
            # compute the l1 loss
            x_axial_mae = np.mean(np.abs(gt_x - recon_x))
            y_axial_mae = np.mean(np.abs(gt_y - recon_y))
            x_axial_mae_list.append(x_axial_mae)
            y_axial_mae_list.append(y_axial_mae)
            print(f">> Processing {name_tag}:[{idx_tag}]/[{len(total_file_list)}] z=[{idx_z}]/[{len_z-1}], x_axial_mae: {x_axial_mae:.3f}, y_axial_mae: {y_axial_mae:.3f}")

        # print the mae
        x_axial_mae_list = np.array(x_axial_mae_list)
        y_axial_mae_list = np.array(y_axial_mae_list)
        print(f"x_axial_mae mean: {np.mean(x_axial_mae_list)}, std: {np.std(x_axial_mae_list)}")
        print(f"y_axial_mae mean: {np.mean(y_axial_mae_list)}, std: {np.std(y_axial_mae_list)}")

        # save the mae
        x_axial_mae_savename = f"{root_folder}{name_tag}_x_axial_mae.npy"
        y_axial_mae_savename = f"{root_folder}{name_tag}_y_axial_mae.npy"
        np.save(x_axial_mae_savename, x_axial_mae_list)
        np.save(y_axial_mae_savename, y_axial_mae_list)
        print(f"Save {x_axial_mae_savename} and {y_axial_mae_savename}")

        # save the indices
        x_axial_ind = np.array(x_axial_ind_list)
        y_axial_ind = np.array(y_axial_ind_list)
        x_axial_ind_savename = f"{root_folder}{name_tag}_x_axial_ind.npy"
        y_axial_ind_savename = f"{root_folder}{name_tag}_y_axial_ind.npy"
        np.save(x_axial_ind_savename, x_axial_ind_list)
        np.save(y_axial_ind_savename, y_axial_ind_list)
        print(f"Save {x_axial_ind_savename} and {y_axial_ind_savename}")

        # save the recon
        x_axial_recon_savename = f"{root_folder}{name_tag}_x_axial_recon.npy"
        y_axial_recon_savename = f"{root_folder}{name_tag}_y_axial_recon.npy"
        np.save(x_axial_recon_savename, x_axial_recon)
        np.save(y_axial_recon_savename, y_axial_recon)
        print(f"Save {x_axial_recon_savename} and {y_axial_recon_savename}")

    if not done_coronal:

        # for axial
        for idx_y in range(TOFNAC_data.shape[1]):
            if idx_y == 0:
                slice_1 = TOFNAC_data[:, idx_y, :]
                slice_2 = TOFNAC_data[:, idx_y, :]
                slice_3 = TOFNAC_data[:, idx_y+1, :]
                slice_1 = np.expand_dims(slice_1, axis=1)
                slice_2 = np.expand_dims(slice_2, axis=1)
                slice_3 = np.expand_dims(slice_3, axis=1)
                data_x = np.concatenate([slice_1, slice_2, slice_3], axis=1)

                slice_1 = CTAC_data[:, idx_y, :]
                slice_2 = CTAC_data[:, idx_y, :]
                slice_3 = CTAC_data[:, idx_y+1, :]
                slice_1 = np.expand_dims(slice_1, axis=1)
                slice_2 = np.expand_dims(slice_2, axis=1)
                slice_3 = np.expand_dims(slice_3, axis=1)
                data_y = np.concatenate([slice_1, slice_2, slice_3], axis=1)
            elif idx_y == TOFNAC_data.shape[1] - 1:
                slice_1 = TOFNAC_data[:, idx_y-1, :]
                slice_2 = TOFNAC_data[:, idx_y, :]
                slice_3 = TOFNAC_data[:, idx_y, :]
                slice_1 = np.expand_dims(slice_1, axis=1)
                slice_2 = np.expand_dims(slice_2, axis=1)
                slice_3 = np.expand_dims(slice_3, axis=1)
                data_x = np.concatenate([slice_1, slice_2, slice_3], axis=1)

                slice_1 = CTAC_data[:, idx_y-1, :]
                slice_2 = CTAC_data[:, idx_y, :]
                slice_3 = CTAC_data[:, idx_y, :]
                slice_1 = np.expand_dims(slice_1, axis=1)
                slice_2 = np.expand_dims(slice_2, axis=1)
                slice_3 = np.expand_dims(slice_3, axis=1)
                data_y = np.concatenate([slice_1, slice_2, slice_3], axis=1)
            else:
                data_x = TOFNAC_data[:, idx_y-1:idx_y+2, :]
                data_y = CTAC_data[:, idx_y-1:idx_y+2, :]
            # data_x is 256, 3, 720, convert it to 1, 3, 720, 256
            data_x = np.transpose(data_x, (1, 2, 0))
            data_x = np.expand_dims(data_x, axis=0)
            data_x = torch.tensor(data_x, dtype=torch.float32).to(device)

            data_y = np.transpose(data_y, (1, 2, 0))
            data_y = np.expand_dims(data_y, axis=0)
            data_y = torch.tensor(data_y, dtype=torch.float32).to(device)

            with torch.no_grad():
                return_x, _, ind_x = model(data_x, return_pred_indices=True)
                return_y, _, ind_y = model(data_y, return_pred_indices=True)
            
            x_coronal_ind_list.append(ind_x.detach().cpu().numpy())
            y_coronal_ind_list.append(ind_y.detach().cpu().numpy())
            recon_x = return_x.detach().cpu().numpy()
            recon_y = return_y.detach().cpu().numpy()
            # move the channel dim to the last
            recon_x = np.squeeze(recon_x)[1, :, :]
            recon_y = np.squeeze(recon_y)[1, :, :]
            gt_x = TOFNAC_data[:, idx_y, :]
            gt_y = CTAC_data[:, idx_y, :]

            # [-1 to 1] -> [0 to 1]
            recon_x = np.clip(recon_x, -1, 1)
            recon_y = np.clip(recon_y, -1, 1)
            recon_x = (recon_x + 1) / 2
            recon_y = (recon_y + 1) / 2

            gt_x = (gt_x + 1) / 2
            gt_y = (gt_y + 1) / 2
            
            # reverse the scale
            recon_x = reverse_two_segment_scale(recon_x, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
            gt_x = reverse_two_segment_scale(gt_x, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
            recon_y = recon_y * RANGE_CT + MIN_CT
            gt_y = gt_y * RANGE_CT + MIN_CT

            # save the recon
            x_coronal_recon[:, idx_y, :] = recon_x
            y_coronal_recon[:, idx_y, :] = recon_y

            # compute the l1 loss
            recon_x = np.transpose(recon_x, (1, 0))
            recon_y = np.transpose(recon_y, (1, 0))
            x_coronal_mae = np.mean(np.abs(gt_x - recon_x))
            y_coronal_mae = np.mean(np.abs(gt_y - recon_y))

            x_coronal_mae_list.append(x_coronal_mae)
            y_coronal_mae_list.append(y_coronal_mae)

            print(f">> Processing {name_tag}:[{idx_tag}]/[{len(total_file_list)}] y=[{idx_y}]/[{len_y-1}], x_coronal_mae: {x_coronal_mae:.3f}, y_coronal_mae: {y_coronal_mae:.3f}")

        # print the mae
        x_coronal_mae_list = np.array(x_coronal_mae_list)
        y_coronal_mae_list = np.array(y_coronal_mae_list)
        print(f"x_coronal_mae mean: {np.mean(x_coronal_mae_list)}, std: {np.std(x_coronal_mae_list)}")
        print(f"y_coronal_mae mean: {np.mean(y_coronal_mae_list)}, std: {np.std(y_coronal_mae_list)}")

        # save the mae
        x_coronal_mae_savename = f"{root_folder}{name_tag}_x_coronal_mae.npy"
        y_coronal_mae_savename = f"{root_folder}{name_tag}_y_coronal_mae.npy"
        np.save(x_coronal_mae_savename, x_coronal_mae_list)
        np.save(y_coronal_mae_savename, y_coronal_mae_list)
        print(f"Save {x_coronal_mae_savename} and {y_coronal_mae_savename}")

        # save the indices
        x_coronal_ind = np.array(x_coronal_ind_list)
        y_coronal_ind = np.array(y_coronal_ind_list)
        x_coronal_ind_savename = f"{root_folder}{name_tag}_x_coronal_ind.npy"
        y_coronal_ind_savename = f"{root_folder}{name_tag}_y_coronal_ind.npy"
        np.save(x_coronal_ind_savename, x_coronal_ind_list)
        np.save(y_coronal_ind_savename, y_coronal_ind_list)
        print(f"Save {x_coronal_ind_savename} and {y_coronal_ind_savename}")

        # save the recon
        x_coronal_recon_savename = f"{root_folder}{name_tag}_x_coronal_recon.npy"
        y_coronal_recon_savename = f"{root_folder}{name_tag}_y_coronal_recon.npy"
        np.save(x_coronal_recon_savename, x_coronal_recon)
        np.save(y_coronal_recon_savename, y_coronal_recon)
        print(f"Save {x_coronal_recon_savename} and {y_coronal_recon_savename}")

    if not done_sagittal:
        
        for idx_x in range(TOFNAC_data.shape[0]):
            if idx_x == 0:
                slice_1 = TOFNAC_data[idx_x, :, :]
                slice_2 = TOFNAC_data[idx_x, :, :]
                slice_3 = TOFNAC_data[idx_x+1, :, :]
                slice_1 = np.expand_dims(slice_1, axis=0)
                slice_2 = np.expand_dims(slice_2, axis=0)
                slice_3 = np.expand_dims(slice_3, axis=0)
                data_x = np.concatenate([slice_1, slice_2, slice_3], axis=0)

                slice_1 = CTAC_data[idx_x, :, :]
                slice_2 = CTAC_data[idx_x, :, :]
                slice_3 = CTAC_data[idx_x+1, :, :]
                slice_1 = np.expand_dims(slice_1, axis=0)
                slice_2 = np.expand_dims(slice_2, axis=0)
                slice_3 = np.expand_dims(slice_3, axis=0)
                data_y = np.concatenate([slice_1, slice_2, slice_3], axis=0)
            elif idx_x == TOFNAC_data.shape[0] - 1:
                slice_1 = TOFNAC_data[idx_x-1, :, :]
                slice_2 = TOFNAC_data[idx_x, :, :]
                slice_3 = TOFNAC_data[idx_x, :, :]
                slice_1 = np.expand_dims(slice_1, axis=0)
                slice_2 = np.expand_dims(slice_2, axis=0)
                slice_3 = np.expand_dims(slice_3, axis=0)
                data_x = np.concatenate([slice_1, slice_2, slice_3], axis=0)

                slice_1 = CTAC_data[idx_x-1, :, :]
                slice_2 = CTAC_data[idx_x, :, :]
                slice_3 = CTAC_data[idx_x, :, :]
                slice_1 = np.expand_dims(slice_1, axis=0)
                slice_2 = np.expand_dims(slice_2, axis=0)
                slice_3 = np.expand_dims(slice_3, axis=0)
                data_y = np.concatenate([slice_1, slice_2, slice_3], axis=0)
            else:
                data_x = TOFNAC_data[idx_x-1:idx_x+2, :, :]
                data_y = CTAC_data[idx_x-1:idx_x+2, :, :]
            # data_x is 3, 256, 720, convert it to 1, 3, 720, 256

            data_x = np.transpose(data_x, (0, 2, 1))
            data_x = np.expand_dims(data_x, axis=0)
            data_x = torch.tensor(data_x, dtype=torch.float32).to(device)

            data_y = np.transpose(data_y, (0, 2, 1))
            data_y = np.expand_dims(data_y, axis=0)
            data_y = torch.tensor(data_y, dtype=torch.float32).to(device)

            with torch.no_grad():
                return_x, _, ind_x = model(data_x, return_pred_indices=True)
                return_y, _, ind_y = model(data_y, return_pred_indices=True)

            x_sagittal_ind_list.append(ind_x.detach().cpu().numpy())
            y_sagittal_ind_list.append(ind_y.detach().cpu().numpy())
            recon_x = return_x.detach().cpu().numpy()
            recon_y = return_y.detach().cpu().numpy()

            # move the channel dim to the last
            recon_x = np.squeeze(recon_x)[1, :, :]
            recon_y = np.squeeze(recon_y)[1, :, :]
            gt_x = TOFNAC_data[idx_x, :, :]
            gt_y = CTAC_data[idx_x, :, :]

            # [-1 to 1] -> [0 to 1]
            recon_x = np.clip(recon_x, -1, 1)
            recon_y = np.clip(recon_y, -1, 1)
            recon_x = (recon_x + 1) / 2
            recon_y = (recon_y + 1) / 2

            gt_x = (gt_x + 1) / 2
            gt_y = (gt_y + 1) / 2

            # reverse the scale
            recon_x = reverse_two_segment_scale(recon_x, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
            gt_x = reverse_two_segment_scale(gt_x, MIN_PET, MID_PET, MAX_PET, MIQ_PET)
            recon_y = recon_y * RANGE_CT + MIN_CT
            gt_y = gt_y * RANGE_CT + MIN_CT

            # save the recon
            x_sagittal_recon[idx_x, :, :] = recon_x
            y_sagittal_recon[idx_x, :, :] = recon_y

            # compute the l1 loss
            recon_x = np.transpose(recon_x, (1, 0))
            recon_y = np.transpose(recon_y, (1, 0))
            x_sagittal_mae = np.mean(np.abs(gt_x - recon_x))
            y_sagittal_mae = np.mean(np.abs(gt_y - recon_y))

            x_sagittal_mae_list.append(x_sagittal_mae)
            y_sagittal_mae_list.append(y_sagittal_mae)

            print(f">> Processing {name_tag}:[{idx_tag}]/[{len(total_file_list)}] x=[{idx_x}]/[{len_x-1}], x_sagittal_mae: {x_sagittal_mae:.3f}, y_sagittal_mae: {y_sagittal_mae:.3f}")

        # print the mae
        x_sagittal_mae_list = np.array(x_sagittal_mae_list)
        y_sagittal_mae_list = np.array(y_sagittal_mae_list)
        print(f"x_sagittal_mae mean: {np.mean(x_sagittal_mae_list)}, std: {np.std(x_sagittal_mae_list)}")
        print(f"y_sagittal_mae mean: {np.mean(y_sagittal_mae_list)}, std: {np.std(y_sagittal_mae_list)}")

        # save the mae
        x_sagittal_mae_savename = f"{root_folder}{name_tag}_x_sagittal_mae.npy"
        y_sagittal_mae_savename = f"{root_folder}{name_tag}_y_sagittal_mae.npy"
        np.save(x_sagittal_mae_savename, x_sagittal_mae_list)
        np.save(y_sagittal_mae_savename, y_sagittal_mae_list)
        print(f"Save {x_sagittal_mae_savename} and {y_sagittal_mae_savename}")

        # save the indices
        x_sagittal_ind = np.array(x_sagittal_ind_list)
        y_sagittal_ind = np.array(y_sagittal_ind_list)
        x_sagittal_ind_savename = f"{root_folder}{name_tag}_x_sagittal_ind.npy"
        y_sagittal_ind_savename = f"{root_folder}{name_tag}_y_sagittal_ind.npy"

        np.save(x_sagittal_ind_savename, x_sagittal_ind_list)
        np.save(y_sagittal_ind_savename, y_sagittal_ind_list)
        print(f"Save {x_sagittal_ind_savename} and {y_sagittal_ind_savename}")

        # save the recon
        x_sagittal_recon_savename = f"{root_folder}{name_tag}_x_sagittal_recon.npy"
        y_sagittal_recon_savename = f"{root_folder}{name_tag}_y_sagittal_recon.npy"
        np.save(x_sagittal_recon_savename, x_sagittal_recon)
        np.save(y_sagittal_recon_savename, y_sagittal_recon)
        print(f"Save {x_sagittal_recon_savename} and {y_sagittal_recon_savename}")

    print(f"Processing {name_tag}:[{idx_tag}]/[{len(total_file_list)}] done.")

print("All done.")




