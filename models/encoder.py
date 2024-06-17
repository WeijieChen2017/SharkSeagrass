# here we apply an encoder to the input image to get the feature map
# given the input, there are three convolutional layers
# for each convolutional layer
# we apply conv3d - acti - dropout - norm -

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class conv3d_adn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, acti="PReLU", norm="InstanceNorm3d", dropout=0.0):
        super(conv3d_adn, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            self.get_activation(acti),
            nn.Dropout3d(dropout),
            self.get_norm(norm)
        )
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_activation(self, acti):
        if acti == "ReLU":
            return nn.ReLU()
        elif acti == "PReLU":
            return nn.PReLU()
        elif acti == "LeakyReLU":
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def get_norm(self, norm):
        if norm == "InstanceNorm3d":
            return nn.InstanceNorm3d()
        elif norm == "BatchNorm3d":
            return nn.BatchNorm3d()
        else:
            return nn.InstanceNorm3d()

    def forward(self, x):
        return self.layers(x)

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv3d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(in_dim, res_h_dim, kernel_size=1, stride=1, padding=1, bias=False),
        )
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers
        )
    
    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x

# for the encoder, we have three convolutional layers followed by a downsampling layer using stride 2
class Encoder_vqvae_conv(nn.Module):
    """
    This is to encode the input image to get the feature map
    Suppose the input is 64x64x64
    from VQVAE:
    The encoder consists of 2 strided convolutional layers with stride 2 and window size 4,
        followed by two residual 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units.
    """
    def __init__(self, in_dim, h_dim, n_res_layers=2, res_h_dim=256):
        super(Encoder_vqvae_conv, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            conv3d_adn(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            conv3d_adn(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            conv3d_adn(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)

    