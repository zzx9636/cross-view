from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchinfo.torchinfo import forward_pass

from .EfficientNetEncoder import EfficientNetEncoder
from torchinfo import summary
# Utils


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out
    
    


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Encoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_layers : int
        Number of layers to use in the ResNet
    img_ht : int
        Height of the input RGB image
    img_wt : int
        Width of the input RGB image
    pretrained : bool
        Whether to initialize ResNet with pretrained ImageNet parameters

    Methods
    -------
    forward(x, is_training):
        Processes input image tensors into output feature tensors
    """

    def __init__(self,  pretrained=True, freeze = True):
        super(Encoder, self).__init__()

        self.efficient_encoder = EfficientNetEncoder(pretrained)
        if freeze:
            for param in self.efficient_encoder.parameters():
                param.requires_grad = False
                
        num_ch_enc = self.efficient_encoder.num_ch_enc
        # convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 512)
        self.conv2 = Conv3x3(512, 256)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of Image tensors
            | Shape: (batch_size, 3, img_height, img_width)

        Returns
        -------
        x : torch.FloatTensor
            Batch of low-dimensional image representations
            | Shape: (batch_size, 128, img_height/128, img_width/128)
        """

        x = self.efficient_encoder(x)
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        return x

class UpSample(nn.Module):
    def __init__(self, c_in, c_out, upscale=True):
        super(UpSample, self).__init__()
        self.upscale = upscale
        self.conv0 = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.norm0 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.upsample = nn.ConvTranspose2d(c_in, c_out, 3, 2, 1, 1)
        # self.conv1 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        # self.norm1 = nn.BatchNorm2d(c_out)
        # self.dropout = nn.Dropout3d(0.1)
        
    def forward(self, x):
        if self.upscale:
            x = self.relu(self.norm0(self.upsample(x)))
        else:
            x = self.relu(self.norm0(self.conv0(x)))
        return x
        
class Decoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_ch_enc : list
        channels used by the ResNet Encoder at different layers

    Methods
    -------
    forward(x, ):
        Processes input image features into output occupancy maps/layouts
    """

    def __init__(self,  num_class=3):
        super(Decoder, self).__init__()
        self.num_output_channels = num_class
        
        
        self.block0 = UpSample(256, 256, True) # 32
        self.block1 = UpSample(256, 128, True) # 64
        self.block2 = UpSample(128, 64, True) # 128
        self.block3 = UpSample(64, 32, True) # 256
        self.block4 = UpSample(32, 16, False) # 64

        self.conv_topview = Conv3x3(
            16, self.num_output_channels)
        

    def forward(self, x):

        x = self.block0(x)
        x = self.block1(x)        
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_topview(x)
        
        return x

if __name__ == '__main__':
    model = Encoder(True, True)
    summary(model, [16,3,1024, 1024])