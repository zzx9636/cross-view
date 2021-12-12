
import torch.nn as nn

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
        
        
        self.block0 = UpSample(512, 256, False) # 32
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