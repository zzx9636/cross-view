
import torch.nn as nn


from .EfficientNetEncoder import EfficientNetEncoder
from .positional_encoding import PositionalEncodingPermute2D as PositionalEncoding
from .CycledViewProjection import CycledViewProjection


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

    def __init__(self,  dim = 512, pretrained=True, freeze = False):
        super(Encoder, self).__init__()

        self.efficient_encoder = EfficientNetEncoder(dim, pretrained)
        self.positional_encoder = PositionalEncoding(dim)
        self.projection = CycledViewProjection(32)
        
        if freeze:
            for param in self.efficient_encoder.parameters():
                param.requires_grad = False
                
       

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
        
        b,n,h,w = x.shape
        
        embed = self.positional_encoder(x)
        x_top, x_hat_front = self.projection(x)
        
        x_front = (x + embed).view(b,n,h*w)
        x_top = (x_top+embed).view(b,n,h*w)
        x_hat_front = (x_hat_front+embed).view(b,n,h*w)
        
        return x_front, x_top, x_hat_front