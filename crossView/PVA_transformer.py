from .encoder import Encoder
from .decoder import Decoder
from .attention import MultiHeadAttention
from .combine_loss import combine_loss

import torch
from torch import nn

class PVA_transformer(nn.Module):
    def __init__(self):
        super(PVA_transformer, self).__init__()
        self.encoder = Encoder(dim = 512, pretrained=True)
        
        self.cross_attention = MultiHeadAttention(dim_feature=512)
        
        self.self_attention = MultiHeadAttention(dim_feature=512)
        
        self.decoder = Decoder(num_class=3)
        
        self.loss_func = combine_loss()
        
        
    def forward(self, inputs):
        
        x_front, x_top, x_hat_front = self.encoder(inputs["color"])

        x_front_permute = x_front.permute(0,2,1)
        x_top_permute = x_top.permute(0,2,1)
        x_hat_front_permute = x_hat_front.permute(0,2,1)
        
        # Cross-view Transformation Module
        x = self.cross_attention(query0 = x_top_permute, key0 = x_front_permute, value0 = x_hat_front_permute)
        
        x = self.self_attention(x,x,x).permute(0,2,1)
        
        dim_b, dim_c, _ = x.shape
        x = x.reshape((dim_b, dim_c, 32, 32))
        
        output = self.decoder(x)
        
        losses = self.loss_func(inputs['combine'], output, x_front, x_hat_front)

        return output, losses
    