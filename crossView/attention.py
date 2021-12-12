# pylint: disable=too-few-public-methods
r''' Define the feature embeder for inputs'''
import torch
from torch import nn
from torch.nn.functional import gelu
from torch.nn import Linear, LayerNorm
from torch.nn.init import xavier_normal_
from matplotlib import pyplot as plt





class MultiHeadAttention(nn.Module):
    """ Multi-headed Attention Module used in the Scene-transformer
    to encode the trajectories and interactions between agents"""
    def __init__(self, dim_feature=256, num_head=4, mlp_multi=4, p_dropout = 0.1) -> None:
        super().__init__()
        self.n_head = num_head
        self.mlp_multi = mlp_multi
        self.d_K = dim_feature // self.n_head

        self.input_norm = LayerNorm(dim_feature)
        self.key_norm = LayerNorm(dim_feature)
        self.val_norm = LayerNorm(dim_feature)
        
        self.output_norm = LayerNorm(dim_feature)
        
        self.linear_Q = Linear(dim_feature, dim_feature)
        self.linear_K = Linear(dim_feature, dim_feature)
        self.linear_V = Linear(dim_feature, dim_feature)
                
        self.linear_Y = Linear(dim_feature, dim_feature)
        
        self.attention = ScaledDotProduct(p_dropout)
        
        self.mlp = nn.Sequential(
            Linear(dim_feature, mlp_multi*dim_feature),
            nn.ReLU(),
            Linear(mlp_multi*dim_feature, dim_feature),
            nn.Dropout(p = p_dropout)
        )
        self.dropout = nn.Dropout(p_dropout)
        
    def forward(self, query0, key0, value0):
        """
        Foward pass for the multi-headed attention module
        Input:
            query0:
                Tensor of size [B, W*H, C]. 
                In the scene-transformer, the query is always the agent tensor
            key0:
                Tensor of Size [B, W*H, C]. 
                Optional input for the key if we are doing cross attention
            
           
        """
        # get Batch,Future/1,Agent,Time,Feature dimension
        dim_b, dim_wh, dim_c = query0.shape
        
        X_norm = self.input_norm(query0)
        K_norm = self.key_norm(key0)
        V_norm = self.val_norm(value0)
                
        Q = self.linear_Q(X_norm).view(dim_b, dim_wh, self.n_head, self.d_K)
        
        K = self.linear_K(K_norm).view(dim_b, dim_wh, self.n_head, self.d_K)
        V = self.linear_V(V_norm).view(dim_b, dim_wh, self.n_head, self.d_K)

        Y1 = self.attention(Q, K, V)
        Y1_reshape = Y1.reshape((dim_b, dim_wh, dim_c))
        Y2 = self.dropout(self.linear_Y(Y1_reshape))
        S = query0 + Y2
        
        F = S+self.mlp(self.output_norm(S))
                
        return F
    
        
class ScaledDotProduct(nn.Module):
    def __init__(self, p_dropout=0.1):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, query, key, value, mask=None):
        _, _, _, dim_dk = query.shape
        query = query.permute(0,2,1,3)
        key = key.permute(0,2,1,3)
        value = value.permute(0,2,1,3)
        similarity = torch.matmul(query, key.transpose(-2,-1))/(dim_dk**0.5)
        
        attention_weight = self.softmax(similarity)
        attention_weight = self.dropout(attention_weight)
        score = torch.matmul(attention_weight, value)
        score = score.permute(0,2,1,3)        
        return score
        
