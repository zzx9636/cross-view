import os

#import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchinfo import summary


class CycledViewProjection(nn.Module):
    def __init__(self, img_h):
        super(CycledViewProjection, self).__init__()
        self.transform_module = TransformModule(dim=img_h)
        self.retransform_module = TransformModule(dim=img_h)
        # self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        #B, C, H, W = x.view([-1, int(x.size()[1])] + list(x.size()[2:])).size()
        B = x.shape[0]
        # x = self.bn(x)
        transform_feature = self.transform_module(x)
        transform_features = transform_feature.view([B, int(x.size()[1])] + list(x.size()[2:]))
        retransform_features = self.retransform_module(transform_features)
        return transform_feature, retransform_features


class TransformModule(nn.Module):
    def __init__(self, dim=25):
        super(TransformModule, self).__init__()
        self.dim = dim
        self.mat_list = nn.ModuleList()
        # self.bn = nn.BatchNorm2d(512)
        self.fc_transform = nn.Sequential(
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU(),
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU()
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shape x: B, C, H, W
        # x = self.bn(x)
        #print(x.shape, list(x.size()[:2]) + [self.dim * self.dim, ])
        x = x.view(list(x.size()[:2]) + [self.dim * self.dim, ])
        view_comb = self.fc_transform(x)
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.dim, self.dim])
        return view_comb


if __name__ == '__main__':
    # features = torch.arange(0, 1048576)
    # features = torch.where(features < 20, features, torch.zeros_like(features))
    # # features = features.view([2, 3, 4]).float()
    # features = features.view([8, 128, 32, 32]).float()
    # CVP = CycledViewProjection(128)
    # print(CVP(features)[0].shape)
    
    CVP = CycledViewProjection(32)
    summary(CVP, [6,256,32,32])