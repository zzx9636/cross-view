
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as PLT
# import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class compute_transform_losses(nn.Module):
    def __init__(self):
        super(compute_transform_losses, self).__init__()
        self.l1_loss = L1Loss()

    def forward(self, input, target):
        loss = F.l1_loss(input, target, size_average=False)
        return loss


class combine_loss(nn.Module):
    def __init__(self):
        super(combine_loss, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss()
        weight = torch.Tensor([1., 5., 15.])
        weight.to(device)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, gt, outputs, features, retransform_features):
        losses = {}
        #print(gt.shape, outputs["topview"].shape)
        losses["topview_loss"] = self.compute_topview_loss(
            outputs["topview"], gt)
        losses["transform_topview_loss"] = self.compute_topview_loss(
            outputs["transform_topview"],gt)
        losses["transform_loss"] = self.compute_transform_losses(
            retransform_features,
            features
            )
        losses["loss"] = losses["topview_loss"] + 0.001 * losses["transform_loss"] + 1 * losses["transform_topview_loss"]

        return losses

    def compute_topview_loss(self, outputs, true_top_view):
        generated_top_view = outputs
        #true_top_view = torch.squeeze(true_top_view.long())
        true_top_view = true_top_view.long()
        output = self.cross_entropy(generated_top_view, true_top_view)

        return output

    def compute_transform_losses(self, input, target):
        loss = self.L1Loss(input, target)
        return loss
