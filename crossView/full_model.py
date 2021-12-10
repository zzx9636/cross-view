import time
from .CrossViewTransformer import CrossViewTransformer
from .CycledViewProjection import CycledViewProjection
from .model import Encoder, Decoder
from .combine_loss import combine_loss

import torch.nn as nn

class PVA_model(nn.Module):
    def __init__(self, opt, device):
        super(PVA_model, self).__init__()
        self.opt = opt
        self.models = {}
        self.parameters_to_train = []
        self.transform_parameters_to_train = []
        self.detection_parameters_to_train = []
        self.base_parameters_to_train = []
        self.parameters_to_train = []
        self.parameters_to_train_D = []
        self.criterion = combine_loss()
        self.device = device
        # Initializing models
        self.models["encoder"] = Encoder(18, self.opt.height, self.opt.width, True)

        self.models['CycledViewProjection'] = CycledViewProjection(in_dim=8)
        self.models["CrossViewTransformer"] = CrossViewTransformer(128)

        self.models["decoder"] = Decoder(
            self.models["encoder"].resnet_encoder.num_ch_enc, 3)
        self.models["transform_decoder"] = Decoder(
            self.models["encoder"].resnet_encoder.num_ch_enc, 3, "transform_decoder")

        for key in self.models.keys():
            self.models[key].to(self.device)
            if "discr" in key:
                self.parameters_to_train_D += list(
                    self.models[key].parameters())
            elif "transform" in key:
                self.transform_parameters_to_train += list(self.models[key].parameters())
            else:
                self.base_parameters_to_train += list(self.models[key].parameters())
        self.parameters_to_train = [
            {"params": self.transform_parameters_to_train, "lr": self.opt.lr_transform},
            {"params": self.base_parameters_to_train, "lr": self.opt.lr},
        ]
        self.criterion.to(self.device)
    
      
    def forward(self, inputs):
        outputs = {}
        features = self.models["encoder"](inputs["color"])

        # Cross-view Transformation Module
        x_feature = features
        transform_feature, retransform_features = self.models["CycledViewProjection"](features)
        features = self.models["CrossViewTransformer"](features, transform_feature, retransform_features)

        outputs["topview"] = self.models["decoder"](features)
        outputs["transform_topview"] = self.models["transform_decoder"](transform_feature)

        losses = self.criterion(inputs['combine'], outputs, x_feature, retransform_features)

        return outputs, losses
    
