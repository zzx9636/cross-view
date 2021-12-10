import argparse
import os
import random
import time
from pathlib import Path

import crossView

import numpy as np
import copy

import torch
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map

from opt import get_args

import tqdm

from losses import compute_losses
from utils import mean_IU, mean_precision


class PVA_model:
    def __init__(self):
        self.opt = get_args()
        self.models = {}
        self.device = "cuda"
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.transform_parameters_to_train = []
        self.detection_parameters_to_train = []
        self.base_parameters_to_train = []
        self.parameters_to_train = []
        self.parameters_to_train_D = []
        self.criterion = compute_losses()
        self.create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.epoch = 0
        self.start_epoch = 0
        self.scheduler = 0
        # Save log and models path
        self.opt.log_root = os.path.join(self.opt.log_root, self.opt.split)
        self.opt.save_path = os.path.join(self.opt.save_path, self.opt.split)
        if self.opt.split == "argo":
            self.opt.log_root = os.path.join(self.opt.log_root, self.opt.type)
            self.opt.save_path = os.path.join(self.opt.save_path, self.opt.type)
        self.writer = SummaryWriter(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time))
        self.log = open(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time,
                                     '%s.csv' % self.opt.model_name), 'w')

        if self.seed != 0:
            self.set_seed()  # set seed

        # Initializing models
        self.models["encoder"] = crossView.Encoder(18, self.opt.height, self.opt.width, True)

        self.models['CycledViewProjection'] = crossView.CycledViewProjection(in_dim=8)
        self.models["CrossViewTransformer"] = crossView.CrossViewTransformer(128)

        self.models["decoder"] = crossView.Decoder(
            self.models["encoder"].resnet_encoder.num_ch_enc, self.opt.num_class)
        self.models["transform_decoder"] = crossView.Decoder(
            self.models["encoder"].resnet_encoder.num_ch_enc, self.opt.num_class, "transform_decoder")

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

        



    def process_batch(self, inputs, validation=False):
        outputs = {}
        for key, input in inputs.items():
            if key != "filename":
                inputs[key] = input.to(self.device)

        features = self.models["encoder"](inputs["color"])

        # Cross-view Transformation Module
        x_feature = features
        transform_feature, retransform_features = self.models["CycledViewProjection"](features)
        features = self.models["CrossViewTransformer"](features, transform_feature, retransform_features)

        outputs["topview"] = self.models["decoder"](features)
        outputs["transform_topview"] = self.models["transform_decoder"](transform_feature)
        if validation:
            return outputs
        losses = self.criterion(self.opt, self.weight, inputs, outputs, x_feature, retransform_features)

        return outputs, losses

 

if __name__ == "__main__":
    start_time = time.ctime()
    print(start_time)
    trainer = Trainer()
    trainer.train()
    end_time = time.ctime()
    print(end_time)
