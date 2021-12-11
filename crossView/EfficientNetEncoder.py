import torchvision.models as models
import torch.nn as nn
class EfficientNetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, pretrained=True):
        super(EfficientNetEncoder, self).__init__()

        self.raw_model = models.efficientnet_b7(pretrained=pretrained)
        self.encoder = list(self.raw_model.children())[0]
        self.num_ch_enc = [64, 32, 48, 80, 160, 224, 384, 640, 2560]

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        return self.encoder(x)
    

