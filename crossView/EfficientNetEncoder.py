import torchvision.models as models
import torch.nn as nn
class EfficientNetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, dim = 512, pretrained=True):
        super(EfficientNetEncoder, self).__init__()

        self.raw_model = models.efficientnet_b4(pretrained=pretrained)
        self.encoder = list(self.raw_model.children())[0]
        self.conv1 = nn.Conv2d(1792, dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder(x)
        x = self.bn1(self.conv1(x))
        return self.bn2(self.conv2(self.relu(x)))
    

