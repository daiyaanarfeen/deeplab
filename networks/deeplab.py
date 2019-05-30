from networks.resnet_feature_extractor import ResNetFeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck as Bottleneck 

class ASPP(nn.Module):

    def __init__(self, rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.branch0 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) 
        self.branch2 = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.mix = nn.Sequential(
            nn.Conv2d(5*256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b4 = F.interpolate(b4, size=(x.shape[2], x.shape[3]), mode='bilinear')
        p_out = torch.cat((b0, b1, b2, b3, b4), dim=1)
        out = self.mix(p_out)
        return out

class DeepLabv3(ResNetFeatureExtractor):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], output_stride=16, num_classes=150):
        super(DeepLabv3, self).__init__(block, layers, output_stride)
        if output_stride == 8:
            rates = (12, 24, 26)
        elif output_stride == 16:
            rates = (6, 12, 18)
        self.aspp = ASPP(rates=rates)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        x = super(DeepLabv3, self).forward(x)
        x = self.aspp(x)
        x = self.classifier(x)
        return x
