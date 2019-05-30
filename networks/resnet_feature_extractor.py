from torchvision import models as models
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck as Bottleneck
from torchvision.models.resnet import conv1x1, conv3x3

def _make_layer(inplanes, planes, block, blocks, stride=1, norm_layer=None):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride=stride, downsample=downsample,))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


class ResNetFeatureExtractor(nn.Module):

    def __init__(self, block, layers, output_stride=None):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet101()
        # Stem
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Repurposed resnet components
        self.maxpool = resnet.maxpool
        self.layer1 = _make_layer(128, 64, Bottleneck, 3)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Change strides/dilations according to output stride
        if output_stride is not None:
             assert(output_stride % 4 == 0)

             rate = 1
             stride = 4
             for layer in range(1, 5):
                 cur_layer = getattr(self, 'layer' + str(layer))
                 for block_num in range(len(cur_layer)):
                     block = cur_layer[block_num]
                     conv3x3 = block.conv2
                     if stride == output_stride:
                         conv3x3.dilation = (rate, rate)
                         conv3x3.padding = (rate, rate)
                         rate *= conv3x3.stride[0]
                         conv3x3.stride = (1, 1)
                         if block_num == 0:
                             downsample = block.downsample[0]
                             downsample.stride = (1, 1)
                     else:
                         stride *= conv3x3.stride[0]
                     

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

res = ResNetFeatureExtractor(Bottleneck, [3, 4, 23, 3], output_stride=4)
res.load_state_dict(torch.load('resnet101_v1s.pth'))
with torch.no_grad():
    print(res(torch.ones((1, 3, 256, 256))).shape)
