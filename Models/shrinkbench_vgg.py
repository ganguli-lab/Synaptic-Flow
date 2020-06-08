# Code taken from: https://shrinkbench.github.io

# Faithful reimplementation of
# https://github.com/szagoruyko/cifar.torch/blob/master/models/vgg_bn_drop.lua
# BLOG POST : http://torch.ch/blog/2015/07/30/cifar.html

import math
import torch
import torch.nn as nn
from Layers import layers
import torch.nn.init as init


class ConvBNReLU(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(ConvBNReLU, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv = layers.Conv2d(in_planes, out_planes, kernel_size=3, padding=3//2)
        self.bn = layers.BatchNorm2d(out_planes, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VGGBnDrop(nn.Module):

    def __init__(self, num_channels, num_classes, dense_classifier, size):
        super(VGGBnDrop, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(

            ConvBNReLU(self.num_channels, 64), nn.Dropout(0.3),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            ConvBNReLU(64, 128), nn.Dropout(0.4),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            ConvBNReLU(128, 256), nn.Dropout(0.4),
            ConvBNReLU(256, 256), nn.Dropout(0.4),
            ConvBNReLU(256, 256),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            ConvBNReLU(256, 512), nn.Dropout(0.4),
            ConvBNReLU(512, 512), nn.Dropout(0.4),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            ConvBNReLU(512, 512), nn.Dropout(0.4),
            ConvBNReLU(512, 512), nn.Dropout(0.4),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )

        dim = size * 512
        
        modules = [
            nn.Dropout(0.5),
            layers.Linear(dim, dim),
            layers.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        ]
        if dense_classifier:
            modules.append(nn.Linear(dim, self.num_classes))
        else:
            modules.append(layers.Linear(dim, self.num_classes))
        self.classifier = nn.Sequential(*modules)
        

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_weights(self):

        def init_weights(module):
            if isinstance(module, layers.Conv2d) or isinstance(module, layers.Linear):
                #fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                #init.normal_(module.weight, 0, math.sqrt(2/fan_in))
                init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                init.zeros_(module.bias)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                #fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                #init.normal_(module.weight, 0, math.sqrt(2/fan_in))
                init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                init.zeros_(module.bias)

        self.apply(init_weights)


def vgg16_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    num_channels, width, height = input_shape
    size = math.ceil(width / 2**5) * math.ceil(height / 2**5)
    model = VGGBnDrop(num_channels=num_channels, 
                      num_classes=num_classes, 
                      dense_classifier=dense_classifier, 
                      size=size)
    model.reset_weights()
    if pretrained:
        file = 'shrinkbench-vgg16.pt'
        pretrained_path = "Models/pretrained/{}".format(file)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    return model
