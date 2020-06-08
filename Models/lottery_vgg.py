# Based on code taken from https://github.com/facebookresearch/open_lth

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
from Layers import layers

class ConvModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvModule, self).__init__()
        self.conv = layers.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class ConvBNModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvBNModule, self).__init__()
        self.conv = layers.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn = layers.BatchNorm2d(out_filters)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class VGG(nn.Module):
    """A VGG-style neural network designed for CIFAR-10."""

    def __init__(self, plan, conv, num_classes=10, dense_classifier=False):
        super(VGG, self).__init__()
        layer_list = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_list.append(conv(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layer_list)        

        self.fc = layers.Linear(512, num_classes)
        if dense_classifier:
            self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _plan(num):
    if num == 11:
        plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 13:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 16:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    elif num == 19:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    else:
        raise ValueError('Unknown VGG model: {}'.format(num))
    return plan

def _vgg(arch, plan, conv, num_classes, dense_classifier, pretrained):
    model = VGG(plan, conv, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg11(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11)
    return _vgg('vgg11_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg11_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11)
    return _vgg('vgg11_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg13(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(13)
    return _vgg('vgg13_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg13_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(13)
    return _vgg('vgg13_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg16(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(16)
    return _vgg('vgg16_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg16_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(16)
    return _vgg('vgg16_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg19(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(19)
    return _vgg('vgg19_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg19_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(19)
    return _vgg('vgg19_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)
