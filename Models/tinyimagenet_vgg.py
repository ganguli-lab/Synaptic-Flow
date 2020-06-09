# Based on code taken from https://github.com/weiaicunzai/pytorch-cifar100

"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from Layers import layers

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=200, dense_classifier=False):
        super().__init__()
        self.features = features
        
        self.Linear = layers.Linear
        if dense_classifier:
            self.Linear = nn.Linear

        dim = 512 * 4
        
        self.classifier = nn.Sequential(
            self.Linear(dim, dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.Linear(dim//2, dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.Linear(dim//2, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layers.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (layers.Linear, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layer_list = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layer_list += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layer_list += [layers.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layer_list += [layers.BatchNorm2d(l)]
        
        layer_list += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layer_list)

def _vgg(arch, features, num_classes, dense_classifier, pretrained):
    model = VGG(features, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-cifar{}.pt'.format(arch, num_classes)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg11(input_shape, num_classes, dense_classifier=False, pretrained=False):
    features = make_layers(cfg['A'], batch_norm=False)
    return _vgg('vgg11', features, num_classes, dense_classifier, pretrained)

def vgg11_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    features = make_layers(cfg['A'], batch_norm=True)
    return _vgg('vgg11_bn', features, num_classes, dense_classifier, pretrained)

def vgg13(input_shape, num_classes, dense_classifier=False, pretrained=False):
    features = make_layers(cfg['B'], batch_norm=False)
    return _vgg('vgg13', features, num_classes, dense_classifier, pretrained)

def vgg13_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    features = make_layers(cfg['B'], batch_norm=True)
    return _vgg('vgg13_bn', features, num_classes, dense_classifier, pretrained)

def vgg16(input_shape, num_classes, dense_classifier=False, pretrained=False):
    features = make_layers(cfg['D'], batch_norm=False)
    return _vgg('vgg16', features, num_classes, dense_classifier, pretrained)

def vgg16_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    features = make_layers(cfg['D'], batch_norm=True)
    return _vgg('vgg16_bn', features, num_classes, dense_classifier, pretrained)

def vgg19(input_shape, num_classes, dense_classifier=False, pretrained=False):
    features = make_layers(cfg['E'], batch_norm=False)
    return _vgg('vgg19', features, num_classes, dense_classifier, pretrained)

def vgg19_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    features = make_layers(cfg['E'], batch_norm=True)
    return _vgg('vgg19_bn', features, num_classes, dense_classifier, pretrained)
