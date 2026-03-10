import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Layers import layers


class LeNet(nn.Module):

    # network structure
    def __init__(self, input_c = 1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_c, 6, 6, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 6, stride=2)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = layers.Linear(84, 10)
        self.activation = nn.Tanh()
        

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(-1, self.num_flat_features(x))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)
    
    
def _lenet(arch, num_classes, dense_classifier, pretrained):
    model = LeNet()
    if pretrained:
        pretrained_path = 'Models/models/new/' + 'cnn_adv_tanh.pth'
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
    
def LeNet5(input_shape, num_classes, dense_classifier=False, pretrained=False):
    return _lenet('relu', num_classes, dense_classifier, pretrained)