import torch
import torch.nn as nn
import numpy as np
from Layers import layers
from torch.nn import functional as F


def fc(input_shape, num_classes, dense_classifier=False, pretrained=False, L=6, N=100, nonlinearity=nn.ReLU()):
  size = np.prod(input_shape)
  
  # Linear feature extractor
  modules = [nn.Flatten()]
  modules.append(layers.Linear(size, N))
  modules.append(nonlinearity)
  for i in range(L-2):
    modules.append(layers.Linear(N,N))
    modules.append(nonlinearity)

  # Linear classifier
  if dense_classifier:
    modules.append(nn.Linear(N, num_classes))
  else:
    modules.append(layers.Linear(N, num_classes))
  model = nn.Sequential(*modules)

  # Pretrained model
  if pretrained:
    print("WARNING: this model does not have pretrained weights.")
  
  return model


def conv(input_shape, num_classes, dense_classifier=False, pretrained=False, L=3, N=32, nonlinearity=nn.ReLU()): 
  channels, width, height = input_shape
  
  # Convolutional feature extractor
  modules = []
  modules.append(layers.Conv2d(channels, N, kernel_size=3, padding=3//2))
  modules.append(nonlinearity)
  for i in range(L-2):
    modules.append(layers.Conv2d(N, N, kernel_size=3, padding=3//2))
    modules.append(nonlinearity)
      
  # Linear classifier
  modules.append(nn.Flatten())
  if dense_classifier:
    modules.append(nn.Linear(N * width * height, num_classes))
  else:
    modules.append(layers.Linear(N * width * height, num_classes))
  model = nn.Sequential(*modules)

  # Pretrained model
  if pretrained:
    print("WARNING: this model does not have pretrained weights.")
  
  return model