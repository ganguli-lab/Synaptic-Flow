import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Layers import layers
from Utils import load
from Utils import generator
from train import *
from prune import *
import matplotlib.pylab as plt

# Experiment Hyperparameters
model_architecture = 'vgg19'
model_class = 'imagenet'
dataset = 'imagenet'
pretrained = False
directory = 'Results/layer_conservation/{}/{}'.format(dataset, model_architecture)
try:
    os.makedirs(directory)
except FileExistsError:
    pass

torch.manual_seed(seed=1)
device = load.device(gpu=5)
input_shape, num_classes = load.dimension(dataset=dataset) 
data_loader = load.dataloader(dataset=dataset, 
                              batch_size=16, 
                              train=True, 
                              workers=30,
                              length=10*num_classes)
model = load.model(model_architecture=model_architecture, 
                   model_class=model_class) 
model = model(input_shape=input_shape,
              num_classes=num_classes,
              dense_classifier=False, 
              pretrained=False).to(device)
loss = nn.CrossEntropyLoss()
pruners = []#'rand','mag','cs','grasp','sf']


def layer_names(model):
    names = []
    inv_size = []
    for name, module in model.named_modules():
        if isinstance(module, (layers.Linear, layers.Conv2d)):
            num_elements = np.prod(module.weight.shape)
            if module.bias is not None:
                num_elements += np.prod(module.bias.shape)
            names.append(name)
            inv_size.append(1.0/num_elements)
    return names, inv_size

def average_layer_score(model, scores):
    average_scores = []
    for name, module in model.named_modules():
        if isinstance(module, (layers.Linear, layers.Conv2d)):
            W = module.weight
            W_score = scores[id(W)].detach().cpu().numpy()
            score_sum = W_score.sum()
            num_elements = np.prod(W.shape)

            if module.bias is not None:
                b = module.bias
                b_score = scores[id(b)].detach().cpu().numpy()
                score_sum += b_score.sum()
                num_elements += np.prod(b.shape)

            average_scores.append(np.abs(score_sum / num_elements))
    return average_scores


names, inv_size = layer_names(model)
average_scores = []
unit_scores = []
for i, p in enumerate(pruners):
    pruner = load.pruner(p)(generator.masked_parameters(model, True, False, False))
    prune_loop(model, loss, pruner, data_loader, device, 1.0, False, 'global', 1)
    average_score = average_layer_score(model, pruner.scores)
    average_scores.append(average_score)
    np.save('{}/{}-{}'.format(directory, p, 'pretrained' if pretrained else 'initialization'), np.array(average_score))
np.save('{}/{}'.format(directory,'inv-size'), inv_size)
