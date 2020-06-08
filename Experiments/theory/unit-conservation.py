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
directory = 'Results/unit_conservation/{}/{}'.format(dataset, model_architecture)
try:
    os.makedirs(directory)
except FileExistsError:
    pass

torch.manual_seed(seed=1)
device = load.device(gpu=6)
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
              pretrained=pretrained).to(device)
loss = nn.CrossEntropyLoss()
pruners = ['rand','mag','cs','grasp','sf']


def unit_score_sum(model, scores):
    in_scores = []
    out_scores = []
    for name, module in model.named_modules():
        # # Only plot hidden units between convolutions
        # if isinstance(module, layers.Linear):
        #     W = module.weight
        #     b = module.bias

        #     W_score = scores[id(W)].detach().cpu().numpy()
        #     b_score = scores[id(b)].detach().cpu().numpy()

        #     in_scores.append(W_score.sum(axis=1) + b_score)
        #     out_scores.append(W_score.sum(axis=0))
        if isinstance(module, layers.Conv2d):
            W = module.weight
            W_score = scores[id(W)].detach().cpu().numpy()
            in_score = W_score.sum(axis=(1,2,3)) 
            out_score = W_score.sum(axis=(0,2,3))

            if module.bias is not None:
                b = module.bias
                b_score = scores[id(b)].detach().cpu().numpy()
                in_score += b_score
            
            in_scores.append(in_score)
            out_scores.append(out_score)

    in_scores = np.concatenate(in_scores[:-1])
    out_scores = np.concatenate(out_scores[1:])
    return in_scores, out_scores


unit_scores = []
for i, p in enumerate(pruners):
    pruner = load.pruner(p)(generator.masked_parameters(model, True, False, False))
    prune_loop(model, loss, pruner, data_loader, device, 1.0, False, 'global', 1)
    unit_score = unit_score_sum(model, pruner.scores)
    unit_scores.append(unit_score)
    np.save('{}/{}-{}'.format(directory, p, 'pretrained' if pretrained else 'initialization'), unit_score)
