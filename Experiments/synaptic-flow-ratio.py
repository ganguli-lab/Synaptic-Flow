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
directory = 'Results/flow/{}/{}'.format(dataset, model_architecture)
try:
    os.makedirs(directory)
except FileExistsError:
    pass

torch.manual_seed(seed=1)
device = load.device(gpu=0)
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
torch.save(model.state_dict(),"{}/model.pt".format(directory))
loss = nn.CrossEntropyLoss()
exponents = [0.0,0.5,1.0,1.5,2.0,2.5,3.0]
iterations = [1,10,20,30,40,50,60,70,80,90,100]


def score(parameters, model, loss, dataloader, device):
    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs
    @torch.no_grad()
    def nonlinearize(model, signs):
        for name, param in model.state_dict().items():
            param.mul_(signs[name])
    
    signs = linearize(model)
    (data, _) = next(iter(dataloader))
    input_dim = list(data[0,:].shape)
    input = torch.ones([1] + input_dim).to(device)
    output = model(input)
    maxflow = torch.sum(output)
    maxflow.backward()
    scores = {}
    for _, p in parameters:
        scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
        p.grad.data.zero_()
    nonlinearize(model, signs)

    return scores, maxflow.item()

def mask(parameters, scores, sparsity):
    global_scores = torch.cat([torch.flatten(v) for v in scores.values()])
    k = int((1.0 - sparsity) * global_scores.numel())
    cutsize = 0
    if not k < 1:
        cutsize = torch.sum(torch.topk(global_scores, k, largest=False).values).item()
        threshold, _ = torch.kthvalue(global_scores, k)
        for mask, param in parameters:
            score = scores[id(param)]
            zero = torch.tensor([0.]).to(mask.device)
            one = torch.tensor([1.]).to(mask.device)
            mask.copy_(torch.where(score <= threshold, zero, one))
    return cutsize

@torch.no_grad()
def apply_mask(parameters):
    for mask, param in parameters:
        param.mul_(mask)

results = []
for style in ['linear', 'exponential']:
    print(style)
    sparsity_ratios = []
    for i, exp in enumerate(exponents):
        max_ratios = []
        for j, epochs in enumerate(iterations):
            model.load_state_dict(torch.load("{}/model.pt".format(directory), map_location=device))
            parameters = list(generator.masked_parameters(model, False, False, False))
            model.eval()
            ratios = []
            for epoch in tqdm(range(epochs)):
                apply_mask(parameters)
                scores, maxflow = score(parameters, model, loss, data_loader, device)
                sparsity = 10**(-float(exp))
                if style == 'linear':
                    sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
                if style == 'exponential':
                    sparse = sparsity**((epoch + 1) / epochs)
                cutsize = mask(parameters, scores, sparse)
                ratios.append(cutsize / maxflow)
            max_ratios.append(max(ratios))
        sparsity_ratios.append(max_ratios)
    results.append(sparsity_ratios)
np.save('{}/ratios'.format(directory), np.array(results))
