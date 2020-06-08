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
import matplotlib as mpl
import copy

# Experiment Hyperparameters
model_architecture = 'vgg19'
model_class = 'lottery'
dataset = 'cifar10'
lr = 0.001
epochs = 100
verbose = True
directory = 'Results/lottery_layer_conservation/{}/{}'.format(dataset, model_architecture)
try:
    os.makedirs(directory)
except FileExistsError:
    pass

torch.manual_seed(seed=1)
device = load.device(gpu=9)
input_shape, num_classes = load.dimension(dataset=dataset) 
train_loader = load.dataloader(dataset=dataset, 
                              batch_size=128, 
                              train=True, 
                              workers=30)
model = load.model(model_architecture=model_architecture, 
                   model_class=model_class) 
init_model = model(input_shape=input_shape,
                   num_classes=num_classes,
                   dense_classifier=False, 
                   pretrained=False).to(device)
train_model = copy.deepcopy(init_model)

opt_class, opt_kwargs = load.optimizer('momentum')
optimizer = opt_class(generator.parameters(train_model), lr=lr, weight_decay=1e-4, **opt_kwargs)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
loss = nn.CrossEntropyLoss()


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

def average_mag_score(model):
    average_scores = []
    for module in model.modules():
        if isinstance(module, (layers.Linear, layers.Conv2d)):
            W = module.weight.detach().cpu().numpy()
            W_score = W**2
            score_sum = W_score.sum()
            num_elements = np.prod(W.shape)

            if module.bias is not None:
                b = module.bias.detach().cpu().numpy()
                b_score = b**2
                score_sum += b_score.sum()
                num_elements += np.prod(b.shape)

            average_scores.append(np.abs(score_sum / num_elements))
    return average_scores

def average_layer_score(init_model, train_model):
    average_scores = []
    for init_module, train_module in zip(init_model.modules(), train_model.modules()):
        if isinstance(init_module, (layers.Linear, layers.Conv2d)) and isinstance(train_module, (layers.Linear, layers.Conv2d)):
            W0 = init_module.weight.detach().cpu().numpy()
            W1 = train_module.weight.detach().cpu().numpy()
            W_score = W1**2 - W0**2
            score_sum = W_score.sum()
            num_elements = np.prod(W_score.shape)

            if (init_module.bias is not None) and (train_module.bias is not None):
                b0 = init_module.bias.detach().cpu().numpy()
                b1 = train_module.bias.detach().cpu().numpy()
                b_score = b1**2 - b0**2
                score_sum += b_score.sum()
                num_elements += np.prod(b_score.shape)

            average_scores.append(np.abs(score_sum / num_elements))
    return average_scores


_, inv_size = layer_names(init_model)
W0 = []
WT = []
WDelta = []

for epoch in tqdm(range(epochs)):
    train(train_model, loss, optimizer, train_loader, device, epoch, verbose)
    scheduler.step()
    W0.append(average_mag_score(init_model))
    WT.append(average_mag_score(train_model))
    WDelta.append(average_layer_score(init_model, train_model))

np.save('{}/{}'.format(directory,'inv-size'), inv_size)
np.save('{}/lr-{}_epochs-{}_W0'.format(directory, lr, epochs), np.array(W0))
np.save('{}/lr-{}_epochs-{}_WT'.format(directory, lr, epochs), np.array(WT))
np.save('{}/lr-{}_epochs-{}_WDelta'.format(directory, lr, epochs), np.array(WDelta))


# Setup Plot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

scores = [W0, WT, WDelta]

# Average Layer Score
for i, name in enumerate(['$W(0)^2$', '$W(T)^2$', '$W(T)^2 - W(0)^2$']):
    ax = axs[i%3]
    
    # Set line width of axes
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    
    # Set labels and titles
    ax.set_xlabel('Inverse layer size', size=20)
    ax.set_ylabel('Average layer score', size=20)
    ax.set_yscale('log')
    ax.set_xscale('log')

    cmap = mpl.cm.get_cmap('Spectral')
    for j in range(len(scores[i])):
      ax.scatter(inv_size, scores[i][j], s=50, color=cmap(j/len(scores[i])))

    # ax.set_xlim([0.5*np.min(inv_size), 2*np.max(inv_size)])
    # ax.set_ylim([0.5*np.min(scores[i]), 2*np.max(scores[i])])

    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', zorder=0)

fig.subplots_adjust(hspace=0.2)
fig.tight_layout()
plt.savefig('lottery-layer.pdf', bbox_inches="tight")
