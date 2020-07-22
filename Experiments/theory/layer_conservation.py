import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Layers import layers
from Utils import load
from Utils import generator
from train import *
from prune import *

def run(args):
    if not args.save:
        print("This experiment requires an expid.")
        quit()

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    input_shape, num_classes = load.dimension(args.dataset) 
    data_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)

    ## Model, Loss, Optimizer ##
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(device)
    loss = nn.CrossEntropyLoss()

    ## Compute Layer Name and Inv Size ##
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

    ## Compute Average Layer Score ##
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


    ## Loop through Pruners and Save Data ##
    names, inv_size = layer_names(model)
    average_scores = []
    unit_scores = []
    for i, p in enumerate(args.pruner_list):
        pruner = load.pruner(p)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
        sparsity = 10**(-float(args.compression))
        prune_loop(model, loss, pruner, data_loader, device, sparsity, 
                   args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize)
        average_score = average_layer_score(model, pruner.scores)
        average_scores.append(average_score)
        np.save('{}/{}'.format(args.result_dir, p), np.array(average_score))
    np.save('{}/{}'.format(args.result_dir,'inv-size'), inv_size)
