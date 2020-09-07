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


    ## Compute per Neuron Score ##
    def unit_score_sum(model, scores, compute_linear=False):
        in_scores = []
        out_scores = []
        for name, module in model.named_modules():
            if (isinstance(module, layers.Linear) and compute_linear):
                W = module.weight
                b = module.bias

                W_score = scores[id(W)].detach().cpu().numpy()
                b_score = scores[id(b)].detach().cpu().numpy()

                in_scores.append(W_score.sum(axis=1) + b_score)
                out_scores.append(W_score.sum(axis=0))
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

    ## Loop through Pruners and Save Data ##
    unit_scores = []
    pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    sparsity = 10**(-float(args.compression))
    prune_loop(model, loss, pruner, data_loader, device, sparsity, 
               args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode)
    unit_score = unit_score_sum(model, pruner.scores)
    unit_scores.append(unit_score)
    np.save('{}/{}'.format(args.result_dir, args.pruner), unit_score)
