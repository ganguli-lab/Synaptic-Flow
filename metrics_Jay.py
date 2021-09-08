import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from prune import * 
from Layers import layers

def summary(model, scores, flops, prunable):
    r"""Summary of compression results for a model.
    """
    rows = []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            pruned = prunable(module) and id(param) in scores.keys()
            if pruned:
                sparsity = getattr(module, pname+'_mask').detach().cpu().numpy().mean()
                score = scores[id(param)].detach().cpu().numpy()
            else:
                sparsity = 1.0
                score = np.zeros(1)
            shape = param.detach().cpu().numpy().shape
            flop = flops[name][pname]
            score_mean = score.mean()
            score_var = score.var()
            score_sum = score.sum()
            score_abs_mean = np.abs(score).mean()
            score_abs_var  = np.abs(score).var()
            score_abs_sum  = np.abs(score).sum()
            rows.append([name, pname, sparsity, np.prod(shape), shape, flop,
                         score_mean, score_var, score_sum, 
                         score_abs_mean, score_abs_var, score_abs_sum, 
                         pruned])

    columns = ['module', 'param', 'sparsity', 'size', 'shape', 'flops', 'score mean', 'score variance', 
               'score sum', 'score abs mean', 'score abs variance', 'score abs sum', 'prunable']
    return pd.DataFrame(rows, columns=columns)

def flop(model, input_shape, device):

    total = {}
    def count_flops(name):
        def hook(module, input, output):
            flops = {}
            if isinstance(module, layers.Linear) or isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                flops['weight'] = in_features * out_features
                if module.bias is not None:
                    flops['bias'] = out_features
            if isinstance(module, layers.Conv2d) or isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = int(np.prod(module.kernel_size))
                output_size = output.size(2) * output.size(3)
                flops['weight'] = in_channels * out_channels * kernel_size * output_size
                if module.bias is not None:
                    flops['bias'] = out_channels * output_size
            if isinstance(module, layers.BatchNorm1d) or isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    flops['weight'] = module.num_features
                    flops['bias'] = module.num_features
            if isinstance(module, layers.BatchNorm2d) or isinstance(module, nn.BatchNorm2d):
                output_size = output.size(2) * output.size(3)
                if module.affine:
                    flops['weight'] = module.num_features * output_size
                    flops['bias'] = module.num_features * output_size
            if isinstance(module, layers.Identity1d):
                flops['weight'] = module.num_features
            if isinstance(module, layers.Identity2d):
                output_size = output.size(2) * output.size(3)
                flops['weight'] = module.num_features * output_size
            total[name] = flops
        return hook
    
    for name, module in model.named_modules():
        module.register_forward_hook(count_flops(name))

    input = torch.ones([1] + list(input_shape)).to(device)
    model(input)

    return total


# def conservation(model, scores, batchnorm, residual):
#     r"""Summary of conservation results for a model.
#     """
#     rows = []
#     bias_flux = 0.0
#     mu = 0.0
#     for name, module in reversed(list(model.named_modules())):
#         if prunable(module, batchnorm, residual):
#             weight_flux = 0.0
#             for pname, param in module.named_parameters(recurse=False):
                
#                 # Get score
#                 score = scores[id(param)].detach().cpu().numpy()
                
#                 # Adjust batchnorm bias score for mean and variance
#                 if isinstance(module, (layers.Linear, layers.Conv2d)) and pname == "bias":
#                     bias = param.detach().cpu().numpy()
#                     score *= (bias - mu) / bias
#                     mu = 0.0
#                 if isinstance(module, (layers.BatchNorm1d, layers.BatchNorm2d)) and pname == "bias":
#                     mu = module.running_mean.detach().cpu().numpy()
                
#                 # Add flux
#                 if pname == "weight":
#                     weight_flux += score.sum()
#                 if pname == "bias":
#                     bias_flux += score.sum()
#             layer_flux = weight_flux
#             if not isinstance(module, (layers.Identity1d, layers.Identity2d)):
#                 layer_flux += bias_flux
#             rows.append([name, layer_flux])
#     columns = ['module', 'score flux']

#     return pd.DataFrame(rows, columns=columns)

def neural_persistence(model):
    total = {}
    
    # find the maximum weight in the network (h_max), to normalize all weights
    weights_all = []
    for name, param in model.named_parameters():
        if 'weight' in name: # finds max weight per layer
            weights_all.append(np.double(torch.max(torch.abs(param)).detach().numpy()))
    h_max = max(weights_all)
    
    def compute_NP(name, h_max):
        def hook(module, h_max, input, output):
            NPs = {}
            if isinstance(module, layers.Linear) or isinstance(module, nn.Linear):
                """
                Pseudocode was obtained from Algorithm 1 in Rieck et al., 2019 
                """
                # extract weights
                weights = module.weights

                # normalize by the largest weight
                h_prime = torch.abs(weights/h_max)

#                 for k in range(num_layer):
                    # establish filtration of kth layer 
                    
                    
                    # calculate persistence diagram
                                    
                
            if isinstance(module, layers.Conv2d) or isinstance(module, nn.Conv2d): #computes on a per filter basis
                """
                Pseudocode was obtained from Section A.4 in Rieck et al., 2019 
                """
                # extract weights
                weights = module.weights
                
                # initalize number of tuples, tuple counter, weight index
                # number of input neurons (m), number of output neurons (n)
                m = module.in_features
                n = module.out_features
                tau = m + n
                t = 0
                i = 0

                # transform weights for filtration
                h_prime = torch.abs(weights/h_max)

                # sort weights in descending order
                h_sort =  torch.sort(torch.flatten(h_prime), dim=0, descending=True)

                # determine the set of all corner weights for filter H', p & q are filter dimensions
                p = module.conv.kernel_size[0]
                q = module.conv.kernel_size[1]
                
                h_corner = {h_prime[0,0],h_prime[0,q-1], h_prime[p-1,0], h_prime[p-1,q-1]}
                # get the indices for the corner weights in the vectorized version of the filter
                corner_idx = [0, q-1, p*(q-1), (p*q)-1]

                # add tuple for surviving component
                NPs.append((1,0))
            
                # Each corner of H' merges components
                for c in range(len(h_corner)):
                    NPs.append((1, h_corner[c]))
                    t+=1
                
#                 # create the remaining tuples
#                 while 1 do:
#                     # if current weight is corner, write one less tuple
#                     if i in corner_idx:
#                         n_prime = n - 1
#                     else:
#                         n_prime = n 
                    
#                     # if there are at least n' ,or tuples, set merge value to sorted[i]
#                     if t+n_prime <= tau:
#                         for j in range(n_prime):
#                             NPs.append((1,h_sort[i]))
#                         t+=n_prime
#                         i+=1
#                     else:
#                         for j in range(tau-t):
#                             NPs.append((1, h_sort[i]))
#                         break
                # compute norm of approximated persistence diagram
                NPs = torch.norm(NPs, 2)

            total[name] = NPs
        return hook
    for name, module in model.named_modules():
        module.register_forward_hook(compute_NP(name))
    return total