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

def eta_c_compute(model_name, model, dataset_name, input_shape, gpu_idxs, verbose):
    """
    Purpose: 
    - Compute the layer-wise critical compression ratio for a given model and dataset
    - Track shapes on inputs and outputs
    
    Author: 
    - Jakob Krzyston (jakobk@gatech.edu)
    
    Inputs:
    - model_name   = Model name
    - model        = Neural netowork model
    - dataset_name = Dataset name
    - input_shape  = Dimensionality of the input images
    - gpu_idxs     = Locations of the GPUs
    - verbose      = If True, will execute print statement
    
    Outputs:
    - eta_c         = List of layer-wise critical compression ratios 
    - in_out_shapes = List of layer-wise input and output sizes
    
    Notes:
    - These are tailored to how the models and datasets in the SynFlow repo (https://github.com/ganguli-lab/Synaptic-Flow) were scripted
    - These are limited to the model & dataset combinations seen in the paper
    - The input dimensions for convolutional layers assume a padding of 1
    
    """
    # Load Packages
    import torch.nn as nn
    from Utils import load
    
    # generate a sample data point
    shape = tuple([1,input_shape[0],input_shape[1],input_shape[2]])
    out = torch.rand((shape))     
    
    # Device
    device = load.device(gpu_idxs)
    
    # get the names of all of the named_modules
    names = []
    for name, module in model.named_modules():
        names.append(name)

    # log the compression ratios on a per layer basis
    eta_c = []
    
    #log the input and output shapes
    in_out_shapes = []

    outs = [] # keep track of output dimensions
    if "cifar10" in dataset_name:
        # keep track of location in names list
        ind = 0
        if 'vgg' in model_name:
            """Compute eta_c for VGG style models wrt CIFAR 10"""
            for name, module in model.named_modules():
                mod = module.eval()
                outs.append(out.shape)
                if ind+1 < len(names):
                    if '.conv' in names[ind+1]:
                        out = mod(out.clone().detach().float().to(device))
                if '.conv' in name:
                    m = (outs[-1][2]+2)*(outs[-1][3]+2) # dimensions of input, +2 is for padding
                    n = out.shape[2]*out.shape[3]       # dimensions of output
                    eta_c.append((n*(3**2))/(m+n-1))    # kernel size is 3
                    in_out_shapes.append(((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3]))) # +2 is for padding
                if 'fc' in name:
                    eta_c.append(((module.in_features*module.out_features))/(module.in_features+module.out_features-1))
                    in_out_shapes.append((module.in_features,module.out_features))
                ind+=1 

        elif 'resnet' in model_name: 
            '''Compute eta_c for ResNet style models wrt CIFAR 10'''
            for name, module in model.named_modules():
                import torch.nn.functional as F
                mod = module.eval()
                outs.append(out.shape)
                if ind+1 < len(names):
                    if 'conv' in name:
#                         print(module)
                        out = mod(out.clone().detach().float().to(device))
                        if outs[-1][2] != out.shape[2]: # account for downsampling
                            outs[-1] = torch.rand((1,int(outs[-1][1]*2),int(outs[-1][2]/2),int(outs[-1][3]/2))).shape
                        m = (outs[-1][2]+2)*(outs[-1][3]+2) # dimensions of input
                        n = out.shape[2]*out.shape[3]       # dimensions of output
                        # get the kernel size, not hard code
                        eta_c.append((n*(3**2))/(m+n-1))    # kernel size is 3
                        in_out_shapes.append(((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3])))
                        ind += 1
                    if 'fc' in name: # otherwise it's the dense layer
                        out = F.avg_pool2d(out, out.size()[3])
                        out = out.view(out.size(0), -1)
                        out = mod(out.clone().detach().float().to(device))
                        eta_c.append((module.in_features*module.out_features)/(module.in_features+module.out_features-1))
                        in_out_shapes.append((module.in_features,module.out_features))

    elif "tiny" in dataset_name:

        if 'vgg' in model_name:
            """Compute eta_c for VGG11 model wrt tiny-imagenet"""
            conv_locations  = ['.0','.3','.6','.8','.11','.13','.16','.18'] #denotes conv layer locations
            dense_locations = ['0','3','6'] #denotes dense layer locations
            pool_locations  = ['2','5','10','15','20'] #denotes pool layer locations
            for name, module in model.named_modules():#works
                outs.append(out.shape)
                if 'classifier' in name:#dense layers
                    if name[-1].isnumeric():
                        if name[-1] == '0':
                            out = out.view(out.size(0), -1)
                        if any(x in name for x in dense_locations):
                            mod = module.eval()
                            out = mod(out.clone().detach().float().to(device))
                            eta_c.append((module.in_features*module.out_features)/(module.in_features+module.out_features-1))
                            in_out_shapes.append((module.in_features,module.out_features))
                elif 'features' in name:#conv layers
                    if name[-1].isnumeric():
                        #Conv operations
                        if any(x in name for x in conv_locations):  
                            mod = module.eval()
                            out = mod(out.clone().detach().float().to(device))
                            m = (outs[-1][2]+2)*(outs[-1][3]+2) # dimensions of input
                            n = out.shape[2]*out.shape[3]       # dimensions of output
                            eta_c.append((n*(3**2))/(m+n-1))    # kernel size is 3
                            in_out_shapes.append(((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3])))
                        #Maxpool operations
                        elif any(x in name for x in pool_locations):
                            mod = module.eval()
                            out = mod(out.clone().detach().float().to(device))

        elif 'resnet' in model_name:
            """Compute eta_c for ResNet wrt tiny-imagenet"""
            avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            conv_locations = ['0','3']  #denotes conv layer locations w/in residual, NOT in shortcut
            for name, module in model.named_modules():
                mod = module.eval()
                outs.append(out.shape)
                if name == '': #ignore first layer name, is this necessary
                    continue
                elif name == 'conv1.0': 
                    # account for the very first conv layer
                    out = mod(out.clone().detach().float().to(device))
                    if outs[-1][2] != out.shape[2]: # account for downsampling
                        outs[-1] = torch.rand((1,int(outs[-1][1]*2),int(outs[-1][2]/2),int(outs[-1][3]/2))).shape
                    m = (outs[-1][2]+2)*(outs[-1][3]+2) # dimensions of input
                    n = out.shape[2]*out.shape[3]       # dimensions of output
                    eta_c.append((n*(3**2))/(m+n-1))    # kernel size is 3
                    in_out_shapes.append(((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3])))
                elif '_x' in name and 'residual' in name:
                    if name[-1] in conv_locations:
                        # all other conv layers that are not in shortut
                        out = mod(out.clone().detach().float().to(device))
                        if outs[-1][2] != out.shape[2]: # account for downsampling
                            outs[-1] = torch.rand((1,int(outs[-1][1]*2),int(outs[-1][2]/2),int(outs[-1][3]/2))).shape
                        m = (outs[-1][2]+2)*(outs[-1][3]+2) # dimensions of input
                        n = out.shape[2]*out.shape[3]       # dimensions of output
                        eta_c.append((n*(3**2))/(m+n-1))    # kernel size is 3
                        in_out_shapes.append(((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3])))
                elif name == 'fc':
                    # dense layers, will need to change with different sized resnets 
                    # (number of dense layers will change)
                    out = avg_pool(out)
                    out = out.view(out.size(0), -1)
                    out = mod(out.clone().detach().float().to(device))
                    eta_c.append((module.in_features*module.out_features)/(module.in_features+module.out_features-1))
                    in_out_shapes.append((module.in_features,module.out_features))
    if verbose:
        print('Critical compression ratios (eta_c) per layer:')
        for layer_num in range(len(eta_c)):
            print(str(layer_num)+': '+str(round(eta_c[layer_num],5)))
    return eta_c, in_out_shapes