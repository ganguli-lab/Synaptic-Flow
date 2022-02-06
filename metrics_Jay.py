import torch
import torch.nn as nn
from torch.linalg import norm as torchnorm
import numpy as np
from numpy.linalg import norm
import pandas as pd
from prune import * 
from Layers import layers
import networkx as nx

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
    totalNP = {}
    sortedEdges = {} ##for the entire layer
    nodeIDs = {}

    def compute_NP(name):

        def hook(module, input, output):
            # NPs = {}

            if isinstance(module, layers.Linear) or isinstance(module, nn.Linear): ## Fully connected layers
                """
                Uses networkx to compute the MST
                """
                ## extract weights
                ## shape is (nodesIn, nodesOut)
                weights_np = module.weights.detach().cpu().numpy()

                ## find the largest weight
                h_max = np.max(np.abs(weights_np))

                ## normalize by the largest weight
                W_prime = np.abs(weights_np/h_max)

                d1, d2 = weights_np.shape

                cardV = np.sum(d1+d2)
                cardMST = cardV - 1

                Adj = np.zeros((cardV,cardV))

                for ii in range(d1):
                    for jj in range(d2):
                        Adj[d1+jj,ii] = W_prime[ii,jj]

                Adj += Adj.T

                Gr = nx.from_numpy_array(Adj)
                Tr = nx.maximum_spanning_tree(Gr)

                d_arr = np.zeros(cardMST).reshape(-1,1)
                c_arr = np.ones(cardMST).reshape(-1,1)

                for kk in range(cardMST):
                    d_arr[kk] = sorted(Tr.edges(data=True))[kk][2]['weight']

                pdMat = np.hstack((c_arr,d_arr))

                pers = np.abs(np.diff(pdMat,axis=-1))
                layerNP = norm(pers,ord=2)
                if normalized:
                    layerNP = (layerNP-0)/((cardV-2)**0.5)

                totalNP[name] = layerNP
                currentEdges = sorted(Tr.edges(data=True))
                sortedEdges[name] = currentEdges

                ## MST to nodeIDs
                nodeIDs[name] = np.zeros((cardMST,2),dtype=int) ##node In, node Out
                for ii in range(cardMST):
                    nodeIDs[name][ii,0] = currentEdges[ii][0]
                    nodeIDs[name][ii,1] = currentEdges[ii][1] - d1

            # return layerNP, sorted(Tr.edges(data=True)), nodeIDs

            if isinstance(module, layers.Conv2d) or isinstance(module, nn.Conv2d): ## Convolutional Layers

                ## extract weights
                ## weights shape = (#filters, #channels, krows, kcols)
                weights_np = module.weights.detach().cpu().numpy()

                nFilters = weights_np.shape[0]
                allWeightsSum = np.sum(np.abs(weights_np),axis=1) ## Absolute weights summed across channels of a filter

                filterNP_list = np.zeros(nFilters)
                filterEdges = {}
                filterNodeIDs = {}

                for filtNum in range(nFilters):
                    h_max = np.max(allWeightsSum[filtNum])
                    W_prime = allWeightsSum[filtNum]/h_max ## Normalized weights of conv kernel

                    if len(W_prime.shape)==3:
                        nChannels, kRows, kCols = W_prime.shape ##W_prime is a 3D tensor
                    else:
                        nChannels = 1
                        kRows, kCols = W_prime.shape ##W_prime is a 2D tensor, same size as spatial kernel

                    inputSizes, outputSizes = in_out_sizes

                    inRows, inCols = inputSizes
                    outRows, outCols = outputSizes

                    nodesIn = np.prod(inputSizes)
                    nodesOut = np.prod(outputSizes)

                    cardV = np.sum(nodesIn+nodesOut)
                    cardMST = cardV - 1

                    bigW_prime = np.zeros((nodesOut,nodesIn))

                    ##Arrange conv weights into FC-type weight matrix bigW_prime
                    convWeights = np.zeros(nodesIn)
                    for ii in range(kRows):
                        convWeights[(ii*inCols):(ii*inCols)+kCols] = W_prime[ii]

                    ## assume stride of 1 in both directions
                    nHSteps = inCols - kCols + 1
                    nVSteps = inRows - kRows + 1

                    rowCtr = 0
                    colCtr = 0

                    for jj in range(nodesOut):
                        rollIdx = np.ravel_multi_index(np.array([rowCtr,colCtr]),(inRows,inCols))
                        bigW_prime[jj] = np.roll(convWeights.copy(),rollIdx)
                        colCtr += 1
                        if colCtr%nHSteps == 0:
                            rowCtr += 1
                            colCtr = 0

                    Adj = np.zeros((cardV,cardV))
                    d1, d2 = bigW_prime.shape

                    for ii in range(d1):
                        for jj in range(d2):
                            Adj[d1+jj,ii] = bigW_prime[ii,jj]

                    Adj += Adj.T

                    Gr = nx.from_numpy_array(Adj.T) ## NOTE: Graph is computed on the transposed Adj to ensure consistency with linear layers
                    Tr = nx.maximum_spanning_tree(Gr)

                    d_arr = np.zeros(cardMST).reshape(-1,1)
                    c_arr = np.ones(cardMST).reshape(-1,1)

                    for kk in range(cardMST):
                        d_arr[kk] = sorted(Tr.edges(data=True))[kk][2]['weight']

                    pdMat = np.hstack((c_arr,d_arr))

                    pers = np.abs(np.diff(pdMat,axis=-1))
                    filterNP = norm(pers,ord=2)

                    if normalized:
                        filterNP = (filterNP-0)/((cardV-2)**0.5)

                    filterNP_list[filtNum] = filterNP
                    currentEdges = sorted(Tr.edges(data=True))
                    filterEdges[filtNum] = currentEdges

                    ## MST to nodeIDs
                    fNodeIDs = np.zeros((cardMST,2),dtype=int) ##node In, node Out

                    for ii in range(cardMST):
                        fNodeIDs[ii,0] = currentEdges[ii][0]
                        fNodeIDs[ii,1] = currentEdges[ii][1] - nodesIn

                    filterNodeIDs[filtNum] = fNodeIDs

                totalNP[name] = np.sum(filterNP_list)
                sortedEdges[name] = filterEdges
                nodeIDs[name] = filterNodeIDs

            # return layerNP, sorted(Tr.edges(data=True))

        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(compute_NP(name))

    return totalNP, sortedEdges, nodeIDs

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
    - eta_c_list         = List of layer-wise critical compression ratios
    - layers_n_shapes = List of layer-wise input and output sizes
    - eta_c_all     = Total compression ratio
    
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
    eta_c_list = []
    
    # convolutional, dense, and total compresstion ratio = (total parameters)/(parameters kept)
    num_conv = 0
    denom_conv = 0
    num_dense = 0
    denom_dense = 0
    num_tot = 0
    denom_tot = 0
    
    # log the input and output shapes for the layers that will be pruned
    layers_n_shapes = {}

    outs = [] # keep track of output dimensions
    if "mnist" in dataset_name:
        if "fc" in model_name:
            """Compute eta_c for FCNN style model wrt MNIST"""
            layer_names_of_interest = ['1','3','5','7','9','11']
            for name, module in model.named_modules():
                if any(x in name for x in layer_names_of_interest):
                    eta_c_list.append(((module.in_features*module.out_features))/
                                      (module.in_features+module.out_features-1))
                    layers_n_shapes[name] = ((module.in_features,module.out_features))
                    num_dense   += (module.in_features*module.out_features)
                    denom_dense += (module.in_features+module.out_features-1)
                    num_tot     += (module.in_features*module.out_features)
                    denom_tot   += (module.in_features+module.out_features-1)
            denom_conv = 1
        elif "conv" in model_name:
            """Compute eta_c for CNN style model wrt MNIST"""
            conv_layers = ['0','2']
            for name, module in model.named_modules():
                mod = module.eval()
                outs.append(out.shape)
                # conv layers
                if any(x in name for x in conv_layers):
                    out = mod(out.clone().detach().float().to(device))
                    if outs[-1][2] != out.shape[2]: # account for downsampling/pooling
                        outs[-1] = torch.rand((1,int(outs[-1][1]*2),int(outs[-1][2]/2),int(outs[-1][3]/2))).shape
                    m = (outs[-1][2]+2)*(outs[-1][3]+2) # dimensions of input
                    n = out.shape[2]*out.shape[3]       # dimensions of output
                    eta_c_list.append((n*(3**2))/(m+n-1))    # kernel size is 3
                    layers_n_shapes[name] = (((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3])))
                    num_conv   += (n*(3**2))
                    denom_conv += (m+n-1)
                    num_tot    += (n*(3**2))
                    denom_tot  += (m+n-1)
                # dense layer
                elif name == "5":
                    eta_c_list.append(((module.in_features*module.out_features))/
                                      (module.in_features+module.out_features-1))
                    layers_n_shapes[name] = ((module.in_features,module.out_features))
                    num_dense   += (module.in_features*module.out_features)
                    denom_dense += (module.in_features+module.out_features-1)
                    num_tot     += (module.in_features*module.out_features)
                    denom_tot   += (module.in_features+module.out_features-1)
            
    elif "cifar10" in dataset_name:
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
                    eta_c_list.append((n*(3**2))/(m+n-1))    # kernel size is 3
                    layers_n_shapes[name] = (((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3]))) # +2 is for padding
                    num_conv   += (n*(3**2))
                    denom_conv += (m+n-1)
                    num_tot    += (n*(3**2))
                    denom_tot  += (m+n-1)
                if 'fc' in name:
                    eta_c_list.append(((module.in_features*module.out_features))/(module.in_features+module.out_features-1))
                    layers_n_shapes[name] = ((module.in_features,module.out_features))
                    num_dense   += (module.in_features*module.out_features)
                    denom_dense += (module.in_features+module.out_features-1)
                    num_tot     += (module.in_features*module.out_features)
                    denom_tot   += (module.in_features+module.out_features-1)
                ind+=1 

        elif 'resnet' in model_name: 
            '''Compute eta_c for ResNet style models wrt CIFAR 10'''
            for name, module in model.named_modules():
                import torch.nn.functional as F
                mod = module.eval()
                outs.append(out.shape)
                if ind+1 < len(names):
                    if 'conv' in name:
                        out = mod(out.clone().detach().float().to(device))
                        if outs[-1][2] != out.shape[2]: # account for downsampling/pooling
                            outs[-1] = torch.rand((1,int(outs[-1][1]*2),int(outs[-1][2]/2),int(outs[-1][3]/2))).shape
                        m = (outs[-1][2]+2)*(outs[-1][3]+2) # dimensions of input
                        n = out.shape[2]*out.shape[3]       # dimensions of output
                        eta_c_list.append((n*(3**2))/(m+n-1))    # kernel size is 3
                        layers_n_shapes[name] = (((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3])))
                        num_conv   += (n*(3**2))
                        denom_conv += (m+n-1)
                        num_tot    += (n*(3**2))
                        denom_tot  += (m+n-1)
                        ind += 1
                    if 'fc' in name: # otherwise it's the dense layer
                        out = F.avg_pool2d(out, out.size()[3])
                        out = out.view(out.size(0), -1)
                        out = mod(out.clone().detach().float().to(device))
                        eta_c_list.append((module.in_features*module.out_features)/(module.in_features+module.out_features-1))
                        layers_n_shapes[name] = ((module.in_features,module.out_features))
                        num_dense   += (module.in_features*module.out_features)
                        denom_dense += (module.in_features+module.out_features-1)
                        num_tot     += (module.in_features*module.out_features)
                        denom_tot   += (module.in_features+module.out_features-1)

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
                            eta_c_list.append((module.in_features*module.out_features)/(module.in_features+module.out_features-1))
                            layers_n_shapes[name] = ((module.in_features,module.out_features))
                            num_dense   += (module.in_features*module.out_features)
                            denom_dense += (module.in_features+module.out_features-1)
                            num_tot     += (module.in_features*module.out_features)
                            denom_tot   += (module.in_features+module.out_features-1)
                elif 'features' in name:#conv layers
                    if name[-1].isnumeric():
                        #Conv operations
                        if any(x in name for x in conv_locations):  
                            mod = module.eval()
                            out = mod(out.clone().detach().float().to(device))
                            m = (outs[-1][2]+2)*(outs[-1][3]+2) # dimensions of input
                            n = out.shape[2]*out.shape[3]       # dimensions of output
                            eta_c_list.append((n*(3**2))/(m+n-1))    # kernel size is 3
                            layers_n_shapes[name] = (((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3])))
                            num_conv   += (n*(3**2))
                            denom_conv += (m+n-1)
                            num_tot    += (n*(3**2))
                            denom_tot  += (m+n-1)
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
                    eta_c_list.append((n*(3**2))/(m+n-1))    # kernel size is 3
                    layers_n_shapes[name] = (((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3])))
                    num_conv   += (n*(3**2))
                    denom_conv += (m+n-1)
                    num_tot    += (n*(3**2))
                    denom_tot  += (m+n-1)
                elif '_x' in name and 'residual' in name:
                    if name[-1] in conv_locations:
                        # all other conv layers that are not in shortut
                        out = mod(out.clone().detach().float().to(device))
                        if outs[-1][2] != out.shape[2]: # account for downsampling
                            outs[-1] = torch.rand((1,int(outs[-1][1]*2),int(outs[-1][2]/2),int(outs[-1][3]/2))).shape
                        m = (outs[-1][2]+2)*(outs[-1][3]+2) # dimensions of input
                        n = out.shape[2]*out.shape[3]       # dimensions of output
                        eta_c_list.append((n*(3**2))/(m+n-1))    # kernel size is 3
                        layers_n_shapes[name] = (((outs[-1][2]+2,outs[-1][3]+2),(out.shape[2],out.shape[3])))
                        num_conv   += (n*(3**2))
                        denom_conv += (m+n-1)
                        num_tot    += (n*(3**2))
                        denom_tot  += (m+n-1)
                elif name == 'fc':
                    # dense layers, will need to change with different sized resnets 
                    # (number of dense layers will change)
                    out = avg_pool(out)
                    out = out.view(out.size(0), -1)
                    out = mod(out.clone().detach().float().to(device))
                    eta_c_list.append((module.in_features*module.out_features)/(module.in_features+module.out_features-1))
                    layers_n_shapes[name] = ((module.in_features,module.out_features))
                    num_dense   += (module.in_features*module.out_features)
                    denom_dense += (module.in_features+module.out_features-1)
                    num_tot     += (module.in_features*module.out_features)
                    denom_tot   += (module.in_features+module.out_features-1)
    
    # compute average compression ratios for conv and dense layers as well as total compression ratio
    eta_c_total = num_tot/denom_tot
    eta_c_conv  = num_conv/denom_conv
    eta_c_dense = num_dense/denom_dense
    
    if verbose:
        print('Critical compression ratios (eta_c) per layer:')
        for n in range(len(eta_c_list)):
            print(str(n)+'   '+list(layers_n_shapes.keys())[n]+': '+str(round(eta_c_list[n],5)))
        print('Average critical compression ratio (eta_c) for Conv Layers: {}'.format(str(round(eta_c_conv,5))))
        print('Average critical compression ratio (eta_c) for Dense Layers: {}'.format(str(round(eta_c_dense,5))))
        print('Net critical compression ratio (eta_c): {}'.format(str(round(eta_c_total,5))))
    return eta_c_list, layers_n_shapes, eta_c_total