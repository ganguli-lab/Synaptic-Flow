import numpy as np
from numpy.linalg import norm
import networkx as nx
import torch
from torch.linalg import norm as torchnorm

def neural_persistence(model):
    totalNP = {}
    sortedEdges = {}

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

                pdMat = torch.hstack((c_arr,d_arr))

                pers = torch.abs(np.diff(pdMat,axis=-1))
                layerNP = norm(pers,ord=2)
                if normalized:
                    layerNP = (layerNP-0)/((cardV-2)**0.5)

                totalNP[name] = layerNP
                sortedEdges[name] = sorted(Tr.edges(data=True))
            # return layerNP, sorted(Tr.edges(data=True))

            if isinstance(module, layers.Conv2d) or isinstance(module, nn.Conv2d): ## Convolutional Layers

                ## extract weights
                ## weights shape = (#filters, #channels, krows, kcols)
                weights_np = module.weights.detach().cpu().numpy()

                nFilters = weights_np.shape[0]
                allWeightsSum = np.sum(np.abs(weights_np),axis=1) ## Absolute weights summed across channels of a filter

                filterNP_list = np.zeros(nFilters)
                filterEdges = {}

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

                    Gr = nx.from_numpy_array(Adj)
                    Tr = nx.maximum_spanning_tree(Gr)

                    d_arr = np.zeros(cardMST).reshape(-1,1)
                    c_arr = np.ones(cardMST).reshape(-1,1)

                    for kk in range(cardMST):
                        d_arr[kk] = sorted(Tr.edges(data=True))[kk][2]['weight']

                    pdMat = torch.hstack((c_arr,d_arr))

                    pers = torch.abs(np.diff(pdMat,axis=-1))
                    filterNP = norm(pers,ord=2)
                    if normalized:
                        filterNP = (filterNP-0)/((cardV-2)**0.5)

                    filterEdges[filtNum] = sorted(Tr.edges(data=True))

                totalNP[name] = np.sum(filterNP_list)
                sortedEdges[name] = filterEdges
            # return layerNP, sorted(Tr.edges(data=True))

        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(compute_NP(name))

    return totalNP, sortedEdges
