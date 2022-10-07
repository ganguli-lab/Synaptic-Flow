import numpy as np
from numpy.linalg import norm
import networkx as nx
import torch
import multiprocessing as mp

# def layers_to_big_adjacency(layerTuple):
#     ll = len(layerTuple)
#     dims = []

#     for ii in range(ll):
#         dims.append(layerTuple[ii].shape)

#     dims = np.array(dims).ravel()
#     dims_final = np.append(dims[::2],dims[-1])
#     D = np.sum(dims_final)

#     bigAdj = np.zeros((D,D))

#     for ii in range(ll):


#     return bigAdj

def layer_neural_persistence(W=None,NPType='NP',Adj=None,normalized=True, in_out_sizes = ((1,1),(1,1))):
    #Inputs: W: Weight matrix
    # NPType: Score for entire layer, neural persistence
    # Adj: If there is a precomputed adjacency matrix
    # normalized: If the neural persistence calc is to be normalized
    # in_out_sizes: tuple of sizes of inputs and outputs to layer
    
    if W is not None:
        #For dense layers
        if len(W.shape) == 2:
            d1, d2 = W.shape

            wm = np.max(np.abs(W)) #Normalizing weight matrix
            W_prime = np.abs(W)/(wm)

            cardV = np.sum(d1+d2) #Cardinality
            cardMST = cardV - 1

            #Adjacency matrix calculation if needed
            if Adj is None:
                Adj = np.zeros((cardV,cardV))

                for ii in range(d1):
                    for jj in range(d2):
                        Adj[d1+jj,ii] = W_prime[ii,jj]

                Adj += Adj.T

            cardV, cardV = Adj.shape
            cardMST = cardV - 1 #Calculate cardinality of MST

            #Build spanning tree of graph, graph built from adjacency matrix
            Gr = nx.from_numpy_array(Adj)
            Tr = nx.maximum_spanning_tree(Gr)

            #death times (d_arr), creation times (c_arr), persistence diagram
            d_arr = np.zeros(cardMST).reshape(-1,1)
            c_arr = np.ones(cardMST).reshape(-1,1) #Everything is born at 1

            for kk in range(cardMST):
                d_arr[kk] = sorted(Tr.edges(data=True))[kk][2]['weight'] #Sort based on persistence, giving death times

            pdMat = np.hstack((c_arr,d_arr)) #persistence diagram matrix
            
            # formerly outputNP
            if NPType == 'NP': #Neural persistence 
                pers = np.abs(np.diff(pdMat,axis=-1))
                outputNP = norm(pers,ord=2)
                if normalized:
                    outputNP = (outputNP-0)/((cardV-2)**0.5)
            elif NPType == 'EP': #Edge persistence (unfinished trial work)
                outputNP = norm(pdMat,ord='fro')
                if normalized:
                    outputNP = (outputNP-(cardV**0.5))/((2*cardMST)**0.5)
        
        else: # Convolutional Layer, similar order of steps as earlier except w.r.t. looping over filters and Toeplitz matrix building
            nFilters = W.shape[0]
            allWeightsSum = np.sum(np.abs(W),axis=1) ## Absolute weights summed across channels of a filter

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

                #Final toeplitz matrix
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
                Tr = nx.maximum_spanning_tree(Gr) #TODO possible GPU usage

                # d_arr = np.zeros(cardMST).reshape(-1,1)
                c_arr = np.ones(cardMST).reshape(-1,1)
                d_arr = []

                # for kk in range(cardMST):
                #     d_arr[kk] = sorted(Tr.edges(data=True))[kk][2]['weight']
                
                ## Parallelized for speedup
                d_arr = []
                pool = mp.Pool(mp.cpu_count())
                d_arr = [pool.apply(sort_edges, args = (d_arr, Gr, kk)) for kk in range(cardMST)]
                
                pdMat = np.hstack((c_arr,np.array(d_arr)))

                pers = np.abs(np.diff(pdMat,axis=-1))
                filterNP = norm(pers,ord=2)
                if normalized:
                    filterNP = (filterNP-0)/((cardV-2)**0.5)
                filterNP_list[filtNum] = filterNP
            outputNP = np.sum(filterNP_list)
            #OutputNP: Single layer 'score'
            #sorted(Tr...) is the score for layer weights -> TODO make sure conv output is in the required shape for a Pruner
    return outputNP#, sorted(Tr.edges(data=True))

def sort_edges(d_arr, Gr, kk):
    import networkx as nx
    Tr = nx.maximum_spanning_tree(Gr)
    d_arr.append(sorted(Tr.edges(data=True))[kk][2]['weight'])
    return d_arr