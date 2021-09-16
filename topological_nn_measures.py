import numpy as np
from numpy.linalg import norm
import networkx as nx

def layers_to_big_adjacency(layerTuple):
    ll = len(layerTuple)
    dims = []

    for ii in range(ll):
        dims.append(layerTuple[ii].shape)

    dims = np.array(dims).ravel()
    dims_final = np.append(dims[::2],dims[-1])
    D = np.sum(dims_final)

    bigAdj = np.zeros((D,D))

    for ii in range(ll):


    return bigAdj

def layer_neural_persistence(W=None,NPType='NP',Adj=None,normalized=True):

    if W is not None:
        d1, d2 = W.shape

        wm = np.max(np.abs(W))
        W_prime = np.abs(W)/(wm)

        cardV = np.sum(d1+d2)
        cardMST = cardV - 1

    if Adj is None:
        Adj = np.zeros((cardV,cardV))

        for ii in range(d1):
            for jj in range(d2):
                Adj[d1+jj,ii] = W_prime[ii,jj]

        Adj += Adj.T

    cardV, cardV = Adj.shape
    cardMST = cardV - 1

    Gr = nx.from_numpy_array(Adj)
    Tr = nx.maximum_spanning_tree(Gr)

    d_arr = np.zeros(cardMST).reshape(-1,1)
    c_arr = np.ones(cardMST).reshape(-1,1)

    for kk in range(cardMST):
        d_arr[kk] = sorted(Tr.edges(data=True))[kk][2]['weight']

    pdMat = np.hstack((c_arr,d_arr))

    if NPType == 'NP':
        pers = np.abs(np.diff(pdMat,axis=-1))
        layerNP = norm(pers,ord=2)
        if normalized:
            layerNP = (layerNP-0)/((cardV-2)**0.5)
    elif NPType == 'EP':
        layerNP = norm(pdMat,ord='fro')
        if normalized:
            layerNP = (layerNP-(cardV**0.5))/((2*cardMST)**0.5)

    return layerNP, sorted(Tr.edges(data=True))
