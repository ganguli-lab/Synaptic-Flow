import numpy as np
from numpy.linalg import norm
import networkx as nx
import torch

def neural_persistence(model):
    total = {}
    def compute_NP(name):
        def hook(module, input, output):
            NPs = {}
            if isinstance(module, layers.Linear) or isinstance(module, nn.Linear):
                """
                Uses networkx to compute the MST
                """
                # extract weights
                weights = module.weights

                # find the largest weight
                h_max = torch.max(weights)

                # normalize by the largest weight
                W_prime = torch.abs(weights/h_max)

                d1, d2 = weights.shape

                cardV = np.sum(d1+d2)
                cardMST = cardV - 1

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

                pers = np.abs(np.diff(pdMat,axis=-1))
                layerNP = norm(pers,ord=2)
                if normalized:
                    layerNP = (layerNP-0)/((cardV-2)**0.5)

            return layerNP, sorted(Tr.edges(data=True))


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

                # determine largest absolute weight
                h_max = torch.max(weights)

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
                    NPs.append((1, h_corner[c])
                    t+=1

                # create the remaining tuples
                while 1 do:
                    # if current weight is corner, write one less tuple
                    if i in corner_idx:
                        n_prime = n - 1
                    else:
                        n_prime = n

                    # if there are at least n' ,or tuples, set merge value to sorted[i]
                    if t+n_prime <= tau:
                        for j in range(n_prime):
                            NPs.append((1,h_sort[i]))
                        t+=n_prime
                        i+=1
                    else:
                        for j in range(tau-t):
                            NPs.append((1, h_sort[i]))
                        break
                # Compute norm of approximated persistence diagram
                NPs = torch.norm(NPs, 2)

            total[name] = NPs
        return hook
    for name, module in model.named_modules():
        module.register_forward_hook(compute_NP(name))
    return total
