def frac_overlap(W,p):

    W = W.cpu().numpy()

    m,n = W.shape

    W_tot = m*n
    W_keep = int(p*m*n)

    k = m+n-1
    diff = W_keep - k

    rand_block = np.zeros((m-1)*(n-1))
    rand_block[:diff] = 1
    rand_block = permutation(rand_block)
    rand_block = rand_block.reshape(m-1,n-1)

    mask = np.zeros((m,n))
    mask[:,0] = 1
    mask[0,:] = 1
    mask[1:,1:] = rand_block

    W_masked = W*mask
    W_masked_N = np.abs(W_masked)
    wM = np.max(W_masked_N)
    W_masked_N = W_masked_N/wM

    layerNP, MST = layer_neural_persistence(W_masked)

    MST_weights = np.zeros(k)
    for ii in range(k):
        MST_weights[ii] = MST[ii][2]['weight']

    topk = np.flipud(np.sort(W_masked_N.ravel()))[:k]

    frac_over = len(np.intersect1d(topk,MST_weights))/k
    print(p,'-->',frac_over)

    return frac_over
