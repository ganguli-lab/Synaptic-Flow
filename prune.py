from tqdm import tqdm
import torch

def prune_loop(model, loss, pruner, dataloader, device,
               sparsity, normalize, scope, epochs, reinitialize=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    model.eval()
    for epoch in tqdm(range(epochs)):
        pruner.apply_mask()
        pruner.score(model, loss, dataloader, device)
        pruner.process(normalize)
        sparse = sparsity**((epoch + 1) / epochs) # Exponential
        # sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs) # Linear
        pruner.mask(sparse, scope)
    if reinitialize:
        model._initialize_weights()