from tqdm import tqdm
import torch
import numpy as np

def prune_loop(model, loss, pruner, dataloader, device,
               sparsity, linear_schedule, scope, epochs, reinitialize=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    model.eval()
    for epoch in tqdm(range(epochs)):
        pruner.apply_mask()
        pruner.score(model, loss, dataloader, device)
        if linear_schedule:
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs) # Linear
        else:
            sparse = sparsity**((epoch + 1) / epochs) # Exponential
        pruner.mask(sparse, scope)
    if reinitialize:
        model._initialize_weights()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 1:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
        quit()