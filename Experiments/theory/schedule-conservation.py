import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Layers import layers
from Utils import load
from Utils import generator
from train import *
from prune import *

def run(args):
    if not args.save:
        print("This experiment requires an expid.")
        quit()

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    input_shape, num_classes = load.dimension(args.dataset) 
    data_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)

    ## Model, Loss, Optimizer ##
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(device)
    loss = nn.CrossEntropyLoss()
    torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))


    def score(parameters, model, loss, dataloader, device):
        @torch.no_grad()
        def linearize(model):
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs
        @torch.no_grad()
        def nonlinearize(model, signs):
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)
        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)
        output = model(input)
        maxflow = torch.sum(output)
        maxflow.backward()
        scores = {}
        for _, p in parameters:
            scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()
        nonlinearize(model, signs)

        return scores, maxflow.item()

    def mask(parameters, scores, sparsity):
        global_scores = torch.cat([torch.flatten(v) for v in scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        cutsize = 0
        if not k < 1:
            cutsize = torch.sum(torch.topk(global_scores, k, largest=False).values).item()
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in parameters:
                score = scores[id(param)]
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
        return cutsize

    @torch.no_grad()
    def apply_mask(parameters):
        for mask, param in parameters:
            param.mul_(mask)

    results = []
    for style in ['linear', 'exponential']:
        print(style)
        sparsity_ratios = []
        for i, exp in enumerate(args.compression_list):
            max_ratios = []
            for j, epochs in enumerate(args.prune_epoch_list):
                model.load_state_dict(torch.load("{}/model.pt".format(args.result_dir), map_location=device))
                parameters = list(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
                model.eval()
                ratios = []
                for epoch in tqdm(range(epochs)):
                    apply_mask(parameters)
                    scores, maxflow = score(parameters, model, loss, data_loader, device)
                    sparsity = 10**(-float(exp))
                    if style == 'linear':
                        sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
                    if style == 'exponential':
                        sparse = sparsity**((epoch + 1) / epochs)
                    cutsize = mask(parameters, scores, sparse)
                    ratios.append(cutsize / maxflow)
                max_ratios.append(max(ratios))
            sparsity_ratios.append(max_ratios)
        results.append(sparsity_ratios)
    np.save('{}/ratios'.format(args.result_dir), np.array(results))
