import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
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
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset) 
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)

    ## Model ##
    print('Creating {} model.'.format(args.model))
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier,
                                                     args.pretrained).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    ## Save Original ##
    torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
    torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
    torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

    ## Train-Prune Loop ##
    comp_exponents = np.arange(0, 6, 1) # log 10 of inverse sparsity
    train_iterations = [100] # number of epochs between prune periods
    prune_iterations = np.arange(1, 6, 1) # number of prune periods
    for i, comp_exp in enumerate(comp_exponents[::-1]):
        for j, train_iters in enumerate(train_iterations):
            for k, prune_iters in enumerate(prune_iterations):
                print('{} compression ratio, {} train epochs, {} prune iterations'.format(comp_exp, train_iters, prune_iters))
                
                # Reset Model, Optimizer, and Scheduler
                model.load_state_dict(torch.load("{}/model.pt".format(args.result_dir), map_location=device))
                optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
                scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))
                
                for l in range(prune_iters):

                    # Pre Train Model
                    train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                    test_loader, device, train_iters, args.verbose)

                    # Prune Model
                    pruner = load.pruner('mag')(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
                    sparsity = (10**(-float(comp_exp)))**((l + 1) / prune_iters)
                    prune_loop(model, loss, pruner, prune_loader, device, sparsity,
                               args.normalize_score, args.mask_scope, 1, args.reinitialize)

                    # Reset Model's Weights
                    original_dict = torch.load("{}/model.pt".format(args.result_dir), map_location=device)
                    original_weights = dict(filter(lambda v: (v[1].requires_grad == True), model.state_dict().items()))
                    model_dict = model.state_dict()
                    model_dict.update(original_weights)
                    model.load_state_dict(model_dict)
                    
                    # Reset Optimizer and Scheduler
                    optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
                    scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))

                # Prune Result
                prune_result = metrics.summary(model, 
                                               pruner.scores,
                                               metrics.flop(model, input_shape, device),
                                               lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
                # Train Model
                post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                              test_loader, device, args.post_epochs, args.verbose)
                
                # Save Data
                prune_result.to_pickle("{}/compression-{}-{}-{}.pkl".format(args.result_dir, str(comp_exp), str(train_iters), str(prune_iters)))
                post_result.to_pickle("{}/performance-{}-{}-{}.pkl".format(args.result_dir, str(comp_exp), str(train_iters), str(prune_iters)))


    
