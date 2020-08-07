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

    ## Pre-Train ##
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, args.pre_epochs, args.verbose)
    pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
    
    ## Save Original ##
    torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
    torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
    torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

    ## Prune and Fine-Tune##
    for compression in args.compression_list:
        for p, p_epochs in zip(args.pruner_list, args.prune_epoch_list):
            print('{} compression ratio, {} pruners'.format(compression, p))
            
            # Reset Model, Optimizer, and Scheduler
            model.load_state_dict(torch.load("{}/model.pt".format(args.result_dir), map_location=device))
            optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
            scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))

            # Prune Model
            pruner = load.pruner(p)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
            sparsity = 10**(-float(compression))
            prune_loop(model, loss, pruner, prune_loader, device, sparsity,
                       args.compression_schedule, args.mask_scope, p_epochs, args.reinitialize, args.prune_train_mode)   
            
            # Prune Result
            prune_result = metrics.summary(model, 
                                           pruner.scores,
                                           metrics.flop(model, input_shape, device),
                                           lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))

            # Train Model
            post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                          test_loader, device, args.post_epochs, args.verbose)
            
            # Save Data
            post_result.to_pickle("{}/post-train-{}-{}-{}.pkl".format(args.result_dir, p, str(compression), p_epochs))
            prune_result.to_pickle("{}/compression-{}-{}-{}.pkl".format(args.result_dir, p, str(compression), p_epochs))

