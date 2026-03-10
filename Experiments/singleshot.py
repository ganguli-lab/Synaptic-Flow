import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *
import pickle
import time

def revise_masks(model, bias=False, batchnorm=False, residual=False):
    """
    Flip all pruning masks in the model:
    1 → 0, 0 → 1
    Works in-place on the model's registered buffers.
    """
    for i, (mask, _) in enumerate(generator.masked_parameters(model, bias, batchnorm, residual)):
        # print(i)
        if i != 2:
            mask.data = mask.data
        else:
            mask.data = torch.ones_like(mask.data)

def flip_all_masks(model, bias=False, batchnorm=False, residual=False):
    """
    Flip all pruning masks in the model:
    1 → 0, 0 → 1
    Works in-place on the model's registered buffers.
    """
    # Get all masked parameters first
    masked_params = list(generator.masked_parameters(model, bias, batchnorm, residual))

    for i, (mask, _) in enumerate(masked_params):
        mask.data = 1 - mask.data
        
        # if i != len(masked_params) - 1:
        #     mask.data = 1 - mask.data  # invert mask
        # else:
        #     mask.data = torch.ones_like(mask.data)  # last one set to all ones

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)
    
    acc_wo_ft = []
    acc = []
    removed_num = []
    acc_flip = []
    removed_num_flip = []
    x_axis = None

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset) 
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes, name = args.dataset + '_val')
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers, name = args.dataset + '_train')
    test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers, name = args.dataset + '_val')


    ## Train-Prune Loop ##
    for compression in args.compression_list:
        print('Compression: ', compression)
        ## Model, Loss, Optimizer ##
        print('Creating {}-{} model.'.format(args.model_class, args.model))
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

        # tick = time.time()
        # torch.cuda.synchronize(device)
        # # 🔹 reset peak stats for this layer
        # torch.cuda.reset_peak_memory_stats(device)
        # torch.cuda.synchronize(device)

        ## Prune ##
        print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
        pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
        sparsity = 10**(-float(compression))
        prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
                args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)
        
        # end_t = time.time()
        # print(f'time for one sparsity = {end_t-tick}')
        # # 🔹 timing end
        # torch.cuda.synchronize(device)

        # 🔹 memory stats
        # alloc = torch.cuda.memory_allocated(device) / 1024**2
        # reserved = torch.cuda.memory_reserved(device) / 1024**2
        # peak = torch.cuda.max_memory_allocated(device) / 1024**2

        # # 🔹 write per-layer log
        # print(f"{alloc:9.1f} | {reserved:11.1f} | {peak:8.1f}\n")
            
        
        # flip mask
        # revise_masks(model, args.prune_bias, args.prune_batchnorm, args.prune_residual)

        ## Post-Train ##
        print('Post-Training for {} epochs.'.format(args.post_epochs))
        post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                    test_loader, device, args.post_epochs, args.verbose) 

        ## Display Results ##
        frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
        train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
        prune_result, total_fc_remaining, total_fc_full, total_removed_e = metrics.summary(model, 
                                    pruner.scores,
                                    metrics.flop(model, input_shape, device),
                                    lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual), pruner.masked_parameters)

        # print(total_fc_full, total_fc_remaining, total_removed_e, total_fc_full-total_fc_remaining)
        # print(total_fc_full)
        third_row = train_result.loc['Post-Prune']
        final_row = train_result.loc['Final']
        acc.append(final_row['top1_accuracy'].values[0]/100)
        acc_wo_ft.append(third_row['top1_accuracy'].values[0]/100)
        removed_num.append(total_removed_e)
                
        
        total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
        possible_params = prune_result['size'].sum()
        
        print(f"Removing {total_removed_e} edges...")
        print("Train results:\n", train_result)
        # print("Prune results:\n", prune_result)
        # print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
        print()
        
        init = train_result.loc['Pre-Prune']['top1_accuracy'].values[0]
        final = train_result.loc['Post-Prune']['top1_accuracy'].values[0]
        
        if (((init-final)) > 2.5):
            break
        else:
            # print(model.state_dict())
            torch.save(model.state_dict(), "Results/" + "vgg9_10_ori_relu_s2_pruned.pth")
        
        # flip mask
        flip_all_masks(model, args.prune_bias, args.prune_batchnorm, args.prune_residual)

        ## Post-Train ##
        print('Post-Training for {} epochs.'.format(args.post_epochs))
        post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                    test_loader, device, args.post_epochs, args.verbose) 

        ## Display Results ##
        frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
        train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
        prune_result, total_fc_remaining, total_fc_full, total_removed_e = metrics.summary(model, 
                                    pruner.scores,
                                    metrics.flop(model, input_shape, device),
                                    lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual), pruner.masked_parameters)

        third_row = train_result.loc['Post-Prune']
        final_row = train_result.loc['Final']
        acc_flip.append(final_row['top1_accuracy'].values[0]/100)
        removed_num_flip.append(total_removed_e)
        
        
        print(f"Removing {total_removed_e} edges...")
        print("Train results:\n", train_result)
        print()
        # print("Prune results:\n", prune_result)
        # input()
        # print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
        # print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

        ## Save Results and Model ##
        if args.save:
            print('Saving results.')
            pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
            post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
            prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
            torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
            torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
            torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))
            
    x_axis = list(np.linspace(0, max(removed_num), num=10, dtype=int))
    metrics.plot_curve(acc, acc_flip, removed_num, removed_num_flip, str(args.model) + str(args.pruner) + str(args.dataset), "Results/vgg16/", x_axis = x_axis)

    # pack into a dictionary
    data = {
        "clean_acc": acc,
        "flip_clean_acc": acc_flip,
        "remove_num": removed_num,
        "flip_remove_num": removed_num_flip,
    }

    # save to pickle file
    with open("Results/" + str(args.model) + str(args.pruner) + "_results.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Saved variables to results.pkl")
