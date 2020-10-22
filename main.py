import argparse
import json
import os
from Experiments import singleshot
from Experiments import multishot
from Experiments.theory import unit_conservation
from Experiments.theory import layer_conservation
from Experiments.theory import imp_conservation
from Experiments.theory import schedule_conservation

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Network Compression')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','tiny-imagenet','imagenet'],
                        help='dataset (default: mnist)')
    training_args.add_argument('--model', type=str, default='fc', choices=['fc','conv',
                        'vgg11','vgg11-bn','vgg13','vgg13-bn','vgg16','vgg16-bn','vgg19','vgg19-bn',
                        'resnet18','resnet20','resnet32','resnet34','resnet44','resnet50',
                        'resnet56','resnet101','resnet110','resnet110','resnet152','resnet1202',
                        'wide-resnet18','wide-resnet20','wide-resnet32','wide-resnet34','wide-resnet44','wide-resnet50',
                        'wide-resnet56','wide-resnet101','wide-resnet110','wide-resnet110','wide-resnet152','wide-resnet1202'],
                        help='model architecture (default: fc)')
    training_args.add_argument('--model-class', type=str, default='default', choices=['default','lottery','tinyimagenet','imagenet'],
                        help='model class (default: default)')
    training_args.add_argument('--dense-classifier', type=bool, default=False,
                        help='ensure last layer of model is dense (default: False)')
    training_args.add_argument('--pretrained', type=bool, default=False,
                        help='load pretrained weights (default: False)')
    training_args.add_argument('--optimizer', type=str, default='adam', choices=['sgd','momentum','adam','rms'],
                        help='optimizer (default: adam)')
    training_args.add_argument('--train-batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    training_args.add_argument('--test-batch-size', type=int, default=256,
                        help='input batch size for testing (default: 256)')
    training_args.add_argument('--pre-epochs', type=int, default=0,
                        help='number of epochs to train before pruning (default: 0)')
    training_args.add_argument('--post-epochs', type=int, default=10,
                        help='number of epochs to train after pruning (default: 10)')
    training_args.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    training_args.add_argument('--lr-drops', type=int, nargs='*', default=[],
                        help='list of learning rate drops (default: [])')
    training_args.add_argument('--lr-drop-rate', type=float, default=0.1,
                        help='multiplicative factor of learning rate drop (default: 0.1)')
    training_args.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    # Pruning Hyperparameters
    pruning_args = parser.add_argument_group('pruning')
    pruning_args.add_argument('--pruner', type=str, default='rand', 
                        choices=['rand','mag','snip','grasp','synflow'],
                        help='prune strategy (default: rand)')
    pruning_args.add_argument('--compression', type=float, default=0.0,
                        help='quotient of prunable non-zero prunable parameters before and after pruning (default: 1.0)')
    pruning_args.add_argument('--prune-epochs', type=int, default=1,
                        help='number of iterations for scoring (default: 1)')
    pruning_args.add_argument('--compression-schedule', type=str, default='exponential', choices=['linear','exponential'],
                        help='whether to use a linear or exponential compression schedule (default: exponential)')
    pruning_args.add_argument('--mask-scope', type=str, default='global', choices=['global','local'],
                        help='masking scope (global or layer) (default: global)')
    pruning_args.add_argument('--prune-dataset-ratio', type=int, default=10,
                        help='ratio of prune dataset size and number of classes (default: 10)')
    pruning_args.add_argument('--prune-batch-size', type=int, default=256,
                        help='input batch size for pruning (default: 256)')
    pruning_args.add_argument('--prune-bias', type=bool, default=False,
                        help='whether to prune bias parameters (default: False)')
    pruning_args.add_argument('--prune-batchnorm', type=bool, default=False,
                        help='whether to prune batchnorm layers (default: False)')
    pruning_args.add_argument('--prune-residual', type=bool, default=False,
                        help='whether to prune residual connections (default: False)')
    pruning_args.add_argument('--prune-train-mode', type=bool, default=False,
                        help='whether to prune in train mode (default: False)')
    pruning_args.add_argument('--reinitialize', type=bool, default=False,
                        help='whether to reinitialize weight parameters after pruning (default: False)')
    pruning_args.add_argument('--shuffle', type=bool, default=False,
                        help='whether to shuffle masks after pruning (default: False)')
    pruning_args.add_argument('--invert', type=bool, default=False,
                        help='whether to invert scores during pruning (default: False)')
    pruning_args.add_argument('--pruner-list', type=str, nargs='*', default=[],
                        help='list of pruning strategies for singleshot (default: [])')
    pruning_args.add_argument('--prune-epoch-list', type=int, nargs='*', default=[],
                        help='list of prune epochs for singleshot (default: [])')
    pruning_args.add_argument('--compression-list', type=float, nargs='*', default=[],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
    pruning_args.add_argument('--level-list', type=int, nargs='*', default=[],
                        help='list of number of prune-train cycles (levels) for multishot (default: [])')
    ## Experiment Hyperparameters ##
    parser.add_argument('--experiment', type=str, default='singleshot', 
                        choices=['singleshot','multishot','unit-conservation',
                        'layer-conservation','imp-conservation','schedule-conservation'],
                        help='experiment name (default: example)')
    parser.add_argument('--expid', type=str, default='',
                        help='name used to save results (default: "")')
    parser.add_argument('--result-dir', type=str, default='Results/data',
                        help='path to directory to save results (default: "Results/data")')
    parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')
    parser.add_argument('--workers', type=int, default='4',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--verbose', action='store_true',
                        help='print statistics during training and testing')
    args = parser.parse_args()


    ## Construct Result Directory ##
    if args.expid == "":
        print("WARNING: this experiment is not being saved.")
        setattr(args, 'save', False)
    else:
        result_dir = '{}/{}/{}'.format(args.result_dir, args.experiment, args.expid)
        setattr(args, 'save', True)
        setattr(args, 'result_dir', result_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            val = ""
            while val not in ['yes', 'no']:
                val = input("Experiment '{}' with expid '{}' exists.  Overwrite (yes/no)? ".format(args.experiment, args.expid))
            if val == 'no':
                quit()

    ## Save Args ##
    if args.save:
        with open(args.result_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, sort_keys=True, indent=4)

    ## Run Experiment ##
    if args.experiment == 'singleshot':
        singleshot.run(args)
    if args.experiment == 'multishot':
        multishot.run(args)
    if args.experiment == 'unit-conservation':
    	unit_conservation.run(args)
    if args.experiment == 'layer-conservation':
        layer_conservation.run(args)
    if args.experiment == 'imp-conservation':
        imp_conservation.run(args)
    if args.experiment == 'schedule-conservation':
        schedule_conservation.run(args)

