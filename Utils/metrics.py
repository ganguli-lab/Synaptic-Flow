import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from prune import * 
from Layers import layers


def summary(model, scores, flops, prunable, masked_parameters):
    r"""Summary of compression results for a model.
    Also returns:
      - total_fc_remaining: total number of remaining (unpruned) weights in all FC layers
      - total_fc_full: total number of weights before pruning in all FC layers
    """
    rows = []
    total_fc_remaining = 0
    total_fc_full = 0
    
    total_removed_e = 0
    total_removed_e_flip = 0

    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            pruned = prunable(module) and id(param) in scores.keys()

            # Compute sparsity and score
            if pruned:
                sparsity = getattr(module, pname + '_mask').detach().cpu().numpy().mean()
                score = scores[id(param)].detach().cpu().numpy()
                
            else:
                sparsity = 1.0
                score = np.zeros(1)

            shape = param.detach().cpu().numpy().shape
            flop = 0 # flops[name][pname]

            # Compute score statistics
            score_mean = score.mean()
            score_var = score.var()
            score_sum = score.sum()
            score_abs_mean = np.abs(score).mean()
            score_abs_var = np.abs(score).var()
            score_abs_sum = np.abs(score).sum()

            rows.append([
                name, pname, sparsity, np.prod(shape), shape, flop,
                score_mean, score_var, score_sum,
                score_abs_mean, score_abs_var, score_abs_sum,
                pruned
            ])
            
            # Count total and remaining weights in fully connected (fc) layers
            if 'weight' in pname.lower():
                total_params = np.prod(shape)
                remaining = total_params * sparsity
                total_fc_full += total_params
                total_fc_remaining += remaining

            

    columns = [
        'module', 'param', 'sparsity', 'size', 'shape', 'flops',
        'score mean', 'score variance', 'score sum',
        'score abs mean', 'score abs variance', 'score abs sum',
        'prunable'
    ]
    
    for i, (mask, param) in enumerate(masked_parameters):
        if i in [0]:
            total_removed_e += ((mask == 0).sum().item()*1)
        elif i in [1]:
            total_removed_e += ((mask == 0).sum().item()*1)
        else:
            total_removed_e += (mask == 0).sum().item()
        print(f'Removed {(mask == 0).sum().item()}, remaining {(mask == 1).sum().item()}')

    df = pd.DataFrame(rows, columns=columns)
    return df, total_fc_remaining, total_fc_full, total_removed_e


def flop(model, input_shape, device):

    total = {}
    def count_flops(name):
        def hook(module, input, output):
            flops = {}
            if isinstance(module, layers.Linear) or isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                flops['weight'] = in_features * out_features
                if module.bias is not None:
                    flops['bias'] = out_features
            if isinstance(module, layers.Conv2d) or isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = int(np.prod(module.kernel_size))
                output_size = output.size(2) * output.size(3)
                flops['weight'] = in_channels * out_channels * kernel_size * output_size
                if module.bias is not None:
                    flops['bias'] = out_channels * output_size
            if isinstance(module, layers.BatchNorm1d) or isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    flops['weight'] = module.num_features
                    flops['bias'] = module.num_features
            if isinstance(module, layers.BatchNorm2d) or isinstance(module, nn.BatchNorm2d):
                output_size = output.size(2) * output.size(3)
                if module.affine:
                    flops['weight'] = module.num_features * output_size
                    flops['bias'] = module.num_features * output_size
            if isinstance(module, layers.Identity1d):
                flops['weight'] = module.num_features
            if isinstance(module, layers.Identity2d):
                output_size = output.size(2) * output.size(3)
                flops['weight'] = module.num_features * output_size
            total[name] = flops
        return hook
    
    for name, module in model.named_modules():
        module.register_forward_hook(count_flops(name))

    input = torch.ones([1] + list(input_shape)).to(device)
    model(input)

    return total


def plot_curve(
    neg_clean_acc, pos_clean_acc,
    neg_remove_num, pos_remove_num,
    label, res_path,
    neg_freq_labels=None, pos_freq_labels=None, x_axis=None
):
    # Colors
    neg_color = "#00E013"
    pos_color = "#EC6A00"
    text_color = "#012DF1"

    plt.figure(figsize=(10, 6))

    # Plot lines
    plt.plot(neg_remove_num, neg_clean_acc, label='Small weights removed first',
             marker='o', linestyle='--', linewidth=3., markersize=13, color=pos_color)

    plt.plot(pos_remove_num, pos_clean_acc, label='Large weights removed first',
             marker='x', linestyle='-', linewidth=3., markersize=13, color=neg_color)

    # Annotate frequencies BELOW points
    if neg_freq_labels:
        for i, (x, y, r) in enumerate(zip(neg_remove_num, neg_clean_acc, neg_freq_labels)):
            if (i % 1 == 0):
                plt.annotate(r, (x, y), textcoords='offset points',
                            xytext=(-10, 5), ha='left', fontsize=18, color='#000000')

    flag = 0
    
    if pos_freq_labels:
        for i, (x, y, r) in enumerate(zip(pos_remove_num, pos_clean_acc, pos_freq_labels)):
            if (i % 5 == 0) or (i == len(pos_remove_num)-1):
                plt.annotate(r, (x, y), textcoords='offset points',
                    xytext=(0, 15), ha='center', va='bottom', fontsize=22, color=text_color)
                
            # elif (r < 0) and (~flag):
            #     plt.annotate(r, (x, y), textcoords='offset points',
            #         xytext=(0, 15), ha='center', fontsize=28, color=pos_color)
            #     flag = 1

    # Labels and title
    plt.xlabel('Number of Parameters Removed', fontsize=28, fontweight='semibold')
    plt.ylabel('Accuracy', fontsize=31, fontweight='semibold')
    
    # plt.title('Accuracy vs. Edge Removal Count', fontsize=28, fontweight='semibold')
    plt.ylim(0.0, 1.0)

    # Set scientific notation on x-axis
    ax = plt.gca()
    
    # Override the x-axis ticks/labels if `x_axis` is given
    if x_axis is not None:
        # Compute exponent (e.g., 1e+3, 1e+4) based on the max value
        exponent = int(np.floor(np.log10(max(x_axis))))
        scale = 10 ** exponent

        # Scale values and format tick labels as mantissas only
        scaled_ticks = [x / scale for x in x_axis]
        mantissa_labels = [f"{v:.1f}" for v in scaled_ticks]

        # Set the ticks and the scaled mantissa labels
        plt.xticks(ticks=x_axis, labels=mantissa_labels, fontsize=26, fontweight='semibold')

        # Add scientific scale as offset text (e.g., ×1e4) to the end of the x-axis
        ax.annotate(
            f"×1e{exponent}",
            xy=(1.0, 0.0), xycoords='axes fraction',  # Right end of x-axis
            xytext=(10, -35), textcoords='offset points',  # Just below and slightly to the left
            ha='right', va='top',
            fontsize=18, fontweight='semibold'
        )
    else:
        plt.xticks(fontsize=22, fontweight='semibold')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(24)
        ax.xaxis.get_offset_text().set_fontweight('semibold')

    # Ticks
    plt.xticks(fontsize=24, fontweight='semibold')
    plt.yticks(fontsize=26, fontweight='semibold')
    
    # === Axis borders ===
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')

    # Grid and legend
    plt.grid(True, linestyle='--', linewidth=2.5, color='gray', alpha=0.85)
    legend = plt.legend(fontsize=24, loc=0)  # create the legend
    for text in legend.get_texts():
        text.set_fontweight('semibold')  # or 'bold'

    plt.tight_layout()
    plt.savefig(os.path.join(res_path, f'{label}_curve_all.png'), dpi=400, bbox_inches="tight")
    plt.close()