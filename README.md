# Synaptic Flow


## Getting Started
First clone this repo, then install all dependencies
```
pip install -r requirements.txt
```
The code was tested with Python 3.6.0.

## Code Base
Below is a description of the major sections of the code base.

### Datasets
This code base supports the following datasets: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet, ImageNet. 

All datasets except Tiny ImagNet and ImageNet will download automatically.  For Tiny ImageNet, download the data directly from [https://tiny-imagenet.herokuapp.com](https://tiny-imagenet.herokuapp.com), move the unzipped folder ``tiny-imagnet-200`` into the ```Data``` folder, run the script `python Utils/tiny-imagenet-setup.py` from the home folder. For ImageNet, first copy the file ```/mnt/fs1/chengxuz/imagenet_raw.tar.gz``` into the node's local data storage ```/data5/kunin/Dataset/```.  Then uncompress the file with ```gunzip -c imagenet_raw.tar.gz | tar xopf -```.

### Models

####  Default Models

This model class supports a basic dense and convolutional model.  These models work with MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet, but do not support pretrained parameters. 

<!---
####  ShrinkBench Models
This code base supports the following models based on [ShrinkBench](https://shrinkbench.github.io/): VGG-16 BN, ResNet-20, ResNet-32, ResNet-44, ResNet-56, ResNet-110, ResNet-1202. These models work with MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet. These models have pretrained weights on CIFAR-10 from [ShrinkBench](https://shrinkbench.github.io/).
--->
####  Lottery Ticket Hypothesis Models
This code base supports the following models based on [OpenLTH](https://github.com/facebookresearch/open_lth): .


####  Tiny ImageNet Models
This code base supports the following models based on this Github [repository](https://github.com/weiaicunzai/pytorch-cifar100): VGG-11, VGG-11 BN, VGG-13, VGG-13 BN, VGG-16, VGG-16 BN, VGG-19, VGG-19 BN, ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, SqueezeNet1-0, DenseNet-121, DenseNet-161, DenseNet-169, DenseNet-201, GoogleNet. These models work with CIFAR-10 and CIFAR-100 and support pretrained parameters for both datasets.


### Hyperparameters
#### CIFAR-10/100
Below are the hyperparameters used to train models on CIFAR-10/100.
| Model       | Test Accuracy | Optimizer | Epochs | Batch Size | Learning Rate | Learning Rate Drops | Drop Factor | Weight Decay |
|-------------|---------------|-----------|--------|------------|---------------|---------------------|-------------|--------------|
| VGG-16 BN (Lottery)     |         | momentum  | 160    | 128        | 0.1           | (60 120)        | 0.1         | 1e-4         |
| VGG-19 BN (Lottery)     |         | momentum  | 160    | 128        | 0.1           | (60 120)        | 0.1         | 1e-4         |
| ResNet-18 (Tiny)     |         | momentum  | 160    | 128        | 0.01          | (60 120)        | 0.2         | 5e-4         |
| WideResNet-18 (Tiny) |         | momentum  | 160    | 128        | 0.01          | (60 120)        | 0.2         | 5e-4         |

#### Tiny ImageNet
Below are the hyperparameters used to train models on TinyImageNet.
| Model       | Test Accuracy | Optimizer | Epochs | Batch Size | Learning Rate | Learning Rate Drops | Drop Factor | Weight Decay |
|-------------|---------------|-----------|--------|------------|---------------|---------------------|-------------|--------------|
| VGG-16 BN (Lottery)     |         | momentum  | 100    | 128        | 0.01          | (30 60 80) | 0.1         | 1e-4         |
| VGG-19 BN (Lottery)     |         | momentum  | 100    | 128        | 0.01          | (30 60 80) | 0.1         | 1e-4         |
| ResNet-18 (Tiny)     |         | momentum  | 100    | 128        | 0.01          | (30 60 80) | 0.1         | 1e-4         |
| WideResNet-18 (Tiny) |         | momentum  | 100    | 128        | 0.01          | (30 60 80) | 0.1         | 1e-4         |


####  ImageNet Models

This code base supports the following models for ImageNet based on [Torchvision models](https://pytorch.org/docs/stable/torchvision/models.html): VGG-11, VGG-11 BN, VGG-13, VGG-13 BN, VGG-16, VGG-16 BN, VGG-19, VGG-19 BN, ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNext-50, ResNext-101, Wide-ResNet-50, Wide-ResNet-101, SqueezeNet1-0, SqueezeNet1-1, DenseNet-121, DenseNet-161, DenseNet-169, DenseNet-201, GoogleNet. These models work with ImageNet and support pretrained weights on ImageNet from [Torchvision](https://pytorch.org/docs/stable/torchvision/models.html).


### Layers

### Pruners

### Experiments

## Citation
If you use this code for your research, please cite our paper,
["Pruning neural networks without any data by iteratively conserving synaptic flow"]()

## References
Parts of this code are based on 
- [PyTorch](https://github.com/pytorch) (... model class)
- [OpenLTH](https://github.com/facebookresearch/open_lth) (lottery model class)
- [weiaicunzai](https://github.com/weiaicunzai/pytorch-cifar100) (Tiny ImageNet model class)
<!---[ShrinkBench](https://shrinkbench.github.io/)--->
