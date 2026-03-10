import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Layers import layers as l

class VGG16_CIFAR10(nn.Module):
    def __init__(self, input_c = 3, num_classes = 10):
        super().__init__()

        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                              std=(0.2023, 0.1994, 0.2010))
        self.activation = nn.ReLU(inplace=True)
        # Helper for conv blocks (used for first 10 conv layers)
        def conv_block(in_c, out_c, num_convs):
            layers = []
            for _ in range(num_convs):
                layers += [
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                ]
                in_c = out_c
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return layers

        # First 10 convolutional layers using conv_block (up to conv4_2)
        self.features = nn.Sequential(
            *conv_block(input_c, 64, 2),     # conv1_1, conv1_2 16
            *conv_block(64, 128, 2),         # conv2_1, conv2_2 8
            *conv_block(128, 256, 3),        # conv3_1, conv3_2, conv3_3 4
        )
        
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 4
        self.bn4_1 = nn.BatchNorm2d(256)

        # Last 3 conv layers (conv4_3, conv5_1, conv5_2) defined explicitly
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 4
        self.bn4_2 = nn.BatchNorm2d(256)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)  # 4
        self.bn4_3 = nn.BatchNorm2d(256)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4 x 4 -> 2 x 2

        # Last 3 conv layers (conv4_3, conv5_1, conv5_2) defined explicitly
        self.conv5_1 = l.Conv2d(256, 256, kernel_size=3, padding=1) # 2
        self.bn5_1 = nn.BatchNorm2d(256)

        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1) # 2
        self.bn5_2 = nn.BatchNorm2d(256)
        
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4 x 4 -> 2 x 2

        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=1, padding=0, stride = 1) # 2
        self.bn5_3 = nn.BatchNorm2d(256)

        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 → 1x1

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = l.Linear(256, num_classes)


    def forward(self, x):
        # x = self.normalize(x)
        x = self.features(x)
        
        x = self.activation(self.bn4_1(self.conv4_1(x)))
        x = self.activation(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        
        x = self.activation(self.bn4_3(self.conv4_3(x)))
        
        x = self.activation(self.bn5_1(self.conv5_1(x)))
        x = self.activation(self.bn5_2(self.conv5_2(x)))
        x = self.pool5(x)
        
        x = self.activation(self.bn5_3(self.conv5_3(x)))
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        x = self.fc3(x)
        return x
    

def _vgg(arch, num_classes, dense_classifier, pretrained):
    model = VGG16_CIFAR10(num_classes=num_classes)
    if pretrained:
        if num_classes == 10:
            pretrained_path = 'Models/models/new/' + 'vgg16_10_ori_relu_pool4.pth'
        elif num_classes == 100:
            pretrained_path = 'Models/models/new/' + 'vgg16_100_ori_relu.pth'
            
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
    
def VGG(input_shape, num_classes, dense_classifier=False, pretrained=False):
    return _vgg('relu', num_classes, dense_classifier, pretrained)