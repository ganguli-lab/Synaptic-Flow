import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Layers import layers as l

class VGG16_CIFAR10(nn.Module):
    def __init__(self, input_c = 3, num_classes = 10):
        super().__init__()

        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                              std=(0.2023, 0.1994, 0.2010))
        
        # self.normalize = transforms.Normalize(mean=(0.507, 0.4865, 0.4409), 
        #                                       std=(0.2673, 0.2564, 0.276))

        self.activation = nn.ReLU(inplace=True)
        # self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.conv1_1 = l.Conv2d(input_c, 64, kernel_size=2, padding=0, stride=2)  # 16
        self.bn1_1 = nn.BatchNorm2d(64)

        self.conv2_1 = l.Conv2d(64, 128, kernel_size=3, padding=0, stride = 1)  # 14
        self.bn2_1 = nn.BatchNorm2d(128)
        
        self.conv3_1 = l.Conv2d(128, 128, kernel_size=3, padding=0)  # 12
        self.bn3_1 = nn.BatchNorm2d(128)
        
        self.conv3_2 = l.Conv2d(128, 128, kernel_size=3, padding=0, stride=2)  # 5
        self.bn3_2 = nn.BatchNorm2d(128)
        
        self.conv4_1 = l.Conv2d(128, 256, kernel_size=3, padding=0, stride = 1)  # 3
        self.bn4_1 = nn.BatchNorm2d(256)

        # Last 3 conv layers (conv4_3, conv5_1, conv5_2) defined explicitly
        self.conv4_2 = l.Conv2d(256, 256, kernel_size=3, padding=0, stride = 1)  # 1
        self.bn4_2 = nn.BatchNorm2d(256)
        
        
        # self.conv5_1 = l.Conv2d(256, 256, kernel_size=1, padding=0, stride = 2)  # 1
        # self.bn5_1 = nn.BatchNorm2d(256)s

        # Fully connected layers
        self.fc1 = l.Linear(256 * 1 * 1, 512)
        self.fc2 = l.Linear(512, 128)
        self.fc3 = l.Linear(128, num_classes)

    def forward(self, x):
        # x = self.normalize(x)
        
        x = self.activation(self.bn1_1(self.conv1_1(x)))
        
        # x = self.activation(self.bn1_2(self.conv1_2(x)))
        
        x = self.activation(self.bn2_1(self.conv2_1(x)))
        
        # x = self.activation(self.bn2_2(self.conv2_2(x)))
        
        x = self.activation(self.bn3_1(self.conv3_1(x)))
        
        x = self.activation(self.bn3_2(self.conv3_2(x)))
        
        # x = self.activation(self.bn3_3(self.conv3_3(x)))
        
        x = self.activation(self.bn4_1(self.conv4_1(x)))
        
        x = self.activation(self.bn4_2(self.conv4_2(x)))
        
        # x = self.activation(self.bn4_3(self.conv4_3(x)))
        
        # x = self.activation(self.bn5_1(self.conv5_1(x)))
        # x = self.activation(self.bn5_2(self.conv5_2(x)))
        # x = self.pool5(x)
        
        # x = self.activation(self.bn5_3(self.conv5_3(x)))
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        x = self.fc3(x)
        return x
    
    
def _vgg(arch, num_classes, dense_classifier, pretrained):
    model = VGG16_CIFAR10(num_classes=num_classes)
    if pretrained:
        if num_classes == 10:
            pretrained_path = 'Models/models/new/' + 'vgg9_10_ori_relu_s2.pth'
        elif num_classes == 100:
            pretrained_path = 'Models/models/new/' + 'vgg9_100_wd_relu_s2.pth'
            
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
    
def VGG(input_shape, num_classes, dense_classifier=False, pretrained=False):
    return _vgg('relu', num_classes, dense_classifier, pretrained)
