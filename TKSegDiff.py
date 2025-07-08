import torch
import torch.nn as nn
import torch.nn.functional as F

class TKS2EA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(TKS2EA, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = in_channels // reduction_ratio
        
        # Search operation components
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        
        # Compression operation components
        self.soft_pool = nn.AdaptiveAvgPool2d(1)  # Using adaptive avg pool as approximation
        
        # Excitation operation components
        self.fc1 = nn.Linear(in_channels, self.reduced_channels)
        self.fc2 = nn.Linear(self.reduced_channels, in_channels)
        self.leaky_relu = nn.LeakyReLU()
        self.hard_sigmoid = nn.Hardsigmoid()
        
        # Adjustment operation components
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Search operation
        U1 = self.conv3x3(x)  # Main figure
        U2 = self.conv5x5(x)  # Related areas
        U3 = self.conv7x7(x)  # Background areas
        U = U1 + U2 + U3  # Element-wise summation
        
        # Compression operation
        batch_size, channels, height, width = U.size()
        
        # Soft pooling approximation
        y = self.soft_pool(U).view(batch_size, channels)
        
        # Excitation operation
        s = self.fc1(y)
        s = self.leaky_relu(s)
        s = self.fc2(s)
        s = self.hard_sigmoid(s).view(batch_size, channels, 1, 1)
        
        # Adjustment operation
        # Apply channel-wise attention
        X_prime = U * s
        
        return X_prime

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        
        # Spatial feature transformation block
        self.spatial_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Two 3x3 convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Decision head
        self.decision_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        # Spatial feature transformation
        transformed = self.spatial_transform(features)
        
        # Convolution layers
        conv_out = self.conv_layers(transformed)
        
        # Decision head
        output = self.decision_head(conv_out)
        
        return output

# Example usage with ResNet-152
class ResNetWithTKS2EA(nn.Module):
    def __init__(self, original_resnet):
        super(ResNetWithTKS2EA, self).__init__()
        self.resnet = original_resnet
        self.tks2ea = TKS2EA(2048)  # Assuming ResNet-152 final channels
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Apply TKS2EA before final layers
        x = self.tks2ea(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x