import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet101, ResNet101_Weights


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone_name="resnet101", pretrained=True, dropout=0.5):
        super().__init__()
        
        # Load backbone (ResNet101)
        if pretrained:
            weights = ResNet101_Weights.IMAGENET1K_V1
            backbone = resnet101(weights=weights)
        else:
            backbone = resnet101(weights=None)
        
        # Split backbone into layers to get low-level features
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # Outputs (N, 256, H/4, W/4)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4  # Outputs (N, 2048, H/32, W/32)
        
        # ASPP module
        self.aspp = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Low-level features projection
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        # Final conv layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Backbone forward pass
        x = self.stem(x)
        x_low = self.layer1(x)  # (N, 256, H/4, W/4)
        x = self.layer2(x_low)
        x = self.layer3(x)
        x_high = self.layer4(x)  # (N, 2048, H/32, W/32)
        
        # Process high level features
        x_high = self.aspp(x_high)
        x_high = nn.functional.interpolate(x_high, size=x_low.shape[-2:], mode='bilinear', align_corners=False)
        
        # Process low level features
        x_low = self.low_level_conv(x_low)
        
        # Concatenate
        x = torch.cat([x_low, x_high], dim=1)
        
        # Final conv
        x = self.final_conv(x)
        x = self.dropout(x)
        x = self.final(x)
        
        # Upsample to input size
        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x
