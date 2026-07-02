import torch
import torchvision

print("=== Torchvision DeepLabV3 ResNet101 backbone test ===")
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None, weights_backbone=None)
x = torch.randn(1, 3, 256, 256)
outputs = model.backbone(x)
print("Backbone output keys:", list(outputs.keys()))

for key, val in outputs.items():
    print(f"  Key '{key}': shape {val.shape}")
