import torch
import warnings
warnings.filterwarnings('ignore')

from models.model_factory import build_model

print("=== Testing DeepLabV3+ with ResNet-101 ===")
config_deeplab = {
    'model': {
        'name': 'deeplabv3p',
        'backbone': 'resnet101',
        'pretrained': False,
        'dropout': 0.5,
    },
    'data': {'num_classes': 8}
}

model_deeplab = build_model(config_deeplab).eval()
x = torch.randn(1, 3, 256, 256)  # Test input
out = model_deeplab(x)
print(f"✅ DeepLabV3+ model loaded and tested successfully! Output shape: {out.shape}")
