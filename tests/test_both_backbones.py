import torch
import warnings
warnings.filterwarnings('ignore')

from models.model_factory import build_model

print("=== Testing ResNet Backbone (swsl_resnet18) ===")
config_resnet = {
    'model': {
        'name': 'unetformer',
        'backbone': 'swsl_resnet18',
        'pretrained': False,
        'decode_channels': 64,
        'window_size': 8,
        'dropout': 0.1,
    },
    'data': {'num_classes': 8}
}
model_resnet = build_model(config_resnet).eval()
x = torch.randn(1, 3, 256, 256)
out = model_resnet(x)
print(f"✅ ResNet model loaded and tested successfully! Output shape: {out.shape}\n")

print("=== Testing SwinV2 Backbone ===")
config_swin = {
    'model': {
        'name': 'unetformer',
        'backbone': 'swinv2_base_window8_256.ms_in1k',
        'pretrained': False,
        'decode_channels': 64,
        'window_size': 8,
        'dropout': 0.1,
    },
    'data': {'num_classes': 8}
}
model_swin = build_model(config_swin).eval()
out_swin = model_swin(x)
print(f"✅ SwinV2 model loaded and tested successfully! Output shape: {out_swin.shape}\n")

print("=== Testing Swin Base Backbone ===")
config_swin_base = {
    'model': {
        'name': 'unetformer',
        'backbone': 'swin_base_patch4_window7_224.ms_in1k',
        'pretrained': False,
        'decode_channels': 64,
        'window_size': 8,
        'dropout': 0.1,
    },
    'data': {'num_classes': 8}
}
model_swin_base = build_model(config_swin_base).eval()
out_swin_base = model_swin_base(x)
print(f"✅ Swin Base model loaded and tested successfully! Output shape: {out_swin_base.shape}")

print("\n🎉 All backbones work perfectly!")
