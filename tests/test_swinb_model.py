import torch
import warnings
warnings.filterwarnings('ignore')

from models.model_factory import build_model

config = {
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
model = build_model(config)
x = torch.randn(1, 3, 256, 256)
out = model(x)
print(f'✅ Swin-B model loaded and tested successfully! Output shape: {out.shape}')
print(f'Backbone used: {config["model"]["backbone"]}')
