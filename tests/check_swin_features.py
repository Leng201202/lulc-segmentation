import timm
import torch

# Test swinv2
print("=== Testing SwinV2 ===")
model_name = "swinv2_base_window8_256.ms_in1k"
backbone = timm.create_model(model_name, features_only=True, pretrained=False)
print("Feature info:")
print("  Feature channels:", backbone.feature_info.channels())
print("  Feature reductions:", backbone.feature_info.reduction())

# Test swin_base
print("\n=== Testing Swin Base ===")
model_name2 = "swin_base_patch4_window7_224.ms_in1k"
backbone2 = timm.create_model(model_name2, features_only=True, pretrained=False)
print("Feature info:")
print("  Feature channels:", backbone2.feature_info.channels())
print("  Feature reductions:", backbone2.feature_info.reduction())

print("\n✅ Feature info checked successfully!")
