import timm
import torch

x = torch.randn(1, 3, 256, 256)

print("=== ResNet backbone (swsl_resnet18) ===")
backbone_resnet = timm.create_model("swsl_resnet18", features_only=True, out_indices=(1,2,3,4), pretrained=False)
outs_resnet = backbone_resnet(x)
for i, out in enumerate(outs_resnet):
    print(f"  Output {i+1}: shape {out.shape}")

print("\n=== SwinV2 backbone ===")
backbone_swin = timm.create_model("swinv2_base_window8_256.ms_in1k", features_only=True, out_indices=(0,1,2,3), pretrained=False)
outs_swin = backbone_swin(x)
for i, out in enumerate(outs_swin):
    print(f"  Output {i+1}: shape {out.shape}")
