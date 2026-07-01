"""Test script to verify NaN fix for missing classes."""
import torch
import numpy as np
from metrics.segmentation_metrics import compute_metrics, mean_ignore_nan
from losses.losses import DiceLoss, SegmentationLoss

# Test 1: Metrics with missing class (Bareland scenario)
print("=" * 60)
print("Test 1: Metrics with missing class (Bareland)")
print("=" * 60)

# Simulate predictions and targets where Bareland (class 7) is missing
preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])  # No class 7
targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])  # No class 7

metrics = compute_metrics(preds, targets, num_classes=8)
print(f"OA: {metrics['oa']:.4f}")
print(f"mIoU: {metrics['miou']:.4f}")
print(f"mF1: {metrics['mf1']:.4f}")
print("\nPer-class IoU:")
for i, iou in enumerate(metrics['per_class_iou']):
    print(f"  Class {i}: {iou:.4f}")

# Verify no NaN
assert not any(np.isnan(v) for v in metrics['per_class_iou']), "NaN found in per_class_iou!"
assert not any(np.isnan(v) for v in metrics['per_class_f1']), "NaN found in per_class_f1!"
print("\n✓ No NaN values in metrics!")

# Test 2: DiceLoss with missing class
print("\n" + "=" * 60)
print("Test 2: DiceLoss numerical stability")
print("=" * 60)

dice_loss = DiceLoss(num_classes=8)
logits = torch.randn(4, 8, requires_grad=True)
targets = torch.tensor([0, 1, 2, 6])  # No class 7

loss = dice_loss(logits, targets)
print(f"DiceLoss: {loss.item():.4f}")
assert not torch.isnan(loss), "DiceLoss returned NaN!"
assert torch.isfinite(loss), "DiceLoss returned infinite value!"
print("✓ DiceLoss is numerically stable!")

# Test 3: SegmentationLoss with missing class
print("\n" + "=" * 60)
print("Test 3: SegmentationLoss with missing class")
print("=" * 60)

seg_loss = SegmentationLoss(num_classes=8)
logits = torch.randn(4, 8, requires_grad=True)
targets = torch.tensor([0, 1, 2, 6])  # No class 7

loss = seg_loss(logits, targets)
print(f"SegmentationLoss: {loss.item():.4f}")
assert not torch.isnan(loss), "SegmentationLoss returned NaN!"
assert torch.isfinite(loss), "SegmentationLoss returned infinite value!"
print("✓ SegmentationLoss is numerically stable!")

# Test 4: Gradient computation
print("\n" + "=" * 60)
print("Test 4: Gradient computation")
print("=" * 60)

logits = torch.randn(4, 8, requires_grad=True)
targets = torch.tensor([0, 1, 2, 6])
loss = seg_loss(logits, targets)
loss.backward()

grad_norm = torch.nn.utils.clip_grad_norm_(logits, max_norm=1.0)
print(f"Gradient norm: {grad_norm:.4f}")
assert not torch.isnan(grad_norm), "Gradient norm is NaN!"
print("✓ Gradients computed successfully!")

print("\n" + "=" * 60)
print("All tests passed! NaN issues are fixed.")
print("=" * 60)