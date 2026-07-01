# LULC Segmentation with UNetFormer

# Dataset
IRSAMap Dataset

# Model
UNetFormer with ResNet-18 backbone

# Training
python train.py

# Validation
python validate.py

# Testing
python test.py


# Basic inference on test set
python test.py -o outputs/test_predictions

# With TTA (recommended for best results)
python test.py -o outputs/test_predictions -t d4

# Validate split
python test.py -o outputs/val_predictions -s val

# RGB colored outputs
python test.py -o outputs/vis --rgb

# Custom checkpoint
python test.py -ckpt checkpoints/epoch_035.pt -o outputs/test