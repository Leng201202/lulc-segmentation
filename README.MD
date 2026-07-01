Viewed unetformer.py:289-308

Here is a complete, structural walkthrough of the **`lulc-segmentation`** codebase. 

---

## 1. Directory Structure & Architecture
The repository implements a **3-stage Land Use/Land Cover (LULC) semantic segmentation pipeline** (Baseline → Support/Joint Training → Fine-Tuning) using a custom **UNetFormer** neural network.

```
lulc-segmentation/
├── configs/             # YAML configurations for the 3 stages
├── data/                # Dataset directory structure (IRSAMap & LoveDA)
├── datasets/            # Dataset parsers, mappings, and augmentation
├── losses/              # Combined Cross-Entropy & Dice segmentation loss
├── metrics/             # Performance evaluation metrics
├── models/              # Model definitions (UNetFormer & Factory)
├── tools/               # Dataset preparation & validation scripts
├── utils/               # Configuration loaders, seeds, and visualizations
├── train.py             # Model training entrypoint
├── evaluate.py          # Model validation/test evaluation script
└── predict.py           # Visualization & inference script
```

---

## 2. Dataset Pipeline (`datasets/` & `tools/`)
The pipeline processes two datasets: **IRSAMap** (primary dataset) and **LoveDA** (auxiliary/support dataset).

### A. Data Preprocessing (`tools/`)
*   [tools/prepare_irsamap.py](file:///Users/leng/Documents/lulc-segmentation/tools/prepare_irsamap.py): Takes raw folders containing singular `image/` directories and colored `SegLabel_*/` subfolders and restructures them into plural `images/` and `masks_rgb/` splits.
*   [tools/prepare_loveda.py](file:///Users/leng/Documents/lulc-segmentation/tools/prepare_loveda.py): Merges LoveDA's disjointed `Urban` and `Rural` subfolders into a single unified directory layout.
*   [tools/verify_dataset.py](file:///Users/leng/Documents/lulc-segmentation/tools/verify_dataset.py): Validates file matching between images and masks and samples image dimensions to verify they are **1024x1024 px**.

### B. Mappings & Dataset Classes (`datasets/`)
*   [datasets/label_maps.py](file:///Users/leng/Documents/lulc-segmentation/datasets/label_maps.py): 
    *   Defines **8 LULC classes** (Farmland, Tree, Grass, Water, Building, Road, Sport, Bareland).
    *   `IRSAMAP_RGB_TO_CLASS` maps RGB colored masks in IRSAMap to index labels (0-7).
    *   `LOVEDA_TO_IRSAMAP_SUPPORT` maps LoveDA classes into IRSAMap classes, ignoring labels like *Background* or *Low Vegetation* by assigning them to `IGNORE_INDEX` (255).
*   [datasets/irsamap_dataset.py](file:///Users/leng/Documents/lulc-segmentation/datasets/irsamap_dataset.py): Loads IRSAMap. Reads BGR images, converts them to RGB, and parses RGB color masks into integer label maps.
*   [datasets/loveda_dataset.py](file:///Users/leng/Documents/lulc-segmentation/datasets/loveda_dataset.py): Loads LoveDA. It reads grayscale index masks and remaps them to align with IRSAMap class distributions.
*   [datasets/combined_dataset.py](file:///Users/leng/Documents/lulc-segmentation/datasets/combined_dataset.py):
    *   Creates dataloaders for training, validation, and testing.
    *   If LoveDA is enabled as a support dataset, it joins both datasets using `ConcatDataset` and balances the sampling ratio (typically 80% IRSAMap / 20% LoveDA) using `WeightedRandomSampler`.
*   [datasets/transforms.py](file:///Users/leng/Documents/lulc-segmentation/datasets/transforms.py): Applies **Albumentations** transformations (Horizontal/Vertical flips, random 90-degree rotations, resizing, standard ImageNet normalization, and conversion to PyTorch tensors).

---

## 3. Model Architecture (`models/`)
The network is based on **UNetFormer**, a model tailored for remote sensing imagery segmentation.

*   [models/model_factory.py](file:///Users/leng/Documents/lulc-segmentation/models/model_factory.py): Serves as the builder initialization point, loading configurations and returning the model instance.
*   [models/unetformer.py](file:///Users/leng/Documents/lulc-segmentation/models/unetformer.py):
    *   **Backbone**: A ResNet18 model loaded via `timm` (usually `swsl_resnet18` for semi-weakly supervised pre-trained weights) to extract multi-scale encoder features.
    *   **Global-Local Attention (GLA) Block**: Uses window attention mechanisms (similar to Swin Transformer) to model long-range context while retaining local convolutional details.
    *   **WF (Feature Fusion)**: Progressively fuses low-level spatial resolution layers from the encoder with high-level semantic layers from the decoder using learnable weights.
    *   **AuxHead (Auxiliary Head)**: Computes a secondary segmentation loss on intermediate upscaled layers during training to alleviate the vanishing gradient problem.

---

## 4. Losses & Metrics (`losses/` & `metrics/`)
*   [losses/losses.py](file:///Users/leng/Documents/lulc-segmentation/losses/losses.py):
    *   Defines `DiceLoss` which mitigates problems with highly imbalanced class distributions.
    *   `SegmentationLoss` sums standard **Cross-Entropy Loss** and **Dice Loss**. In training mode, it adds the auxiliary loss scaled by `aux_loss_weight` (typically 0.4).
*   [metrics/segmentation_metrics.py](file:///Users/leng/Documents/lulc-segmentation/metrics/segmentation_metrics.py): Calculates **Overall Accuracy (OA)**, **per-class Intersection-over-Union (IoU)**, and **F1-score**, ignoring pixels marked as `IGNORE_INDEX` (255).

---

## 5. Main Runners (`train.py`, `evaluate.py`, `predict.py`)
*   [train.py](file:///Users/leng/Documents/lulc-segmentation/train.py): Runs the epoch loops. It logs loss details to a CSV file and automatically saves the best-performing weights (`best_*.pth`) based on the highest validation Mean IoU (mIoU). It handles optimizer checkpointing and resuming for stage-to-stage learning.
*   [evaluate.py](file:///Users/leng/Documents/lulc-segmentation/evaluate.py): Runs inference on the test split, printing the global mIoU metrics and saving outputs as a structured JSON summary.
*   [predict.py](file:///Users/leng/Documents/lulc-segmentation/predict.py): Computes forward passes for test images, overlays the predicted class maps into RGB color visualizations, and saves side-by-side comparisons (Original Image, Ground Truth, Predicted Mask).

---

## 6. Config System (`configs/`)
Training settings are loaded from YAML files to execute the 3 stages:
1.  **Stage 1** ([configs/irsamap_unetformer_resnet18_8class.yaml](file:///Users/leng/Documents/lulc-segmentation/configs/irsamap_unetformer_resnet18_8class.yaml)): Trains only on IRSAMap (`loveda.enabled: false`) for 100 epochs.
2.  **Stage 2** ([configs/irsamap_loveda_support_unetformer_resnet18_8class.yaml](file:///Users/leng/Documents/lulc-segmentation/configs/irsamap_loveda_support_unetformer_resnet18_8class.yaml)): Turns on `loveda.enabled: true` and trains a combined model using balanced sampling weights.
3.  **Stage 3** ([configs/irsamap_finetune_from_support.yaml](file:///Users/leng/Documents/lulc-segmentation/configs/irsamap_finetune_from_support.yaml)): Resumes the Stage 2 checkpoint (`resume: checkpoints/...`) and fine-tunes only on IRSAMap (`loveda.enabled: false`) with a reduced learning rate.