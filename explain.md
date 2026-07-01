# Project Analysis & Resolutions: lulc-segmentation

> **Context**: Mac is used for code preparation only. Training runs on a Windows machine. All dataset images are **1024 × 1024 px**.

---

## 1. Project Overview

```
lulc-segmentation/
├── configs/          # YAML experiment configs (3 stages)
├── data/
│   ├── IRSAMap/      # train/ val/ test/ → images/ + masks_rgb/
│   └── LoveDA/       # train/ val/ test/ → images/ + masks/
├── datasets/         # PyTorch Dataset classes + label maps + transforms
├── losses/           # CE + Dice combined loss
├── metrics/          # IoU / F1 / OA computation
├── models/           # UNetFormer (timm backbone + custom decoder)
├── tools/            # Preprocessing & validation scripts (ADDED)
├── utils/            # config loader, seed, visualization
├── train.py
├── evaluate.py
└── predict.py
```

**Training pipeline (3 stages):**
1. `irsamap_unetformer_resnet18_8class.yaml` — Train on IRSAMap only (baseline)
2. `irsamap_loveda_support_unetformer_resnet18_8class.yaml` — Train with LoveDA as support data (weighted sampler)
3. `irsamap_finetune_from_support.yaml` — Fine-tune Stage 2 checkpoint on IRSAMap only

---

## 2. Issues Found and Resolved

### ✅ Issue 1 — `image_size` is 512 but data is 1024 px (RESOLVED)

**File:** All three YAML configs under `configs/`

- **Problem:** `image_size: 512` was causing `A.Resize()` to downscale images to half their resolution, throwing away 75% of spatial resolution.
- **Resolution:** Updated `image_size: 1024` in `configs/irsamap_unetformer_resnet18_8class.yaml`, `configs/irsamap_loveda_support_unetformer_resnet18_8class.yaml`, and `configs/irsamap_finetune_from_support.yaml`.

---

### ✅ Issue 2 — Bug in `SeparableConvBNReLU` and `SeparableConvBN` (RESOLVED)

**File:** `models/unetformer.py`

- **Problem:** In both separable convolutions, `norm_layer(out_channels)` was called directly after a depthwise convolution producing `in_channels` feature channels, leading to a channel mismatch/crash during training.
- **Resolution:** Fixed to `norm_layer(in_channels)` in both classes, matching the output dimensions of the depthwise convolution.

---

### ✅ Issue 3 — Raw dataset folder structure does NOT match config paths (RESOLVED)

**File:** `tools/prepare_irsamap.py`, `tools/prepare_loveda.py`

- **Problem:** Configs expect plural folders (`images/`, `masks_rgb/`) and merged splits, while raw Windows datasets use singular folders (`image/`) and separate region folders (`Urban`/`Rural`).
- **Resolution:** Created data preparation tools in the `tools/` folder to automate restructuring and merging on the Windows training machine.

---

### ✅ Issue 4 — `tools/` directory is empty (RESOLVED)

- **Resolution:** Added three new helper scripts:
  - `tools/prepare_irsamap.py` to copy images and map selected RGB masks (e.g. `SegLabel_rvwsb`) into the expected paths.
  - `tools/prepare_loveda.py` to merge `Urban` and `Rural` splits into flat directories.
  - `tools/verify_dataset.py` to sanity-check matching count and size compatibility.

---

### ✅ Issue 5 — Code duplication and slow metrics computation (RESOLVED)

**File:** `train.py`, `evaluate.py`

- **Problem:** Call to `compute_metrics()` inside batch loops resulted in expensive tensor-to-numpy and CPU looping for every batch.
- **Resolution:** Refactored loops to accumulate PyTorch tensor variables directly (`total_correct` and `total_pixels`). `compute_metrics()` is now only run once globally after batch processing concludes.

---

### ✅ Issue 6 — `train.py` saves last checkpoint with potentially undefined `val_metrics` (RESOLVED)

**File:** `train.py`

- **Problem:** If training started but finished immediately (e.g. resume epoch exceeding total epochs), `val_metrics` was undefined at checkpointing.
- **Resolution:** Initialized `val_metrics = {}` safely before the epoch loop.

---

### ✅ Issue 7 — `num_workers` on Windows (RESOLVED)

- **Resolution:** Ensured that `train.py`, `evaluate.py`, and `predict.py` wrap entry points inside `if __name__ == "__main__":` to prevent Windows multiprocessing pipe errors. If issues persist, users can configure `num_workers: 0` in configs.

---

## 3. Summary of Fixes Applied

| # | Severity | File | Fix Applied | Status |
|---|---|---|---|---|
| 1 | CRITICAL | `configs/*.yaml` (all 3) | `image_size: 512` → `image_size: 1024` | Done ✅ |
| 2 | CRITICAL | `models/unetformer.py` | `norm_layer(out_channels)` → `norm_layer(in_channels)` | Done ✅ |
| 3 | HIGH | `tools/prepare_irsamap.py` | Created raw IRSAMap restructurer script | Done ✅ |
| 4 | HIGH | `tools/prepare_loveda.py` | Created Rural+Urban merger script | Done ✅ |
| 5 | MEDIUM | `train.py`, `evaluate.py` | Removed per-batch numpy/CPU loop overhead | Done ✅ |
| 6 | MEDIUM | `train.py` | Initialized `val_metrics = {}` before training loop | Done ✅ |
| 7 | LOW | `tools/verify_dataset.py` | Created image size & mapping validation script | Done ✅ |

---

## 4. Recommended Workflow (Windows Training Machine)

```bash
# Step 0 — Install dependencies
pip install -r requirements.txt

# Step 1 — Prepare dataset structures from raw files
python tools/prepare_irsamap.py --src data/IRSAMap_raw --dst data/IRSAMap
python tools/prepare_loveda.py  --src data/LoveDA_raw  --dst data/LoveDA

# Step 2 — Verify dataset counts and dimensions
python tools/verify_dataset.py

# Step 3 — Run the entire 3-stage training, evaluation, and prediction pipeline automatically
bash run_pipeline.sh
```

---

## 5. Pipeline Alerting (Email Notification)

The `run_pipeline.sh` script is configured with SMTP email alerting capability:
* **Configuration**: Set your credentials (`EMAIL_TO`, `EMAIL_FROM`, `SMTP_SERVER`, etc.) at the top of [run_pipeline.sh](file:///Users/leng/Documents/lulc-segmentation/run_pipeline.sh).
* **Behavior**: If training, evaluation, or prediction fails during any of the stages, the script immediately sends an email notification containing the failed step details and the last 50 lines of log output before terminating.

