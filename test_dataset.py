from pathlib import Path
import torch
from datasets.irsamap import build_dataloaders, IRSAMapDataset, list_image_files
from tools.config import load_config, get_data_paths


def main():
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "config/unetformer_resnet18.yml")
    print("✅ Config loaded!")

    # Test data paths
    paths = get_data_paths(config, project_root)
    print("📂 Data paths:")
    for k, v in paths.items():
        print(f"  {k}: {v} {'✅' if v and v.exists() else '❌'}")

    # Test list_image_files
    train_imgs = list_image_files(paths["train_images"])
    print(f"\n🔢 Train images found: {len(train_imgs)}")
    if train_imgs:
        print(f"  First: {train_imgs[0]}")

    # Test dataloaders
    print("\n🔨 Building dataloaders...")
    train_loader, val_loader = build_dataloaders(config, project_root)
    print(f"✅ Train loader ready: {len(train_loader.dataset)} samples")
    if val_loader:
        print(f"✅ Val loader ready: {len(val_loader.dataset)} samples")

    # Test one batch
    print("\n📦 Testing first batch...")
    batch = next(iter(train_loader))
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Mask shape: {batch['mask'].shape}")
    print(f"  Mask unique values: {torch.unique(batch['mask'])}")
    print(f"  Image path: {batch['image_path'][0]}")

    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    main()
