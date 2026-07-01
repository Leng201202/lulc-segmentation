from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from datasets.irsamap_dataset import IRSAMapDataset
from datasets.loveda_dataset import LoveDADataset
from utils.config import get_project_root, resolve_path


def build_irsamap_dataset(config: dict, split: str, is_train: bool) -> IRSAMapDataset:
    project_root = get_project_root()
    data_cfg = config["data"]
    irsamap_cfg = data_cfg["irsamap"]
    root = resolve_path(project_root, irsamap_cfg["root"])

    if split == "train":
        image_key, mask_key = "train_images", "train_masks"
    elif split == "val":
        image_key, mask_key = "val_images", "val_masks"
    else:
        image_key, mask_key = "test_images", "test_masks"

    return IRSAMapDataset(
        image_dir=resolve_path(root, irsamap_cfg[image_key]),
        mask_dir=resolve_path(root, irsamap_cfg[mask_key]),
        image_size=data_cfg.get("image_size", 512),
        augment_cfg=config.get("augment"),
        is_train=is_train,
    )


def build_train_dataset(config: dict):
    irsamap_train = build_irsamap_dataset(config, split="train", is_train=True)
    loveda_cfg = config["data"].get("loveda", {})

    if not loveda_cfg.get("enabled", False):
        return irsamap_train, None

    project_root = get_project_root()
    root = resolve_path(project_root, loveda_cfg["root"])
    loveda_train = LoveDADataset(
        image_dir=resolve_path(root, loveda_cfg["train_images"]),
        mask_dir=resolve_path(root, loveda_cfg["train_masks"]),
        image_size=config["data"].get("image_size", 512),
        augment_cfg=config.get("augment"),
        is_train=True,
        mapping_name=loveda_cfg.get("mapping", "targeted"),
    )
    return irsamap_train, loveda_train


def build_train_loader(config: dict) -> DataLoader:
    train_cfg = config["train"]
    irsamap_train, loveda_train = build_train_dataset(config)

    if loveda_train is None:
        dataset = irsamap_train
        sampler = None
        shuffle = True
    else:
        loveda_cfg = config["data"]["loveda"]
        irsamap_weight = loveda_cfg.get("irsamap_weight", 0.8)
        loveda_weight = loveda_cfg.get("loveda_weight", 0.2)

        dataset = ConcatDataset([irsamap_train, loveda_train])
        weights = (
            [irsamap_weight / len(irsamap_train)] * len(irsamap_train)
            + [loveda_weight / len(loveda_train)] * len(loveda_train)
        )
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(irsamap_train),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("device", "cuda") == "cuda",
        drop_last=len(dataset) >= train_cfg["batch_size"],
    )


def build_val_loader(config: dict) -> DataLoader:
    train_cfg = config["train"]
    val_dataset = build_irsamap_dataset(config, split="val", is_train=False)
    return DataLoader(
        val_dataset,
        batch_size=train_cfg.get("val_batch_size", train_cfg["batch_size"]),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("device", "cuda") == "cuda",
    )


def build_test_loader(config: dict) -> DataLoader:
    train_cfg = config["train"]
    test_dataset = build_irsamap_dataset(config, split="test", is_train=False)
    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("device", "cuda") == "cuda",
    )
