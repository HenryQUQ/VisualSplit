"""Dataloaders for the ImageNet‑1k dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from ...config import Config
from ...utils.extract_features import FeatureExtractor
from ...utils.random_crop_shorter_edge import RandomCropShorterEdge


def _hash_image(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


def _collate_basic(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = [item["label"] for item in batch]
    return images, labels


def _collate_features(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    edge = torch.stack([item["edge"].squeeze(0) for item in batch], dim=0)
    gray = torch.stack([item["gray_level"].squeeze(0) for item in batch], dim=0)
    seg_ab = torch.stack([item["segmented_ab"].squeeze(0) for item in batch], dim=0)
    ab = torch.stack([item["ab"].squeeze(0) for item in batch], dim=0)
    return images, edge, gray, seg_ab, ab


def load_imagenet_1k_dataloader(
    batch_size: int = 128, image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return standard ImageNet‑1k dataloaders."""

    transform = T.Compose(
        [
            RandomCropShorterEdge(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )

    def set_transform(examples):
        examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
        return examples

    train_ds, val_ds, test_ds = load_dataset(
        "imagenet-1k",
        cache_dir=str(Config.DATASET_IMAGENET_1K_DIR),
        token=Config.ACCESS_TOKEN,
        split=["train", "validation", "test"],
    )

    for ds in (train_ds, val_ds, test_ds):
        ds.set_transform(set_transform)

    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=4, collate_fn=_collate_basic
    )
    val_loader = DataLoader(val_ds, 1, num_workers=4, collate_fn=_collate_basic)
    test_loader = DataLoader(test_ds, 1, num_workers=4, collate_fn=_collate_basic)
    return train_loader, val_loader, test_loader


def load_imagenet_1k_pure_dataloader(
    batch_size: int = 128, image_size: int = 224, flag: str = "kmeans_6"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders with additional handcrafted features."""

    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/vit-mae-base", cache_dir=str(Config.CACHE_DIR)
    )
    image_processor.do_normalize = False
    image_processor.size = (image_size, image_size)

    extractor = FeatureExtractor()

    def transform_features(examples):
        hashes = [_hash_image(img) for img in examples["image"]]

        tensors = [
            image_processor(
                images=img.convert("RGB"), return_tensors="pt"
            ).pixel_values[0]
            for img in examples["image"]
        ]

        features = []
        for h, tensor in zip(hashes, tensors):
            cache_path = Path(Config.DATA_CACHE_DIR) / f"{h}_{flag}.pt"
            if cache_path.exists():
                try:
                    feature = torch.load(cache_path)
                except Exception:
                    cache_path.unlink(missing_ok=True)
                    feature = extractor(tensor.unsqueeze(0))
                    torch.save(feature, cache_path)
            else:
                feature = extractor(tensor.unsqueeze(0))
                torch.save(feature, cache_path)
            features.append(feature)

        edge, gray_level, segmented_ab, ab = zip(*features)
        examples["image"] = tensors
        examples["edge"] = edge
        examples["gray_level"] = gray_level
        examples["segmented_ab"] = segmented_ab
        examples["ab"] = ab
        return examples

    train_ds, val_ds, test_ds = load_dataset(
        "imagenet-1k",
        cache_dir=str(Config.DATASET_IMAGENET_1K_DIR),
        token=Config.ACCESS_TOKEN,
        split=["train", "validation", "test"],
    )

    for ds in (train_ds, val_ds, test_ds):
        ds.set_transform(transform_features)

    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=4, collate_fn=_collate_features
    )
    val_loader = DataLoader(val_ds, 1, num_workers=4, collate_fn=_collate_features)
    test_loader = DataLoader(test_ds, 1, num_workers=4, collate_fn=_collate_features)
    return train_loader, val_loader, test_loader


__all__ = [
    "load_imagenet_1k_dataloader",
    "load_imagenet_1k_pure_dataloader",
]
