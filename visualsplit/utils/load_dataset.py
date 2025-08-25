"""Dataset loading helper."""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader


def load_dataset(
    dataset_name: str, batch_size: int = 128, image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load the specified dataset and return dataloaders."""

    if dataset_name == "ImageNet-1k-pure":
        from ..dataset_loader.ImageNet1k.ImageNet_1K_dataloader import (
            load_imagenet_1k_pure_dataloader,
        )

        return load_imagenet_1k_pure_dataloader(
            batch_size=batch_size, image_size=image_size
        )

    raise ValueError(f"Dataset {dataset_name!r} not found")


__all__ = ["load_dataset"]
