"""Project configuration and directory management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


# Base directories ---------------------------------------------------------
# All paths are resolved relative to this file so the package can be relocated
# without breaking the directory structure.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent / "cache"
TRAINING_FOLDER: Path = ROOT_DIR / "logs"
IMAGENET_1K_DATASET: Path = ROOT_DIR / "data" / "imagenet-1k"

# Ensure expected directories exist
for path in (ROOT_DIR, TRAINING_FOLDER, IMAGENET_1K_DATASET):
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    """Centralised configuration used across the project."""

    ROOT_DIR: Path = ROOT_DIR
    WEIGHTS_DIR: Path = ROOT_DIR / "model_weights"
    SETTING_DIR: Path = ROOT_DIR / "settings"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_CACHE_DIR: Path = ROOT_DIR / "data_cache"
    BLIP_CACHE_DIR: Path = ROOT_DIR / "blip_cache"

    # HuggingFace -----------------------------------------------------------------
    CACHE_DIR: Path = ROOT_DIR / "model_weights"
    ACCESS_TOKEN: str = os.getenv("HF_TOKEN", "") # TODO: CHANGE ME

    # Dataset ---------------------------------------------------------------------
    VIDEO_FPS: int = 25
    VIDEO_FRAME_HEIGHT: int = 240
    VIDEO_FRAME_WIDTH: int = 320
    DATASET_IMAGENET_1K_DIR: Path = IMAGENET_1K_DATASET

    # Training settings -----------------------------------------------------------
    TRAINING_FOLDER: Path = TRAINING_FOLDER
    SAVE_MODEL_EPOCH: int = 1
    LOCAL: str = "local"

    # Autoencoder dataloader ------------------------------------------------------
    SAMPLES_PER_VIDEO: int = 3
    VIDEO_CLIP_TIME_STEP: int = 5
    CONSISTENCY_LOSS_WEIGHT: float = 0.0
    IMAGE_SIZE: int = 32


def ensure_directories() -> None:
    """Create all directories declared in :class:`Config`."""

    for path in [
        Config.ROOT_DIR,
        Config.WEIGHTS_DIR,
        Config.SETTING_DIR,
        Config.DATA_CACHE_DIR,
        Config.BLIP_CACHE_DIR,
        Config.DATASET_IMAGENET_1K_DIR,
        Config.TRAINING_FOLDER,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


# The directories are ensured on import so that other modules can rely on them.
ensure_directories()
