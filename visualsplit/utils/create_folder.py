"""Utility helpers for creating experiment folders."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Tuple

from ..config import Config


def create_folder(
    params: Dict[str, str],
    root_folder: Path = Config.TRAINING_FOLDER,
    start_time: datetime.datetime | None = None,
) -> Tuple[Path, Path, Path]:
    """Create a timestamped folder structure for training runs."""

    start_time = start_time or datetime.datetime.now()
    name = f"{params['name']}_{params['dataset']}_{start_time.strftime('%Y-%m-%dT%H-%M-%S')}"
    folder = Path(root_folder) / name
    model_folder = folder / "model_weights"
    image_folder = folder / "images"

    for path in (model_folder, image_folder):
        path.mkdir(parents=True, exist_ok=True)

    print(f"training data is saved in {folder}")
    return folder, model_folder, image_folder


__all__ = ["create_folder"]
