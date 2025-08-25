"""Model persistence helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_model(model: nn.Module, path: Path | str) -> None:
    """Save the given model's state dictionary to ``path``."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(path))


__all__ = ["save_model"]
