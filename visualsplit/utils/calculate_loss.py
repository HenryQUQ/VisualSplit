"""Loss functions used throughout the project."""

from __future__ import annotations

import torch


def iou_loss_one_dimension(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the IoU loss for 1‑D histograms."""

    intersection = (pred * target).sum(dim=-1)
    union = (pred + target - pred * target).sum(dim=-1)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return 1 - iou.mean()


def chi_square_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the chi‑square distance between two histograms."""

    return torch.mean(torch.pow(pred - target, 2) / (pred + target + 1e-6))


__all__ = ["iou_loss_one_dimension", "chi_square_loss"]
