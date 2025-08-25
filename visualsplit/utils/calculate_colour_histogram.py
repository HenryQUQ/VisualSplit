"""Differentiable histogram implementations."""

from __future__ import annotations

import torch
import torch.nn as nn


class GrayLevelHistogram(nn.Module):
    """Compute a smooth gray level histogram for grayscale images."""

    def __init__(
        self, bins: tuple[int, int, int] = (0, 1, 100), sigma: float = 0.05
    ) -> None:
        super().__init__()
        self.bins = torch.linspace(bins[0], bins[1], bins[2])
        self.sigma = sigma

    def forward(
        self, image: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - simple wrapper
        self.bins = self.bins.to(image.device)
        return self.smooth_histogram(image, bins=self.bins, sigma=self.sigma)

    def smooth_histogram(
        self, x: torch.Tensor, bins: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """Compute a smooth histogram using a Gaussian kernel."""

        n, c, h, w = x.shape
        num_bins = len(bins)
        x = x.view(n, c, -1, 1)
        bins = bins.view(1, 1, 1, num_bins)
        diff = x - bins
        kernel = torch.exp(-0.5 * (diff / sigma) ** 2)
        return kernel.sum(dim=2)


class ColourHistogram(nn.Module):
    """Compute a smooth 2D colour histogram for ``ab`` channels."""

    def __init__(self, bins_per_channel: int = 32, sigma: float = 0.05) -> None:
        super().__init__()
        self.bins_per_channel = bins_per_channel
        self.sigma = sigma
        self.bins_a = torch.linspace(0, 1, bins_per_channel)
        self.bins_b = torch.linspace(0, 1, bins_per_channel)

    def forward(
        self, image: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - simple wrapper
        self.bins_a = self.bins_a.to(image.device)
        self.bins_b = self.bins_b.to(image.device)
        return self.smooth_histogram(image, self.bins_a, self.bins_b, self.sigma)

    def smooth_histogram(
        self, x: torch.Tensor, bins_a: torch.Tensor, bins_b: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """Compute a smooth joint histogram for two channels."""

        n, c, h, w = x.shape
        if c != 2:
            raise ValueError("Input image must have 2 channels")

        channel_a, channel_b = x[:, 0, :, :], x[:, 1, :, :]

        bins_a = bins_a.view(1, 1, 1, -1)
        bins_b = bins_b.view(1, 1, 1, -1)

        diff_a = channel_a.view(n, 1, h * w, 1) - bins_a
        diff_b = channel_b.view(n, 1, h * w, 1) - bins_b

        kernel_a = torch.exp(-0.5 * (diff_a / sigma) ** 2)
        kernel_b = torch.exp(-0.5 * (diff_b / sigma) ** 2)

        return torch.matmul(kernel_a.transpose(2, 3), kernel_b)


__all__ = ["GrayLevelHistogram", "ColourHistogram"]
