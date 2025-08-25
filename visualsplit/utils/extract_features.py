"""Feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torchvision
from kornia.color import rgb_to_lab
from sklearn.cluster import KMeans

from .calculate_colour_histogram import GrayLevelHistogram
from .edge_detection import SobelOperator


@dataclass
class FeatureExtractor:
    """Extracts hand‑crafted features from RGB tensors.

    The extractor computes edge maps, gray level histograms and colour
    segmentation using *k*-means.
    """

    n_clusters: int = 6
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.edge_detector = SobelOperator().to(self.device)
        self.gray_hist = GrayLevelHistogram(bins=(0, 1, 100)).to(self.device)

    def __call__(self, rgb_tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.process(rgb_tensor)

    def process(self, rgb_tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Process an RGB tensor and return extracted features.

        Args:
            rgb_tensor: Tensor of shape ``(N, 3, H, W)`` in ``[0, 1]``.

        Returns:
            A tuple ``(edge, gray_level, segmented_rgb, ab)`` with each element as
            a ``torch.Tensor`` placed back on the input device.
        """

        device = rgb_tensor.device
        rgb_tensor_cpu = rgb_tensor.to("cpu")

        lab_clip = rgb_to_lab(rgb_tensor_cpu)
        gray_clip = torchvision.transforms.Grayscale()(rgb_tensor_cpu)
        edge = self.edge_detector(gray_clip)

        l_channel = (lab_clip[:, 0:1, :, :] - 50) / 50
        gray_level = self.gray_hist(l_channel)
        ab = lab_clip[:, 1:, :, :]

        rgb_shape = rgb_tensor_cpu.shape
        segmented_rgb = torch.zeros(rgb_shape, dtype=rgb_tensor_cpu.dtype)

        for i in range(rgb_shape[0]):
            flat_rgb = rgb_tensor_cpu[i].permute(1, 2, 0).reshape(-1, 3).numpy()
            self.kmeans.fit(flat_rgb)
            labels = self.kmeans.labels_
            centers = self.kmeans.cluster_centers_
            segmented = centers[labels].reshape(rgb_shape[2], rgb_shape[3], 3)
            segmented_rgb[i] = torch.tensor(
                segmented, dtype=rgb_tensor_cpu.dtype
            ).permute(2, 0, 1)

        return (
            edge.to(device),
            gray_level.to(device),
            segmented_rgb.to(device),
            ab.to(device),
        )


__all__ = ["FeatureExtractor"]
