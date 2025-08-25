"""Image transform that crops using the shorter edge."""

from __future__ import annotations

from PIL import Image
import torchvision.transforms as transforms


class RandomCropShorterEdge:
    """Crop the image to a square based on the shorter edge."""

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        short_edge = min(width, height)
        return transforms.CenterCrop(short_edge)(img)


__all__ = ["RandomCropShorterEdge"]
