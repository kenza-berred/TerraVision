"""Image transforms shared by training and inference.

Inference must apply exactly the eval transform used during validation, with
the normalisation stats saved alongside the checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torchvision.transforms import v2


class RandomRot90(torch.nn.Module):
    """Rotate by a random multiple of 90°.

    Overhead imagery has no canonical "up", so 0/90/180/270° are all realistic
    views. torch.rot90 moves pixels exactly (no interpolation), unlike a generic
    rotation. Applies the same k to every input, so it can rotate an image and
    its mask together.
    """

    def forward(self, *inputs: torch.Tensor):
        k = int(torch.randint(0, 4, (1,)))
        out = tuple(torch.rot90(x, k, dims=(-2, -1)) for x in inputs)
        return out[0] if len(out) == 1 else out


def classification_transforms(mean: Sequence[float], std: Sequence[float],
                              train: bool) -> v2.Compose:
    """PIL image -> normalised float tensor (C, H, W).

    Training augmentation (justification):
    - horizontal/vertical flips + 90° rotations: geometry-preserving views of
      the same ground scene.
    - mild brightness/contrast jitter (±10%): atmospheric and illumination
      differences between acquisitions.
    - no hue/saturation changes: colour is a key cue (water vs. vegetation).
    """
    steps: list = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if train:
        steps += [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            RandomRot90(),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps.append(v2.Normalize(mean=list(mean), std=list(std)))
    return v2.Compose(steps)
