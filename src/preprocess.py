from __future__ import annotations

import random
from typing import Tuple

import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageNet
from torch.utils.data import DataLoader


# -------------------------------------------------------------------------
# transformations
# -------------------------------------------------------------------------

def _build_transforms(img_size: int = 224):
    return T.Compose(
        [
            T.Resize(img_size, antialias=True),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# -------------------------------------------------------------------------
# dataloader helpers
# -------------------------------------------------------------------------

def load_stream(dataset: str, root: str, batch_size: int, num_workers: int, *, smoke: bool):
    """Return DataLoader with batch = 1 for streaming evaluation."""
    transform = _build_transforms(224)

    if dataset.lower() == "cifar10":
        ds = CIFAR10(root, train=False, transform=transform, download=True)
        if smoke:
            ds = torch.utils.data.Subset(ds, list(range(256)))
    elif dataset.lower() == "imagenet":
        split = "val"
        ds = ImageNet(root, split=split, transform=transform)
        if smoke:
            indices = random.sample(range(len(ds)), 512)
            ds = torch.utils.data.Subset(ds, indices)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
