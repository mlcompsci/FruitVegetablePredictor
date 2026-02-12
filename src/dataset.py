"""
Dataset and DataLoader utilities for the Produce Identifier.

Provides:
- torchvision transform pipelines for train / val / test
- `get_dataloaders` that wraps ImageFolder into DataLoaders
"""

from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    DATA_PROCESSED_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ───────────────── transform pipelines ─────────────────────────
def get_train_transforms() -> transforms.Compose:
    """Heavy augmentation pipeline for training images.

    Aggressive transforms force the model to learn real visual features
    instead of memorizing dataset-specific artifacts like backgrounds,
    lighting, and camera angles.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """Deterministic pipeline for validation / test / inference images."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE + 32),  # slight over-size then crop
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ───────────────── dataloaders ─────────────────────────────────
def get_dataloaders(
    data_dir: Path | str | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Create train, val, and test DataLoaders from the processed dataset.

    Parameters
    ----------
    data_dir : Path, optional
        Root of the processed dataset (contains train/val/test subdirs).
        Defaults to ``DATA_PROCESSED_DIR``.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of parallel data-loading workers.

    Returns
    -------
    train_loader, val_loader, test_loader, idx_to_class
    """
    data_dir = Path(data_dir) if data_dir else DATA_PROCESSED_DIR

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                "Run `python -m scripts.prepare_data` first."
            )

    train_ds = datasets.ImageFolder(str(train_dir), transform=get_train_transforms())
    val_ds = datasets.ImageFolder(str(val_dir), transform=get_eval_transforms())
    test_ds = datasets.ImageFolder(str(test_dir), transform=get_eval_transforms())

    # Ensure consistent class ordering across splits
    assert train_ds.classes == val_ds.classes == test_ds.classes, (
        "Class lists differ across splits. Regenerate splits."
    )

    idx_to_class: Dict[int, str] = {v: k for k, v in train_ds.class_to_idx.items()}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, idx_to_class
