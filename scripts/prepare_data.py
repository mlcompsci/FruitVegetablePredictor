"""
Stratified split of the raw dataset into train / val / test.

Expects raw data arranged as::

    data/raw/<class_name>/*.jpg

Produces::

    data/processed/train/<class_name>/*.jpg
    data/processed/val/<class_name>/*.jpg
    data/processed/test/<class_name>/*.jpg

Usage
-----
    python -m scripts.prepare_data [--raw_dir data/raw] [--out_dir data/processed]
                                   [--train_ratio 0.7] [--val_ratio 0.15] [--seed 42]
"""

import argparse
import shutil
from pathlib import Path
from typing import List

import numpy as np

from src.config import ALLOWED_EXTENSIONS, DATA_PROCESSED_DIR, DATA_RAW_DIR


def gather_class_images(raw_dir: Path) -> dict[str, List[Path]]:
    """Return {class_name: [image_paths]} from the raw directory."""
    class_images: dict[str, List[Path]] = {}
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if images:
            class_images[class_dir.name] = sorted(images)
    return class_images


def stratified_split(
    class_images: dict[str, List[Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, List[Path]]]:
    """Split each class's images into train/val/test.

    Returns
    -------
    {"train": {class: [paths]}, "val": {…}, "test": {…}}
    """
    rng = np.random.default_rng(seed)
    splits: dict[str, dict[str, List[Path]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    for cls_name, paths in class_images.items():
        n = len(paths)
        indices = rng.permutation(n)

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        # Guarantee at least 1 sample per split if possible
        if len(test_idx) == 0 and n >= 3:
            test_idx = val_idx[-1:]
            val_idx = val_idx[:-1]

        splits["train"][cls_name] = [paths[i] for i in train_idx]
        splits["val"][cls_name] = [paths[i] for i in val_idx]
        splits["test"][cls_name] = [paths[i] for i in test_idx]

    return splits


def copy_split(
    splits: dict[str, dict[str, List[Path]]],
    out_dir: Path,
) -> None:
    """Copy images into the processed directory structure."""
    if out_dir.exists():
        print(f"[INFO] Removing existing processed directory: {out_dir}")
        shutil.rmtree(out_dir)

    total = 0
    for split_name, class_dict in splits.items():
        for cls_name, paths in class_dict.items():
            dest = out_dir / split_name / cls_name
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in paths:
                shutil.copy2(img_path, dest / img_path.name)
                total += 1
        split_total = sum(len(v) for v in class_dict.values())
        print(f"  {split_name:>5s}: {split_total:>6d} images across {len(class_dict)} classes")

    print(f"[INFO] Total images copied: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stratified train/val/test splits.")
    parser.add_argument("--raw_dir", type=str, default=str(DATA_RAW_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DATA_PROCESSED_DIR))
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    print(f"[INFO] Raw data directory : {raw_dir}")
    print(f"[INFO] Output directory   : {out_dir}")
    print(f"[INFO] Split ratios       : train={args.train_ratio}, val={args.val_ratio}, "
          f"test={round(1 - args.train_ratio - args.val_ratio, 2)}")

    class_images = gather_class_images(raw_dir)
    if not class_images:
        print("[ERROR] No image classes found. Place images in data/raw/<class_name>/.")
        return

    print(f"[INFO] Found {len(class_images)} classes, "
          f"{sum(len(v) for v in class_images.values())} total images.")

    splits = stratified_split(
        class_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    copy_split(splits, out_dir)
    print("[INFO] Data preparation complete!")


if __name__ == "__main__":
    main()
