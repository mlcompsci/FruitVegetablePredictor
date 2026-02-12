"""
One-command full pipeline: fetch data (if needed) -> prepare splits -> train -> done.

Usage
-----
    python -m scripts.bootstrap_and_train --auto_download true --epochs 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import (
    BEST_MODEL_PATH,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MAX_CLASSES,
    DEFAULT_MIN_SAMPLES_PER_CLASS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    DEMO_EPOCHS,
    LABEL_MAP_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
)


def _ensure_dirs() -> None:
    """Create required project directories if missing."""
    for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print("[BOOT] Directories verified.")


def _step_fetch(
    auto_download: bool,
    min_samples: int,
    max_classes: int,
    source: str,
) -> bool:
    """Step 1: ensure raw data exists, downloading if needed."""
    from scripts.fetch_dataset import fetch_dataset, raw_data_sufficient

    if raw_data_sufficient(DATA_RAW_DIR, min_samples):
        print("[BOOT] Raw data already present – skipping download.")
        return True

    if not auto_download:
        print("[ERROR] No raw data found and --auto_download is false.")
        print(f"        Place images in: {DATA_RAW_DIR}/<class_name>/")
        print("        Or re-run with: --auto_download true")
        return False

    return fetch_dataset(
        raw_dir=DATA_RAW_DIR,
        min_samples=min_samples,
        max_classes=max_classes,
        source=source,
    )


def _step_prepare() -> bool:
    """Step 2: run stratified split."""
    print("\n[BOOT] Preparing train/val/test splits ...")
    try:
        from scripts.prepare_data import (
            copy_split,
            gather_class_images,
            stratified_split,
        )

        class_images = gather_class_images(DATA_RAW_DIR)
        if not class_images:
            print("[ERROR] No image classes found in raw data.")
            return False

        splits = stratified_split(class_images)
        copy_split(splits, DATA_PROCESSED_DIR)
        print("[BOOT] Data preparation complete.")
        return True
    except Exception as exc:
        print(f"[ERROR] Data preparation failed: {exc}")
        return False


def _step_train(
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    seed: int,
) -> bool:
    """Step 3: train the model."""
    print(f"\n[BOOT] Training model for {epochs} epochs ...")
    try:
        from src.train import train

        train(
            data_dir=str(DATA_PROCESSED_DIR),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
            seed=seed,
        )

        if not BEST_MODEL_PATH.exists():
            print("[ERROR] Training completed but model file not found.")
            return False

        print("[BOOT] Training complete! Model saved.")
        return True
    except KeyboardInterrupt:
        print("\n[BOOT] Training interrupted by user.")
        return False
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        return False


def _print_next_steps() -> None:
    """Print instructions for launching the app."""
    print("\n" + "=" * 60)
    print("  SUCCESS – Produce Identifier is ready!")
    print("=" * 60)
    print()
    print("  Launch the web app:")
    print("    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("  Then open:  http://localhost:8000")
    print()
    print("  Or predict from CLI:")
    print("    python -m src.predict --image path/to/photo.jpg --top_k 3")
    print("=" * 60)


def bootstrap(
    auto_download: bool = True,
    min_samples: int = DEFAULT_MIN_SAMPLES_PER_CLASS,
    max_classes: int = DEFAULT_MAX_CLASSES,
    epochs: int = DEMO_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    source: str = "auto",
) -> bool:
    """Run the full bootstrap pipeline. Returns True on success."""
    overall_start = time.time()
    print("=" * 60)
    print("  Produce Identifier – Full Bootstrap Pipeline")
    print("=" * 60)

    # Step 0: dirs
    _ensure_dirs()

    # Step 1: fetch
    if not _step_fetch(auto_download, min_samples, max_classes, source):
        return False

    # Step 2: prepare
    if not _step_prepare():
        return False

    # Step 3: train
    if not _step_train(epochs, batch_size, lr, num_workers, seed):
        return False

    elapsed = time.time() - overall_start
    print(f"\n[BOOT] Total pipeline time: {elapsed:.1f}s")
    _print_next_steps()
    return True


# ───────────────── CLI ─────────────────────────────────────────
def _str2bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes", "y")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command bootstrap: fetch data + prepare + train."
    )
    parser.add_argument("--auto_download", type=_str2bool, default=True,
                        help="Auto-download dataset if missing (default: true).")
    parser.add_argument("--min_samples_per_class", type=int, default=DEFAULT_MIN_SAMPLES_PER_CLASS)
    parser.add_argument("--max_classes", type=int, default=DEFAULT_MAX_CLASSES)
    parser.add_argument("--epochs", type=int, default=DEMO_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "kaggle", "huggingface", "zip"],
                        help="Force download source or 'auto'.")
    args = parser.parse_args()

    ok = bootstrap(
        auto_download=args.auto_download,
        min_samples=args.min_samples_per_class,
        max_classes=args.max_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        source=args.source,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
