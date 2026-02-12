"""
Training script for the Produce Identifier.

Two-phase transfer learning:
  Phase 1 – Freeze backbone, train only the classifier head (fast convergence)
  Phase 2 – Unfreeze all layers, fine-tune with a very low LR (refine features)

This prevents the high learning rate from destroying pretrained ImageNet features,
which is the #1 reason small-dataset fine-tuning fails on real-world images.

Usage
-----
    python -m src.train --data_dir data/processed --epochs 25 --batch_size 32
"""

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from src.config import (
    ALLOWED_EXTENSIONS,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    EARLY_STOPPING_PATIENCE,
    TRAINING_CURVES_PATH,
)
from src.dataset import get_dataloaders
from src.utils import (
    get_device,
    print_metrics,
    save_class_metadata,
    save_label_map,
    save_model,
    set_seed,
)


# ───────────────── model factory ───────────────────────────────
def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """Build EfficientNet-B0 with a custom classifier head.

    EfficientNet-B0 has far better feature extraction than MobileNetV3-Small,
    and is still fast enough for real-time inference.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    freeze_backbone : bool
        If True, freeze all layers except the classifier.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze the feature-extraction layers."""
    for param in model.features.parameters():
        param.requires_grad = False


def _unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ───────────────── dataset validation ──────────────────────────
def _validate_data_dir(data_dir: Path) -> bool:
    """Return True if processed data directory has train/val/test splits with images."""
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            return False
        classes = [d for d in split_dir.iterdir() if d.is_dir()]
        if not classes:
            return False
    return True


def _log_class_distribution(data_dir: Path) -> None:
    """Print class distribution for each split."""
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        counts = {}
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = sum(1 for f in cls_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS)
                if n > 0:
                    counts[cls_dir.name] = n
        total = sum(counts.values())
        print(f"[INFO] {split:>5s}: {total:>6d} images across {len(counts)} classes")
        if len(counts) <= 30:
            for name, n in sorted(counts.items()):
                print(f"         {name:<22s} {n:>5d}")


# ───────────────── single epoch helpers ────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run validation. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ───────────────── plotting ────────────────────────────────────
def plot_training_curves(
    history: Dict[str, List[float]], save_path: Path
) -> None:
    """Save training/validation loss and accuracy curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved to {save_path}")


# ───────────────── main training loop ──────────────────────────
def train(
    data_dir: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    auto_download: bool = False,
) -> None:
    """Two-phase transfer learning pipeline.

    Phase 1 (head-only):  ~30% of total epochs, backbone frozen, LR = lr
    Phase 2 (full fine-tune): remaining epochs, all layers, LR = lr / 20
    """
    set_seed(seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    data_path = Path(data_dir)

    # ── Pre-flight: validate data exists ──
    if not _validate_data_dir(data_path):
        print("[WARN] Processed dataset not found or incomplete.")

        if auto_download:
            print("[INFO] Auto-download enabled – attempting bootstrap ...")
            try:
                from scripts.fetch_dataset import fetch_dataset, raw_data_sufficient
                from scripts.prepare_data import (
                    copy_split,
                    gather_class_images,
                    stratified_split,
                )

                if not raw_data_sufficient(DATA_RAW_DIR, min_samples=10):
                    fetch_dataset(raw_dir=DATA_RAW_DIR)

                class_images = gather_class_images(DATA_RAW_DIR)
                if class_images:
                    splits = stratified_split(class_images)
                    copy_split(splits, data_path)
            except Exception as exc:
                print(f"[ERROR] Auto-download/prepare failed: {exc}")

        # Re-check
        if not _validate_data_dir(data_path):
            print()
            print("=" * 60)
            print("[ERROR] Cannot train – processed data is missing.")
            print()
            print("  Option 1 (automatic):")
            print("    python -m scripts.bootstrap_and_train --auto_download true")
            print()
            print("  Option 2 (manual):")
            print(f"    1. Place images in {DATA_RAW_DIR}/<class_name>/")
            print("    2. python -m scripts.prepare_data")
            print("    3. python -m src.train")
            print("=" * 60)
            sys.exit(1)

    # ── Log class distribution ──
    print("\n[INFO] Dataset distribution:")
    _log_class_distribution(data_path)
    print()

    # Data
    train_loader, val_loader, _, idx_to_class = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
    )
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    print(f"[INFO] Found {num_classes} classes: {class_names}")

    # Persist label map and metadata
    save_label_map({i: name for i, name in enumerate(class_names)})
    metadata = save_class_metadata(class_names)

    # ── Phase planning ──
    phase1_epochs = max(3, epochs // 3)   # ~30% for head training
    phase2_epochs = epochs - phase1_epochs  # ~70% for fine-tuning
    finetune_lr = lr / 20.0               # much lower LR for backbone

    print(f"\n[INFO] Training plan:")
    print(f"  Phase 1: {phase1_epochs} epochs – classifier head only (LR={lr})")
    print(f"  Phase 2: {phase2_epochs} epochs – full fine-tune  (LR={finetune_lr:.6f})")

    # ── Build model with frozen backbone ──
    model = build_model(num_classes, freeze_backbone=True).to(device)

    # Label smoothing reduces overconfidence and improves generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1 optimizer: only classifier params
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer = AdamW(classifier_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    # Training state
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"\n{'=' * 76}")
    print(f"  PHASE 1: Training classifier head ({phase1_epochs} epochs)")
    print(f"{'=' * 76}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} {'Train Acc':>10} | "
          f"{'Val Loss':>10} {'Val Acc':>10} | {'LR':>10}")
    print("-" * 76)

    start = time.time()
    global_epoch = 0
    try:
        # ════════════════ PHASE 1: Head only ════════════════
        for epoch in range(1, phase1_epochs + 1):
            global_epoch += 1
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"{global_epoch:>6d} | {train_loss:>10.4f} {train_acc:>10.4f} | "
                f"{val_loss:>10.4f} {val_acc:>10.4f} | {current_lr:>10.6f}"
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

        # ════════════════ PHASE 2: Full fine-tune ════════════════
        print(f"\n{'=' * 76}")
        print(f"  PHASE 2: Full fine-tuning ({phase2_epochs} epochs)")
        print(f"{'=' * 76}")

        _unfreeze_backbone(model)

        # Differential LR: backbone gets a much lower rate
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": finetune_lr},
                {"params": head_params, "lr": finetune_lr * 5},  # head keeps a bit more
            ],
            weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs)

        # Reset early stopping for phase 2
        patience_counter = 0
        best_val_loss = float("inf")

        print(f"{'Epoch':>6} | {'Train Loss':>10} {'Train Acc':>10} | "
              f"{'Val Loss':>10} {'Val Acc':>10} | {'LR':>10}")
        print("-" * 76)

        for epoch in range(1, phase2_epochs + 1):
            global_epoch += 1
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"{global_epoch:>6d} | {train_loss:>10.4f} {train_acc:>10.4f} | "
                f"{val_loss:>10.4f} {val_acc:>10.4f} | {current_lr:>10.6f}"
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

            # Early stopping on val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n[INFO] Early stopping triggered at epoch {global_epoch}")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted – saving best model so far ...")

    elapsed = time.time() - start
    print(f"\n[INFO] Training completed in {elapsed:.1f}s")
    print(f"[INFO] Best validation accuracy: {best_val_acc:.4f}")

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
    save_model(model)
    print("[INFO] Best model saved.")

    # Plot
    if history["train_loss"]:
        plot_training_curves(history, TRAINING_CURVES_PATH)

    print_metrics(
        {
            "Best Val Accuracy": best_val_acc,
            "Final Train Loss": history["train_loss"][-1] if history["train_loss"] else 0,
            "Final Val Loss": history["val_loss"][-1] if history["val_loss"] else 0,
            "Total Epochs": len(history["train_loss"]),
        },
        title="Training Summary",
    )


# ───────────────── CLI ─────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the Produce Identifier model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_PROCESSED_DIR),
        help="Path to processed dataset root (default: data/processed).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--auto_download",
        action="store_true",
        default=False,
        help="Auto-download dataset if missing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        auto_download=args.auto_download,
    )
