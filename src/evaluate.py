"""
Evaluation script – compute test-set metrics and confusion matrix.

Usage
-----
    python -m src.evaluate --data_dir data/processed
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import (
    CONFUSION_MATRIX_PATH,
    DATA_PROCESSED_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)
from src.dataset import get_dataloaders
from src.train import build_model
from src.utils import (
    get_device,
    load_class_metadata,
    load_label_map,
    load_model_weights,
    print_metrics,
    set_seed,
)


# ───────────────── gather predictions ──────────────────────────
@torch.no_grad()
def gather_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (all_labels, all_preds) as numpy arrays."""
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


# ───────────────── confusion matrix plot ───────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> None:
    """Create and save a confusion-matrix heatmap."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)

    fig_size = max(8, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved to {save_path}")


# ───────────────── super-category accuracy ─────────────────────
def supercategory_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: Dict[int, str],
    metadata: Dict[str, Dict[str, str]],
) -> Dict[str, float]:
    """Compute accuracy at the fruit-vs-vegetable level."""
    true_cats = [metadata.get(idx_to_class[int(t)], {}).get("category", "unknown") for t in y_true]
    pred_cats = [metadata.get(idx_to_class[int(p)], {}).get("category", "unknown") for p in y_pred]

    correct = sum(1 for tc, pc in zip(true_cats, pred_cats) if tc == pc)
    total = len(true_cats)
    return {"supercategory_accuracy": correct / total if total else 0.0}


# ───────────────── main evaluate ───────────────────────────────
def evaluate(
    data_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> None:
    """Full evaluation pipeline on the test split."""
    set_seed()
    device = get_device()

    # Load label map and metadata
    idx_to_class = load_label_map()
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    metadata = load_class_metadata()

    # Build and load model
    model = build_model(num_classes)
    model = load_model_weights(model, device=device)

    # Data
    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
    )

    # Predictions
    y_true, y_pred = gather_predictions(model, test_loader, device)

    # Overall accuracy
    overall_acc = (y_true == y_pred).mean()

    # Precision / recall / F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print_metrics(
        {
            "Overall Test Accuracy": overall_acc,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1,
        },
        title="Test Set Metrics",
    )

    # Per-class report
    print("\nPer-class Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, CONFUSION_MATRIX_PATH)

    # Super-category accuracy
    sc = supercategory_accuracy(y_true, y_pred, idx_to_class, metadata)
    print_metrics(sc, title="Super-category (Fruit vs Vegetable)")


# ───────────────── CLI ─────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Produce Identifier model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_PROCESSED_DIR),
        help="Path to processed dataset root.",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
