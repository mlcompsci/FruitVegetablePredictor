"""
Utility helpers shared across training, evaluation, and prediction.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.config import (
    BEST_MODEL_PATH,
    CLASS_METADATA_PATH,
    LABEL_MAP_PATH,
    get_category,
)


# ─────────────────── reproducibility ───────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────── device ────────────────────────────────────
def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────── label map I/O ─────────────────────────────
def save_label_map(idx_to_class: Dict[int, str], path: Optional[Path] = None) -> None:
    """Persist index-to-class-name mapping as JSON."""
    path = path or LABEL_MAP_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(idx_to_class, fh, indent=2)


def load_label_map(path: Optional[Path] = None) -> Dict[int, str]:
    """Load index-to-class-name mapping from JSON.

    Keys are cast to ``int`` for consistency.
    """
    path = path or LABEL_MAP_PATH
    if not path.exists():
        raise FileNotFoundError(f"Label map not found at {path}. Train a model first.")
    with open(path, "r", encoding="utf-8") as fh:
        raw: Dict[str, str] = json.load(fh)
    return {int(k): v for k, v in raw.items()}


# ─────────────────── class metadata I/O ────────────────────────
def build_class_metadata(class_names: List[str]) -> Dict[str, Dict[str, str]]:
    """Build metadata dict mapping each class to its super-category."""
    return {name: {"category": get_category(name)} for name in class_names}


def save_class_metadata(
    class_names: List[str], path: Optional[Path] = None
) -> Dict[str, Dict[str, str]]:
    """Build and persist class metadata JSON. Returns the metadata dict."""
    path = path or CLASS_METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_class_metadata(class_names)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def load_class_metadata(path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Load class metadata from JSON."""
    path = path or CLASS_METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Class metadata not found at {path}. Train a model first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────── model I/O ─────────────────────────────────
def save_model(model: torch.nn.Module, path: Optional[Path] = None) -> None:
    """Save model state dict."""
    path = path or BEST_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_weights(
    model: torch.nn.Module, path: Optional[Path] = None, device: Optional[torch.device] = None
) -> torch.nn.Module:
    """Load saved weights into *model* and return it in eval mode."""
    path = path or BEST_MODEL_PATH
    device = device or get_device()
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found at {path}. Train first.")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


# ─────────────────── pretty printing ──────────────────────────
def print_metrics(metrics: Dict[str, Any], title: str = "Metrics") -> None:
    """Print a metrics dictionary in a readable table."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<30s} {value:.4f}")
        else:
            print(f"  {key:<30s} {value}")
    print(f"{'=' * 50}\n")
