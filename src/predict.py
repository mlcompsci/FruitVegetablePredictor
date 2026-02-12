"""
Single-image prediction from the command line.

Usage
-----
    python -m src.predict --image path/to/image.jpg --top_k 3
"""

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from src.config import get_category
from src.dataset import get_eval_transforms
from src.train import build_model
from src.utils import (
    get_device,
    load_class_metadata,
    load_label_map,
    load_model_weights,
)


def predict_image(
    image_path: str,
    top_k: int = 3,
) -> List[Dict[str, object]]:
    """Load model and return top-k predictions for a single image.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    list of dict
        Each dict has keys: label, category, probability.
    """
    device = get_device()

    # Load artifacts
    idx_to_class = load_label_map()
    num_classes = len(idx_to_class)
    metadata = load_class_metadata()

    # Build and load model
    model = build_model(num_classes)
    model = load_model_weights(model, device=device)

    # Preprocess image
    transform = get_eval_transforms()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = probs.topk(min(top_k, num_classes))

    results: List[Dict[str, object]] = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        label = idx_to_class[idx]
        cat_info = metadata.get(label, {})
        results.append(
            {
                "label": label,
                "category": cat_info.get("category", get_category(label)),
                "probability": round(prob, 4),
            }
        )
    return results


# ───────────────── CLI ─────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict produce from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions.")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"[ERROR] Image not found: {path}")
        return

    results = predict_image(str(path), top_k=args.top_k)

    print(f"\nPredictions for: {path.name}")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        print(
            f"  {i}. {r['label']:<20s} [{r['category']:<10s}]  "
            f"{r['probability'] * 100:.2f}%"
        )
    print()


if __name__ == "__main__":
    main()
