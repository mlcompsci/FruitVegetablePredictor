"""
Inference engine – loads model once and exposes a prediction function.

Used by the FastAPI app to avoid reloading weights on every request.
"""

from io import BytesIO
from typing import Dict, List, Optional, Tuple

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


class ProduceClassifier:
    """Singleton-style classifier that loads model artifacts once."""

    def __init__(self) -> None:
        self.device: torch.device = get_device()
        self.model: Optional[torch.nn.Module] = None
        self.idx_to_class: Dict[int, str] = {}
        self.metadata: Dict[str, Dict[str, str]] = {}
        self.transform = get_eval_transforms()
        self._loaded = False

    def load(self) -> None:
        """Load model weights, label map, and class metadata from disk."""
        if self._loaded:
            return

        self.idx_to_class = load_label_map()
        self.metadata = load_class_metadata()
        num_classes = len(self.idx_to_class)

        self.model = build_model(num_classes)
        self.model = load_model_weights(self.model, device=self.device)
        self._loaded = True
        print(f"[INFO] Model loaded on {self.device} with {num_classes} classes.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ─────────────── prediction ─────────────────────────────────
    def predict(
        self, image_bytes: bytes, top_k: int = 3
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        """Run inference on raw image bytes.

        Returns
        -------
        top_k_results : list of dict
            Each dict: {label, category, probability}
        summary : dict
            {fruit_probability, vegetable_probability, predicted_supercategory}
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Decode and preprocess
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        num_classes = probs.size(0)
        k = min(top_k, num_classes)
        top_probs, top_indices = probs.topk(k)

        # Build top-k list
        top_k_results: List[Dict[str, object]] = []
        for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
            label = self.idx_to_class[idx]
            cat_info = self.metadata.get(label, {})
            category = cat_info.get("category", get_category(label))
            top_k_results.append(
                {"label": label, "category": category, "probability": round(prob, 4)}
            )

        # Build super-category summary over ALL classes
        all_probs = probs.cpu().tolist()
        fruit_prob = 0.0
        veg_prob = 0.0
        for i, p in enumerate(all_probs):
            label = self.idx_to_class[i]
            cat_info = self.metadata.get(label, {})
            cat = cat_info.get("category", get_category(label))
            if cat == "fruit":
                fruit_prob += p
            elif cat == "vegetable":
                veg_prob += p

        summary = {
            "fruit_probability": round(fruit_prob, 4),
            "vegetable_probability": round(veg_prob, 4),
            "predicted_supercategory": "fruit" if fruit_prob >= veg_prob else "vegetable",
        }

        return top_k_results, summary


# Module-level singleton
classifier = ProduceClassifier()
