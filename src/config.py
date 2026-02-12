"""
Central configuration for the Produce Identifier project.

Edit CATEGORY_MAP to correct any fruit/vegetable classification.
"""

from pathlib import Path
from typing import Dict

# ──────────────────────────── paths ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
CLASS_METADATA_PATH = MODELS_DIR / "class_metadata.json"

TRAINING_CURVES_PATH = OUTPUTS_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"

# ──────────────────────────── image ────────────────────────────
IMAGE_SIZE: int = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────── training defaults ────────────────
DEFAULT_EPOCHS: int = 20
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LR: float = 1e-3
DEFAULT_NUM_WORKERS: int = 2
DEFAULT_SEED: int = 42
EARLY_STOPPING_PATIENCE: int = 5

# ──────────────────────────── bootstrap defaults ───────────────
DEFAULT_MIN_SAMPLES_PER_CLASS: int = 30
DEFAULT_MAX_CLASSES: int = 25
DEMO_EPOCHS: int = 3

# ──────────────────────────── dataset sources ──────────────────
# Primary: Kaggle via kagglehub
KAGGLE_DATASET_SLUG = "kritikseth/fruit-and-vegetable-image-recognition"
# Fallback 1: Hugging Face dataset
HF_DATASET_NAME = "EdBianchi/fruits-vegetables-dataset"
# Fallback 2: Direct ZIP URL (public Google Drive / GitHub release / etc.)
DIRECT_ZIP_URL = (
    "https://github.com/EdBianchi/fruits-vegetables-dataset/archive/refs/heads/main.zip"
)

# Download / network settings
DOWNLOAD_TIMEOUT_SECONDS: int = 300
DOWNLOAD_RETRY_COUNT: int = 3

# ──────────────────────────── category mapping ─────────────────
# Edit this dictionary to add or correct produce categories.
# Key: class name (must match folder name in data/raw/).
# Value: "fruit" or "vegetable".
CATEGORY_MAP: Dict[str, str] = {
    # Fruits
    "apple": "fruit",
    "banana": "fruit",
    "orange": "fruit",
    "grape": "fruit",
    "strawberry": "fruit",
    "blueberry": "fruit",
    "mango": "fruit",
    "pineapple": "fruit",
    "watermelon": "fruit",
    "pear": "fruit",
    "peach": "fruit",
    "cherry": "fruit",
    "kiwi": "fruit",
    "lemon": "fruit",
    "lime": "fruit",
    "plum": "fruit",
    "pomegranate": "fruit",
    "raspberry": "fruit",
    "papaya": "fruit",
    "coconut": "fruit",
    "fig": "fruit",
    "apricot": "fruit",
    "cantaloupe": "fruit",
    "grapefruit": "fruit",
    "guava": "fruit",
    "lychee": "fruit",
    "passion_fruit": "fruit",
    "dragonfruit": "fruit",
    # Vegetables
    "carrot": "vegetable",
    "broccoli": "vegetable",
    "tomato": "vegetable",
    "cucumber": "vegetable",
    "onion": "vegetable",
    "potato": "vegetable",
    "bell_pepper": "vegetable",
    "spinach": "vegetable",
    "lettuce": "vegetable",
    "corn": "vegetable",
    "peas": "vegetable",
    "cabbage": "vegetable",
    "cauliflower": "vegetable",
    "garlic": "vegetable",
    "ginger": "vegetable",
    "mushroom": "vegetable",
    "eggplant": "vegetable",
    "zucchini": "vegetable",
    "asparagus": "vegetable",
    "celery": "vegetable",
    "radish": "vegetable",
    "sweet_potato": "vegetable",
    "turnip": "vegetable",
    "beetroot": "vegetable",
    "artichoke": "vegetable",
    "kale": "vegetable",
    "leek": "vegetable",
    "pumpkin": "vegetable",
    "squash": "vegetable",
    "jalapeno": "vegetable",
    "chili_pepper": "vegetable",
    # Extended names often found in public datasets
    "capsicum": "vegetable",
    "bean": "vegetable",
    "bitter_gourd": "vegetable",
    "bottle_gourd": "vegetable",
    "papaya_fruit": "fruit",
    "bell pepper": "vegetable",
    "chilli pepper": "vegetable",
    "sweetpotato": "vegetable",
    "sweetcorn": "vegetable",
    "soy_bean": "vegetable",
    "soy bean": "vegetable",
    "raddish": "vegetable",
    "grapes": "fruit",
}

# Allowed image extensions for dataset loading
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Supported upload MIME types for the API
ALLOWED_UPLOAD_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def get_category(class_name: str) -> str:
    """Return 'fruit' or 'vegetable' for *class_name*.

    Falls back to 'unknown' if not present in CATEGORY_MAP.
    """
    return CATEGORY_MAP.get(class_name.lower(), "unknown")


def is_produce_class(name: str) -> bool:
    """Return True if *name* is a known fruit or vegetable."""
    return get_category(name) in ("fruit", "vegetable")
