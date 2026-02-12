# Produce Identifier

A complete end-to-end **Fruits & Vegetables Image Classifier** that can fully self-bootstrap -- automatically downloading a public dataset, preparing splits, training a model, and serving predictions via API and web UI.

---

## Table of Contents

1. [Fastest Start (No Data Needed)](#fastest-start-no-data-needed)
2. [Manual Data Mode](#manual-data-mode)
3. [Project Overview](#project-overview)
4. [Folder Structure](#folder-structure)
5. [Setup](#setup)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Single-Image Prediction](#single-image-prediction)
9. [Running the Web App / API](#running-the-web-app--api)
10. [API Reference](#api-reference)
11. [Editing the Fruit / Vegetable Mapping](#editing-the-fruit--vegetable-mapping)
12. [Troubleshooting Auto-Download](#troubleshooting-auto-download)
13. [Expected Duration](#expected-duration)
14. [Improving Accuracy](#improving-accuracy)
15. [Troubleshooting](#troubleshooting)

---

## Fastest Start (No Data Needed)

Two commands and you're running:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Auto-download dataset + prepare splits + train (3 demo epochs)
python -m scripts.bootstrap_and_train --auto_download true --epochs 3

# 3. Launch the web app
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** -- upload any fruit or vegetable photo and get instant predictions.

> **Alternative: Use the Web UI.**  Just start the server (step 3) and click the **"Auto Setup & Train (Demo)"** button that appears in the UI when no model is found.

---

## Manual Data Mode

If you already have a dataset or want full control:

```bash
# 1. Place images in data/raw/<class_name>/
#    Example:
#      data/raw/apple/apple_001.jpg
#      data/raw/carrot/carrot_001.jpg

# 2. Create stratified train/val/test splits
python -m scripts.prepare_data

# 3. Train (full 20 epochs)
python -m src.train --data_dir data/processed --epochs 20

# 4. Evaluate on test set
python -m src.evaluate --data_dir data/processed

# 5. Launch API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Project Overview

| Component | Detail |
|-----------|--------|
| **Model** | MobileNetV3-Small (pretrained on ImageNet) with a custom classifier head |
| **Input** | 224 x 224 RGB images |
| **Output** | Top-k class labels, each tagged as *fruit* or *vegetable*, plus aggregated super-category probabilities |
| **Stack** | Python 3.11, PyTorch, torchvision, FastAPI, vanilla HTML/CSS/JS |
| **Auto-download** | Kaggle (primary), Hugging Face (fallback), direct ZIP (fallback) |

---

## Folder Structure

```
produce-identifier/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── raw/                      # Place or auto-download dataset here
│   └── processed/                # Generated splits
├── models/
│   ├── best_model.pth            # Saved after training
│   ├── label_map.json
│   └── class_metadata.json
├── outputs/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, hyper-params, category map, source URLs
│   ├── dataset.py                # Transforms & DataLoaders
│   ├── train.py                  # Training loop + CLI (with auto-download flag)
│   ├── evaluate.py               # Test metrics + confusion matrix
│   ├── predict.py                # CLI single-image prediction
│   └── utils.py                  # Seed, I/O helpers
├── app/
│   ├── main.py                   # FastAPI: /, /predict, /health, /status, /setup-demo
│   ├── schemas.py                # Pydantic models
│   ├── inference.py              # Model loader (singleton)
│   ├── static/
│   │   ├── styles.css
│   │   └── app.js
│   └── templates/
│       └── index.html
└── scripts/
    ├── fetch_dataset.py          # Auto-download with Kaggle/HF/ZIP fallbacks
    ├── prepare_data.py           # Stratified train/val/test split
    ├── bootstrap_and_train.py    # One-command full pipeline
    ├── verify_pipeline.py        # End-to-end verification
    └── run_dev.sh                # Quick-start dev server
```

---

## Setup

```bash
cd produce-identifier

# Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **GPU support:** PyTorch will use CUDA automatically if a compatible GPU and drivers are available. No code changes needed.

---

## Training

### Quick demo (auto-download everything)

```bash
python -m scripts.bootstrap_and_train --auto_download true --epochs 3
```

### Full training (after manual data setup)

```bash
python -m src.train --data_dir data/processed --epochs 20 --batch_size 32 --lr 0.001
```

All CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data/processed` | Root of the processed splits |
| `--epochs` | `20` | Max training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--num_workers` | `2` | DataLoader workers |
| `--seed` | `42` | Random seed |
| `--auto_download` | flag | If set, auto-download data when missing |

### Bootstrap CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--auto_download` | `true` | Fetch dataset automatically |
| `--min_samples_per_class` | `30` | Minimum images per class |
| `--max_classes` | `25` | Keep top N classes by sample count |
| `--epochs` | `3` | Training epochs for demo |
| `--source` | `auto` | Force: `kaggle`, `huggingface`, `zip`, or `auto` |

---

## Evaluation

```bash
python -m src.evaluate --data_dir data/processed
```

---

## Single-Image Prediction

```bash
python -m src.predict --image path/to/photo.jpg --top_k 3
```

---

## Running the Web App / API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser. If no model is trained, the UI shows a smart banner with an "Auto Setup & Train" button.

---

## API Reference

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### `GET /status`

Shows system readiness and suggests next action.

```bash
curl http://localhost:8000/status
```

**Before training:**

```json
{
  "model_loaded": false,
  "raw_data_present": false,
  "processed_data_present": false,
  "num_classes": null,
  "suggested_next_command": "python -m scripts.bootstrap_and_train --auto_download true --epochs 3"
}
```

**After training:**

```json
{
  "model_loaded": true,
  "raw_data_present": true,
  "processed_data_present": true,
  "num_classes": 18,
  "suggested_next_command": null
}
```

### `POST /setup-demo`

Runs the full bootstrap pipeline from the UI.

```bash
curl -X POST http://localhost:8000/setup-demo
```

```json
{
  "success": true,
  "logs": [
    {"step": "fetch", "message": "Checking / downloading dataset ...", "success": true},
    {"step": "fetch", "message": "Dataset downloaded successfully.", "success": true},
    {"step": "prepare", "message": "Splits created: 18 classes.", "success": true},
    {"step": "train", "message": "Training complete. Model saved.", "success": true},
    {"step": "reload", "message": "Model loaded with 18 classes.", "success": true}
  ],
  "message": "Setup complete in 142.3s. You can now upload images to classify!",
  "duration_seconds": 142.3
}
```

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict -F "file=@apple.jpg"
```

**Success (200):**

```json
{
  "top_k": [
    {"label": "apple", "category": "fruit", "probability": 0.9321},
    {"label": "pear", "category": "fruit", "probability": 0.041},
    {"label": "tomato", "category": "vegetable", "probability": 0.0158}
  ],
  "summary": {
    "fruit_probability": 0.9794,
    "vegetable_probability": 0.0206,
    "predicted_supercategory": "fruit"
  }
}
```

**Error -- no model (503):**

```json
{
  "detail": "Model not loaded. Train a model first using:\n  python -m scripts.bootstrap_and_train --auto_download true --epochs 3\nThen restart the server, or use the 'Auto Setup & Train' button in the UI."
}
```

**Error -- bad file type (400):**

```json
{
  "detail": "Unsupported file type 'application/pdf'. Allowed: image/bmp, image/jpeg, image/png, image/webp"
}
```

---

## Editing the Fruit / Vegetable Mapping

Open `src/config.py` and edit the `CATEGORY_MAP` dictionary:

```python
CATEGORY_MAP = {
    "apple": "fruit",
    "tomato": "vegetable",   # change to "fruit" if you prefer botanical classification
    # Add new classes here ...
}
```

After editing, re-run training so that `class_metadata.json` is regenerated.

---

## Troubleshooting Auto-Download

| Issue | Fix |
|-------|-----|
| **Kaggle requires authentication** | Run `pip install kagglehub` then `kagglehub` will prompt for login on first use, or set `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars. The system auto-falls back to HF/ZIP if Kaggle fails. |
| **Hugging Face rate-limited** | Retry later, or set `HF_TOKEN` env var. System falls back to ZIP. |
| **Network firewall blocks downloads** | Use `--source zip` to try direct download, or manually download and place in `data/raw/`. |
| **Force a specific source** | `python -m scripts.fetch_dataset --source kaggle` (or `huggingface`, `zip`) |
| **Want to skip auto-download** | `python -m scripts.bootstrap_and_train --auto_download false` |
| **Partial download / corrupted files** | Delete `data/raw/` and re-run. Corrupted images are auto-removed. |

---

## Expected Duration

Approximate timings (varies by hardware and network speed):

| Step | CPU (i7/Ryzen 5) | GPU (RTX 3060) |
|------|------------------|----------------|
| Dataset download | 1-5 min | 1-5 min |
| Data preparation | ~10 sec | ~10 sec |
| Training (3 epochs, ~15 classes) | 5-15 min | 1-3 min |
| Training (20 epochs, ~15 classes) | 30-90 min | 5-15 min |
| Single prediction | <1 sec | <1 sec |

---

## Improving Accuracy

1. **More data** -- 200+ images per class significantly helps.
2. **More epochs** -- use 20+ epochs for serious training.
3. **Stronger backbone** -- swap `mobilenet_v3_small` for `efficientnet_b0` or `resnet50`.
4. **Advanced augmentations** -- add `RandomRotation`, `RandomAffine`, or use `albumentations`.
5. **Learning rate finder** -- sweep LR before training.
6. **Fine-tune in stages** -- freeze backbone first, then unfreeze.
7. **Mixup / CutMix** -- regularization for better generalization.
8. **Test-time augmentation** -- average over flipped/cropped copies.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run with `python -m` from the `produce-identifier/` root |
| `FileNotFoundError: Label map not found` | Train a model first |
| `FileNotFoundError: Split directory not found` | Run `python -m scripts.prepare_data` |
| `RuntimeError: Model not loaded` (API) | Train, then restart server or use /setup-demo |
| `CUDA out of memory` | Reduce `--batch_size` (e.g., 16 or 8) |
| `No image classes found` | Check that `data/raw/<class>/` folders contain images |
| Slow training on CPU | Expected -- reduce epochs for testing |
| `kagglehub` import error | `pip install kagglehub` |
| `datasets` import error | `pip install datasets huggingface-hub` |

---

## Verification

Run the pipeline verification script to check all components:

```bash
# Start the server in one terminal, then in another:
python -m scripts.verify_pipeline --api_url http://localhost:8000
```

It checks raw data, splits, model artifacts, and all API endpoints, printing a PASS/FAIL table.

---

*Built with PyTorch + FastAPI. Happy classifying!*
