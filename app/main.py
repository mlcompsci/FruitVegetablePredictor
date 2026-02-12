"""
FastAPI application for the Produce Identifier.

Endpoints
---------
GET  /           – Serve the web UI
POST /predict    – Upload an image and get classification results
GET  /health     – Health-check
GET  /status     – System readiness and guidance
POST /setup-demo – Run full bootstrap pipeline (download + train)
"""

from __future__ import annotations

import asyncio
import io
import sys
import threading
import time
from contextlib import asynccontextmanager, redirect_stdout
from pathlib import Path
from typing import AsyncGenerator, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import classifier
from app.schemas import (
    HealthResponse,
    PredictionItem,
    PredictionResponse,
    PredictionSummary,
    SetupDemoResponse,
    SetupLogEntry,
    StatusResponse,
)
from src.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_UPLOAD_TYPES,
    BEST_MODEL_PATH,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
)

# ───────────────── paths ───────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ───────────────── setup lock ──────────────────────────────────
_setup_lock = threading.Lock()
_setup_running = False


# ───────────────── helpers ─────────────────────────────────────
def _raw_data_present() -> bool:
    if not DATA_RAW_DIR.exists():
        return False
    for d in DATA_RAW_DIR.iterdir():
        if d.is_dir():
            imgs = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
            if imgs:
                return True
    return False


def _processed_data_present() -> bool:
    return all((DATA_PROCESSED_DIR / s).exists() for s in ("train", "val", "test"))


def _suggested_command() -> str | None:
    if not _raw_data_present():
        return "python -m scripts.bootstrap_and_train --auto_download true --epochs 3"
    if not _processed_data_present():
        return "python -m scripts.prepare_data"
    if not BEST_MODEL_PATH.exists():
        return "python -m src.train --data_dir data/processed --epochs 3"
    if not classifier.is_loaded:
        return "Restart the server to load the trained model."
    return None


# ───────────────── lifespan (startup / shutdown) ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model at startup if available."""
    try:
        classifier.load()
        print("[INFO] Classifier ready.")
    except FileNotFoundError as exc:
        print(f"[WARNING] {exc}")
        print("[WARNING] The /predict endpoint will return 503 until a model is trained.")
    yield


# ───────────────── app factory ─────────────────────────────────
app = FastAPI(
    title="Produce Identifier API",
    description="Classify fruits and vegetables from images.",
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ───────────────── routes ──────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the main web UI."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html template not found.")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Simple health check."""
    return HealthResponse(status="ok")


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """System readiness check with guidance."""
    num_classes = len(classifier.idx_to_class) if classifier.is_loaded else None
    return StatusResponse(
        model_loaded=classifier.is_loaded,
        raw_data_present=_raw_data_present(),
        processed_data_present=_processed_data_present(),
        num_classes=num_classes,
        suggested_next_command=_suggested_command(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Accept an uploaded image and return top-k predictions with category info."""
    # Validate model is loaded
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Train a model first using:\n"
                "  python -m scripts.bootstrap_and_train --auto_download true --epochs 3\n"
                "Then restart the server, or use the 'Auto Setup & Train' button in the UI."
            ),
        )

    # Validate content type
    if file.content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_UPLOAD_TYPES))}",
        )

    # Read and validate size
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(image_bytes) / 1024 / 1024:.1f} MB). "
            f"Max allowed: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB.",
        )

    # Run inference
    try:
        top_k_results, summary = classifier.predict(image_bytes, top_k=3)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        top_k=[PredictionItem(**item) for item in top_k_results],
        summary=PredictionSummary(**summary),
    )


@app.post("/setup-demo", response_model=SetupDemoResponse)
async def setup_demo() -> SetupDemoResponse:
    """Run the full bootstrap pipeline: download data, prepare splits, train.

    Uses a lock to prevent concurrent runs. Runs synchronously in a thread
    to avoid blocking the event loop.
    """
    global _setup_running

    if _setup_running:
        return SetupDemoResponse(
            success=False,
            logs=[SetupLogEntry(step="lock", message="Setup is already running. Please wait.", success=False)],
            message="A setup job is already in progress.",
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_setup_sync)
    return result


def _run_setup_sync() -> SetupDemoResponse:
    """Synchronous bootstrap runner (called in a thread)."""
    global _setup_running

    if not _setup_lock.acquire(blocking=False):
        return SetupDemoResponse(
            success=False,
            logs=[SetupLogEntry(step="lock", message="Could not acquire lock.", success=False)],
            message="A setup job is already in progress.",
        )

    _setup_running = True
    logs: List[SetupLogEntry] = []
    start = time.time()

    try:
        # Capture stdout
        log_buf = io.StringIO()

        # Step 1: Fetch dataset
        logs.append(SetupLogEntry(step="fetch", message="Checking / downloading dataset ...", success=True))
        try:
            from scripts.fetch_dataset import fetch_dataset, raw_data_sufficient

            if raw_data_sufficient(DATA_RAW_DIR, min_samples=10):
                logs.append(SetupLogEntry(step="fetch", message="Raw data already present.", success=True))
            else:
                with redirect_stdout(log_buf):
                    ok = fetch_dataset(raw_dir=DATA_RAW_DIR, min_samples=10, max_classes=25)
                if ok:
                    logs.append(SetupLogEntry(step="fetch", message="Dataset downloaded successfully.", success=True))
                else:
                    logs.append(SetupLogEntry(step="fetch", message="Dataset download failed. See server logs.", success=False))
                    return SetupDemoResponse(
                        success=False, logs=logs,
                        message="Failed to download dataset. Try manually or check network.",
                        duration_seconds=round(time.time() - start, 1),
                    )
        except Exception as exc:
            logs.append(SetupLogEntry(step="fetch", message=f"Error: {exc}", success=False))
            return SetupDemoResponse(
                success=False, logs=logs, message=str(exc),
                duration_seconds=round(time.time() - start, 1),
            )

        # Step 2: Prepare splits
        logs.append(SetupLogEntry(step="prepare", message="Preparing train/val/test splits ...", success=True))
        try:
            from scripts.prepare_data import copy_split, gather_class_images, stratified_split

            class_images = gather_class_images(DATA_RAW_DIR)
            if not class_images:
                logs.append(SetupLogEntry(step="prepare", message="No classes found in raw data.", success=False))
                return SetupDemoResponse(
                    success=False, logs=logs, message="No image classes in raw data.",
                    duration_seconds=round(time.time() - start, 1),
                )
            with redirect_stdout(log_buf):
                splits = stratified_split(class_images)
                copy_split(splits, DATA_PROCESSED_DIR)
            logs.append(SetupLogEntry(
                step="prepare",
                message=f"Splits created: {len(class_images)} classes.",
                success=True,
            ))
        except Exception as exc:
            logs.append(SetupLogEntry(step="prepare", message=f"Error: {exc}", success=False))
            return SetupDemoResponse(
                success=False, logs=logs, message=str(exc),
                duration_seconds=round(time.time() - start, 1),
            )

        # Step 3: Train (short demo)
        logs.append(SetupLogEntry(step="train", message="Training model (3 epochs) ...", success=True))
        try:
            from src.train import train

            with redirect_stdout(log_buf):
                train(
                    data_dir=str(DATA_PROCESSED_DIR),
                    epochs=3,
                    batch_size=32,
                    lr=1e-3,
                    num_workers=0,  # safer for thread context
                    seed=42,
                )

            if BEST_MODEL_PATH.exists():
                logs.append(SetupLogEntry(step="train", message="Training complete. Model saved.", success=True))
            else:
                logs.append(SetupLogEntry(step="train", message="Training ran but model file missing.", success=False))
                return SetupDemoResponse(
                    success=False, logs=logs, message="Model not saved.",
                    duration_seconds=round(time.time() - start, 1),
                )
        except Exception as exc:
            logs.append(SetupLogEntry(step="train", message=f"Error: {exc}", success=False))
            return SetupDemoResponse(
                success=False, logs=logs, message=str(exc),
                duration_seconds=round(time.time() - start, 1),
            )

        # Step 4: Reload model in the classifier singleton
        logs.append(SetupLogEntry(step="reload", message="Loading model into memory ...", success=True))
        try:
            classifier._loaded = False  # force reload
            classifier.load()
            logs.append(SetupLogEntry(
                step="reload",
                message=f"Model loaded with {len(classifier.idx_to_class)} classes.",
                success=True,
            ))
        except Exception as exc:
            logs.append(SetupLogEntry(step="reload", message=f"Error: {exc}", success=False))

        elapsed = round(time.time() - start, 1)
        return SetupDemoResponse(
            success=True,
            logs=logs,
            message=f"Setup complete in {elapsed}s. You can now upload images to classify!",
            duration_seconds=elapsed,
        )

    except KeyboardInterrupt:
        logs.append(SetupLogEntry(step="interrupt", message="Interrupted by user.", success=False))
        return SetupDemoResponse(
            success=False, logs=logs, message="Interrupted.",
            duration_seconds=round(time.time() - start, 1),
        )
    finally:
        _setup_running = False
        _setup_lock.release()
