"""
End-to-end verification of the Produce Identifier pipeline.

Checks:
  1. Raw data exists (or fetch works)
  2. Processed splits exist
  3. Model artifacts exist after training
  4. /health and /status endpoints respond
  5. /predict works on a sample image

Usage
-----
    python -m scripts.verify_pipeline [--api_url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import List, Tuple

from src.config import (
    ALLOWED_EXTENSIONS,
    BEST_MODEL_PATH,
    CLASS_METADATA_PATH,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    LABEL_MAP_PATH,
)


def _check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def check_raw_data() -> bool:
    """Check that raw data directory has at least 2 class folders."""
    if not DATA_RAW_DIR.exists():
        return _check("Raw data exists", False, f"{DATA_RAW_DIR} not found")
    classes = [d for d in DATA_RAW_DIR.iterdir() if d.is_dir()]
    n_images = sum(
        1 for d in classes
        for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )
    return _check("Raw data exists", len(classes) >= 2,
                   f"{len(classes)} classes, {n_images} images")


def check_processed_data() -> bool:
    """Check that processed train/val/test splits exist."""
    ok = all((DATA_PROCESSED_DIR / s).exists() for s in ("train", "val", "test"))
    return _check("Processed splits exist", ok, str(DATA_PROCESSED_DIR))


def check_model_artifacts() -> bool:
    """Check that all model files are present."""
    files = [BEST_MODEL_PATH, LABEL_MAP_PATH, CLASS_METADATA_PATH]
    missing = [f.name for f in files if not f.exists()]
    ok = len(missing) == 0
    detail = "all present" if ok else f"missing: {', '.join(missing)}"
    return _check("Model artifacts exist", ok, detail)


def check_health(api_url: str) -> bool:
    """Ping GET /health."""
    try:
        import requests
        resp = requests.get(f"{api_url}/health", timeout=5)
        ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        return _check("GET /health", ok, f"status={resp.status_code}")
    except Exception as exc:
        return _check("GET /health", False, str(exc))


def check_status(api_url: str) -> bool:
    """Ping GET /status."""
    try:
        import requests
        resp = requests.get(f"{api_url}/status", timeout=5)
        ok = resp.status_code == 200
        data = resp.json()
        return _check("GET /status", ok,
                       f"model_loaded={data.get('model_loaded')}")
    except Exception as exc:
        return _check("GET /status", False, str(exc))


def check_predict(api_url: str) -> bool:
    """POST /predict with a tiny synthetic JPEG."""
    try:
        import requests
        from PIL import Image

        # Create a tiny test image
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = requests.post(
            f"{api_url}/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            timeout=30,
        )
        if resp.status_code == 503:
            return _check("POST /predict", False, "503 – model not loaded")
        ok = resp.status_code == 200 and "top_k" in resp.json()
        detail = f"status={resp.status_code}"
        if ok:
            top = resp.json()["top_k"][0]
            detail += f", top1={top['label']} ({top['probability']:.2%})"
        return _check("POST /predict", ok, detail)
    except Exception as exc:
        return _check("POST /predict", False, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Produce Identifier pipeline.")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000",
                        help="Base URL of the running FastAPI server.")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Produce Identifier – Pipeline Verification")
    print("=" * 60)
    print()

    results: List[Tuple[str, bool]] = []

    # Offline checks
    results.append(("Raw data", check_raw_data()))
    results.append(("Processed splits", check_processed_data()))
    results.append(("Model artifacts", check_model_artifacts()))

    # Online checks (require running server)
    print()
    print("  API checks (server must be running):")
    results.append(("Health endpoint", check_health(args.api_url)))
    results.append(("Status endpoint", check_status(args.api_url)))
    results.append(("Predict endpoint", check_predict(args.api_url)))

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print()
    print("-" * 60)
    print(f"  Results: {passed}/{total} checks passed")
    if passed == total:
        print("  All checks PASSED!")
    else:
        failed = [name for name, ok in results if not ok]
        print(f"  Failed: {', '.join(failed)}")
    print("-" * 60)
    print()

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
