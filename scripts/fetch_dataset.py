"""
Auto-download a public fruits & vegetables image dataset.

Strategy (tried in order):
  1. Kaggle via ``kagglehub``
  2. Hugging Face ``datasets`` library
  3. Direct ZIP download from a public URL

After download the images are normalised into::

    data/raw/<class_name>/...

Usage
-----
    python -m scripts.fetch_dataset [--raw_dir data/raw]
                                    [--min_samples 30] [--max_classes 25]
                                    [--source auto|kaggle|huggingface|zip]
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from src.config import (
    ALLOWED_EXTENSIONS,
    DATA_RAW_DIR,
    DEFAULT_MAX_CLASSES,
    DEFAULT_MIN_SAMPLES_PER_CLASS,
    DIRECT_ZIP_URL,
    DOWNLOAD_RETRY_COUNT,
    DOWNLOAD_TIMEOUT_SECONDS,
    HF_DATASET_NAME,
    KAGGLE_DATASET_SLUG,
    is_produce_class,
)


# ───────────────────────── helpers ─────────────────────────────
def _clean_name(name: str) -> str:
    """Lowercase, strip, replace spaces/hyphens with underscores."""
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    return name


def _is_valid_image(path: Path) -> bool:
    """Return True if *path* can be opened as an RGB image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _count_raw(raw_dir: Path, min_samples: int) -> Dict[str, int]:
    """Return {class_name: count} for classes meeting the threshold."""
    counts: Dict[str, int] = {}
    if not raw_dir.exists():
        return counts
    for d in sorted(raw_dir.iterdir()):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS)
        if n >= min_samples:
            counts[d.name] = n
    return counts


def raw_data_sufficient(raw_dir: Path, min_samples: int = 30, min_classes: int = 2) -> bool:
    """Check whether raw data already has enough classes and samples."""
    counts = _count_raw(raw_dir, min_samples)
    return len(counts) >= min_classes


# ───────────────────── source 1: Kaggle ────────────────────────
def _fetch_kaggle(raw_dir: Path) -> bool:
    """Download dataset via kagglehub. Returns True on success."""
    print("[FETCH] Trying Kaggle via kagglehub ...")
    try:
        import kagglehub  # type: ignore
    except ImportError:
        print("[FETCH]   kagglehub not installed – skipping Kaggle source.")
        return False

    try:
        path_str: str = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
        src = Path(path_str)
        print(f"[FETCH]   Downloaded to: {src}")
        _ingest_folder_tree(src, raw_dir)
        return True
    except Exception as exc:
        print(f"[FETCH]   Kaggle download failed: {exc}")
        return False


# ───────────────────── source 2: Hugging Face ──────────────────
def _fetch_huggingface(raw_dir: Path) -> bool:
    """Download dataset via Hugging Face ``datasets``. Returns True on success."""
    print("[FETCH] Trying Hugging Face datasets ...")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[FETCH]   `datasets` library not installed – skipping HF source.")
        return False

    try:
        ds = load_dataset(HF_DATASET_NAME, trust_remote_code=True)
        _ingest_hf_dataset(ds, raw_dir)
        return True
    except Exception as exc:
        print(f"[FETCH]   Hugging Face download failed: {exc}")
        return False


def _ingest_hf_dataset(ds: object, raw_dir: Path) -> None:
    """Save a Hugging Face DatasetDict into raw_dir/<class>/img_N.jpg."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    counter: Dict[str, int] = {}

    # The HF dataset may have splits (train, validation, test)
    splits = ds if isinstance(ds, dict) else {"default": ds}
    for split_name, split_ds in splits.items():
        print(f"[FETCH]   Processing HF split '{split_name}' ({len(split_ds)} samples) ...")
        # Detect column names – common patterns: image+label, image+labels, img+class
        cols = split_ds.column_names
        img_col = next((c for c in cols if c in ("image", "img", "pixel_values")), None)
        lbl_col = next((c for c in cols if c in ("label", "labels", "class", "class_name")), None)

        if img_col is None or lbl_col is None:
            print(f"[FETCH]   Could not detect image/label columns in {cols} – skipping.")
            continue

        # Resolve label names if ClassLabel feature
        label_names: Optional[List[str]] = None
        feat = split_ds.features.get(lbl_col)
        if hasattr(feat, "names"):
            label_names = feat.names

        for row in split_ds:
            img = row[img_col]
            raw_label = row[lbl_col]
            if label_names is not None and isinstance(raw_label, int):
                class_name = _clean_name(label_names[raw_label])
            else:
                class_name = _clean_name(str(raw_label))

            counter[class_name] = counter.get(class_name, 0) + 1
            cls_dir = raw_dir / class_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            dest = cls_dir / f"{class_name}_{counter[class_name]:05d}.jpg"
            if isinstance(img, Image.Image):
                img.convert("RGB").save(dest, "JPEG", quality=95)
            elif isinstance(img, (bytes, bytearray)):
                Image.open(io.BytesIO(img)).convert("RGB").save(dest, "JPEG", quality=95)
            else:
                # Try treating as file path
                try:
                    Image.open(img).convert("RGB").save(dest, "JPEG", quality=95)
                except Exception:
                    pass

    print(f"[FETCH]   Ingested {sum(counter.values())} images across {len(counter)} classes from HF.")


# ───────────────────── source 3: direct ZIP ────────────────────
def _fetch_zip(raw_dir: Path) -> bool:
    """Download a ZIP archive and extract. Returns True on success."""
    print(f"[FETCH] Trying direct ZIP download from:\n        {DIRECT_ZIP_URL}")
    try:
        import requests  # type: ignore
    except ImportError:
        print("[FETCH]   `requests` not installed – skipping ZIP source.")
        return False

    for attempt in range(1, DOWNLOAD_RETRY_COUNT + 1):
        try:
            print(f"[FETCH]   Attempt {attempt}/{DOWNLOAD_RETRY_COUNT} ...")
            resp = requests.get(DIRECT_ZIP_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS, stream=True)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            zip_path = raw_dir.parent / "_download_tmp.zip"
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
            downloaded = 0
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r[FETCH]   Downloading: {pct}%  ({downloaded // 1024 // 1024} MB)", end="", flush=True)
            print()

            tmp_dir = raw_dir.parent / "_zip_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)

            _ingest_folder_tree(tmp_dir, raw_dir)
            zip_path.unlink(missing_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return True

        except KeyboardInterrupt:
            print("\n[FETCH]   Download interrupted by user.")
            zip_path = raw_dir.parent / "_download_tmp.zip"
            zip_path.unlink(missing_ok=True)
            shutil.rmtree(raw_dir.parent / "_zip_tmp", ignore_errors=True)
            raise
        except Exception as exc:
            zip_path = raw_dir.parent / "_download_tmp.zip"
            zip_path.unlink(missing_ok=True)
            shutil.rmtree(raw_dir.parent / "_zip_tmp", ignore_errors=True)
            print(f"[FETCH]   ZIP download attempt {attempt} failed: {exc}")

    return False


# ──────────────── generic folder tree ingestion ────────────────
def _ingest_folder_tree(src_root: Path, raw_dir: Path) -> None:
    """Walk *src_root* and copy image files into raw_dir/<class_name>/."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    # Collect all directories that directly contain images (leaf class dirs)
    candidate_dirs: List[Path] = []
    for dirpath in sorted(src_root.rglob("*")):
        if not dirpath.is_dir():
            continue
        imgs = [f for f in dirpath.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
        if imgs:
            candidate_dirs.append(dirpath)

    for cls_dir in candidate_dirs:
        class_name = _clean_name(cls_dir.name)
        if not class_name:
            continue

        dest = raw_dir / class_name
        dest.mkdir(parents=True, exist_ok=True)

        existing = sum(1 for _ in dest.iterdir()) if dest.exists() else 0
        counter = existing

        for img_file in sorted(cls_dir.iterdir()):
            if not img_file.is_file() or img_file.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            counter += 1
            target = dest / f"{class_name}_{counter:05d}{img_file.suffix.lower()}"
            shutil.copy2(img_file, target)
            copied += 1

    print(f"[FETCH]   Ingested {copied} images into {raw_dir}")


# ──────────────── post-processing ──────────────────────────────
def _remove_corrupted(raw_dir: Path) -> int:
    """Delete files that cannot be decoded as images. Returns count removed."""
    removed = 0
    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        if not _is_valid_image(img_path):
            img_path.unlink(missing_ok=True)
            removed += 1
    return removed


def _filter_classes(
    raw_dir: Path,
    min_samples: int,
    max_classes: int,
) -> None:
    """Remove classes that are below threshold or non-produce, keep top N."""
    class_counts: Dict[str, int] = {}
    for d in sorted(raw_dir.iterdir()):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS)
        class_counts[d.name] = n

    # Remove non-produce and under-sampled
    to_remove: List[str] = []
    for name, count in class_counts.items():
        if not is_produce_class(name):
            print(f"[FETCH]   Removing non-produce class: {name} ({count} images)")
            to_remove.append(name)
        elif count < min_samples:
            print(f"[FETCH]   Removing under-sampled class: {name} ({count} < {min_samples})")
            to_remove.append(name)

    for name in to_remove:
        shutil.rmtree(raw_dir / name, ignore_errors=True)
        class_counts.pop(name, None)

    # Keep only top N by count
    if len(class_counts) > max_classes:
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        keep = {name for name, _ in sorted_classes[:max_classes]}
        for name in list(class_counts.keys()):
            if name not in keep:
                print(f"[FETCH]   Trimming extra class: {name} ({class_counts[name]} images)")
                shutil.rmtree(raw_dir / name, ignore_errors=True)
                class_counts.pop(name, None)


def print_summary(raw_dir: Path) -> None:
    """Print final class/sample counts."""
    total_images = 0
    total_classes = 0
    print(f"\n{'Class':<25s} {'Samples':>8s}  {'Category':<12s}")
    print("-" * 50)
    for d in sorted(raw_dir.iterdir()):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS)
        if n == 0:
            continue
        from src.config import get_category
        cat = get_category(d.name)
        print(f"  {d.name:<23s} {n:>8d}  {cat:<12s}")
        total_images += n
        total_classes += 1
    print("-" * 50)
    print(f"  {'TOTAL':<23s} {total_images:>8d}  ({total_classes} classes)")
    print()


# ───────────────────── main orchestrator ───────────────────────
def fetch_dataset(
    raw_dir: Optional[Path] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES_PER_CLASS,
    max_classes: int = DEFAULT_MAX_CLASSES,
    source: str = "auto",
) -> bool:
    """Download and prepare the dataset. Returns True on success.

    Parameters
    ----------
    raw_dir : Path
        Destination for raw class folders.
    min_samples : int
        Minimum images per class to keep.
    max_classes : int
        Maximum number of classes to keep (top-N by count).
    source : str
        "auto" tries all sources in order, or force "kaggle", "huggingface", "zip".
    """
    raw_dir = raw_dir or DATA_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Produce Identifier – Dataset Fetcher")
    print("=" * 60)

    # Already have data?
    if raw_data_sufficient(raw_dir, min_samples):
        print(f"[FETCH] Raw data already present at {raw_dir}")
        print_summary(raw_dir)
        return True

    # Try sources
    success = False
    sources = {
        "kaggle": _fetch_kaggle,
        "huggingface": _fetch_huggingface,
        "zip": _fetch_zip,
    }

    if source == "auto":
        for name, fn in sources.items():
            try:
                if fn(raw_dir):
                    success = True
                    print(f"[FETCH] Success via {name}!")
                    break
            except KeyboardInterrupt:
                print("\n[FETCH] Interrupted by user.")
                return False
            except Exception as exc:
                print(f"[FETCH] {name} failed with error: {exc}")
    else:
        fn = sources.get(source)
        if fn is None:
            print(f"[ERROR] Unknown source '{source}'. Use: auto, kaggle, huggingface, zip")
            return False
        try:
            success = fn(raw_dir)
        except KeyboardInterrupt:
            print("\n[FETCH] Interrupted by user.")
            return False

    if not success:
        print("\n" + "=" * 60)
        print("[ERROR] All automatic download sources failed.")
        print("        Please download a dataset manually and place images in:")
        print(f"          {raw_dir}/<class_name>/")
        print()
        print("  Option A – Kaggle CLI:")
        print(f"    kaggle datasets download -d {KAGGLE_DATASET_SLUG} -p data/")
        print("    # then unzip and move class folders into data/raw/")
        print()
        print("  Option B – Download from browser:")
        print(f"    https://www.kaggle.com/datasets/{KAGGLE_DATASET_SLUG}")
        print("=" * 60)
        return False

    # Post-process
    print("[FETCH] Removing corrupted images ...")
    n_removed = _remove_corrupted(raw_dir)
    if n_removed:
        print(f"[FETCH]   Removed {n_removed} corrupted files.")

    print("[FETCH] Filtering classes ...")
    _filter_classes(raw_dir, min_samples, max_classes)

    if not raw_data_sufficient(raw_dir, min_samples):
        print("[ERROR] After filtering, not enough classes remain.")
        return False

    print_summary(raw_dir)
    return True


# ───────────────── CLI ─────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-download fruits & vegetables dataset.")
    parser.add_argument("--raw_dir", type=str, default=str(DATA_RAW_DIR))
    parser.add_argument("--min_samples", type=int, default=DEFAULT_MIN_SAMPLES_PER_CLASS)
    parser.add_argument("--max_classes", type=int, default=DEFAULT_MAX_CLASSES)
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "kaggle", "huggingface", "zip"],
        help="Force a specific download source or 'auto' to try all.",
    )
    args = parser.parse_args()

    ok = fetch_dataset(
        raw_dir=Path(args.raw_dir),
        min_samples=args.min_samples,
        max_classes=args.max_classes,
        source=args.source,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
