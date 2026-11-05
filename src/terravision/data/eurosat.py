"""EuroSAT RGB: download, integrity checks, stratified split, normalisation stats.

Source (verified 2026-11-05): Zenodo record 7711810, v2, DOI 10.5281/zenodo.7711810,
MIT license. See docs/dataset_documentation.md.

Typical use on Colab (one call does everything and is safe to re-run):

    from terravision.data.eurosat import prepare_eurosat
    report = prepare_eurosat(raw_dir=..., data_dir=..., processed_dir=...)

Nothing here assumes the dataset's content: class names, counts, image sizes and
colour modes are *measured* and written to a JSON report, then compared with
what the paper describes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from terravision.data.download import download_verified, extract_archive, extract_once, file_hash  # noqa: F401 (extract_archive re-exported)

# --- Official source -------------------------------------------------------
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
EUROSAT_ARCHIVE = "EuroSAT_RGB.zip"
EUROSAT_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"  # published on the Zenodo record
EUROSAT_SIZE_BYTES = 94_658_721  # from the server's Content-Length header

# Class names described by the authors. Used only to *compare* with the folders
# actually found; the real list always comes from the extracted data.
EXPECTED_CLASSES = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_SEED = 42
# Train / validation / test fractions (spec §2.1).
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


# --- Download and extraction ----------------------------------------------
def file_md5(path: Path) -> str:
    return file_hash(path, "md5")


def download_eurosat(raw_dir: Path, url: str = EUROSAT_URL) -> Path:
    """Download the archive into ``raw_dir`` unless a verified copy exists."""
    return download_verified(url, Path(raw_dir) / EUROSAT_ARCHIVE, EUROSAT_MD5, "md5")


def find_class_root(base: Path) -> Path:
    """Return the folder whose sub-folders are the classes.

    The archive's internal layout isn't documented, so search for the
    shallowest folder whose children are directories containing images.
    """
    base = Path(base)
    candidates = [base, *sorted(p for p in base.rglob("*") if p.is_dir())]
    for folder in candidates:
        subdirs = [d for d in folder.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))]
        if len(subdirs) >= 2 and all(_has_images(d) for d in subdirs):
            return folder
    raise FileNotFoundError(f"No class folders with images found under {base}")


def _has_images(folder: Path) -> bool:
    return any(p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir() if p.is_file())


# --- Inspection ------------------------------------------------------------
@dataclass
class ScanResult:
    index: pd.DataFrame          # one row per readable image: path, class_name, label
    classes: list[str]           # sorted class names; label = position in this list
    corrupt: list[str]           # files that failed to decode
    sizes: dict[str, int]        # "WxH" -> count
    modes: dict[str, int]        # PIL mode -> count
    duplicate_groups: list[list[str]]  # byte-identical files


def scan_dataset(class_root: Path) -> ScanResult:
    """Open every image once: record class, size and mode; flag bad files."""
    class_root = Path(class_root)
    classes = sorted(d.name for d in class_root.iterdir() if d.is_dir() and not d.name.startswith((".", "__")))

    rows, corrupt = [], []
    sizes, modes = Counter(), Counter()
    by_hash: dict[str, list[str]] = {}

    files = [
        (p, name)
        for name in classes
        for p in sorted((class_root / name).iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for path, name in tqdm(files, desc="Scanning images"):
        rel = path.relative_to(class_root).as_posix()
        data = path.read_bytes()
        by_hash.setdefault(hashlib.md5(data).hexdigest(), []).append(rel)
        try:
            with Image.open(path) as img:
                img.load()  # full decode, not just the header
                sizes[f"{img.width}x{img.height}"] += 1
                modes[img.mode] += 1
        except Exception as exc:  # any decode failure means the file is unusable
            corrupt.append(f"{rel}: {type(exc).__name__}: {exc}")
            continue
        rows.append({"path": rel, "class_name": name, "label": classes.index(name)})

    return ScanResult(
        index=pd.DataFrame(rows, columns=["path", "class_name", "label"]),
        classes=classes,
        corrupt=corrupt,
        sizes=dict(sizes),
        modes=dict(modes),
        duplicate_groups=[g for g in by_hash.values() if len(g) > 1],
    )


# --- Split and statistics --------------------------------------------------
def make_splits(index: pd.DataFrame, seed: int = DEFAULT_SEED,
                fractions: tuple[float, float, float] = SPLIT_FRACTIONS) -> pd.DataFrame:
    """Stratified train/val/test split. Returns ``index`` plus a ``split`` column.

    Two stratified cuts: first test off the full set, then validation off the
    remainder (rescaled so the final proportions match ``fractions``).
    """
    train_f, val_f, test_f = fractions
    if not np.isclose(train_f + val_f + test_f, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")

    index = index.sort_values("path").reset_index(drop=True)  # order-independent
    rest, test = train_test_split(index, test_size=test_f, stratify=index["label"], random_state=seed)
    train, val = train_test_split(rest, test_size=val_f / (train_f + val_f),
                                  stratify=rest["label"], random_state=seed)
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    return out.sort_values("path").reset_index(drop=True)


def compute_mean_std(class_root: Path, paths: list[str]) -> tuple[list[float], list[float]]:
    """Per-channel mean and std of RGB pixels scaled to [0, 1].

    Call it with **training** paths only, so no validation/test information
    leaks into preprocessing. Uses float64 running sums (exact, low memory).
    """
    total = np.zeros(3)
    total_sq = np.zeros(3)
    n_pixels = 0
    for rel in tqdm(paths, desc="Normalisation stats"):
        with Image.open(Path(class_root) / rel) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        pixels = arr.reshape(-1, 3)
        total += pixels.sum(axis=0)
        total_sq += (pixels ** 2).sum(axis=0)
        n_pixels += pixels.shape[0]
    mean = total / n_pixels
    std = np.sqrt(total_sq / n_pixels - mean ** 2)
    return mean.round(6).tolist(), std.round(6).tolist()


# --- One-call pipeline -----------------------------------------------------
def prepare_eurosat(raw_dir: Path, data_dir: Path, processed_dir: Path,
                    seed: int = DEFAULT_SEED) -> dict:
    """Download → verify → extract → scan → split → stats → report.

    raw_dir:        persistent folder for the original zip (Drive on Colab)
    data_dir:       where images are extracted (fast local disk on Colab)
    processed_dir:  persistent folder for the split CSV and report JSON

    Corrupt files are excluded from the split and listed in the report; they
    are never silently fixed or deleted.
    """
    archive = download_eurosat(raw_dir)

    extract_dir = extract_once(archive, Path(data_dir) / "eurosat")
    class_root = find_class_root(extract_dir)

    scan = scan_dataset(class_root)
    split_df = make_splits(scan.index, seed=seed)
    mean, std = compute_mean_std(class_root, split_df.loc[split_df["split"] == "train", "path"].tolist())

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_csv = processed_dir / f"eurosat_split_seed{seed}.csv"
    split_df.to_csv(split_csv, index=False)

    report = {
        "source": {"url": EUROSAT_URL, "archive": EUROSAT_ARCHIVE, "md5": EUROSAT_MD5,
                   "zenodo_doi": "10.5281/zenodo.7711810", "license": "MIT"},
        "class_root_in_archive": class_root.relative_to(extract_dir).as_posix(),
        "classes": scan.classes,
        "classes_match_paper": sorted(scan.classes) == sorted(EXPECTED_CLASSES),
        "n_files_scanned": len(scan.index) + len(scan.corrupt),
        "n_readable": len(scan.index),
        "corrupt_files": scan.corrupt,
        "image_sizes": scan.sizes,
        "image_modes": scan.modes,
        "duplicate_groups": scan.duplicate_groups,
        "class_counts": scan.index["class_name"].value_counts().sort_index().to_dict(),
        "split": {
            "seed": seed,
            "fractions": dict(zip(("train", "val", "test"), SPLIT_FRACTIONS)),
            "counts": split_df["split"].value_counts().to_dict(),
            "counts_per_class": split_df.pivot_table(index="class_name", columns="split",
                                                     values="path", aggfunc="count").to_dict(orient="index"),
            "csv": split_csv.name,
        },
        "normalization": {"computed_on": "train", "scale": "[0,1]", "mean": mean, "std": std},
    }
    report_path = processed_dir / "eurosat_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {split_csv} and {report_path}")
    return report


def copy_sample(class_root: Path, split_csv: Path, out_dir: Path, per_class: int = 10) -> Path:
    """Copy a few test-split images per class (plus a matching split CSV) to
    ``out_dir``, for running the loaders on real images on the local CPU."""
    df = pd.read_csv(split_csv)
    sample = df[df["split"] == "test"].groupby("class_name", group_keys=False).head(per_class)
    out_dir = Path(out_dir)
    for rel in sample["path"]:
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(class_root) / rel, out_dir / rel)
    sample.to_csv(out_dir / "sample_split.csv", index=False)
    return out_dir
