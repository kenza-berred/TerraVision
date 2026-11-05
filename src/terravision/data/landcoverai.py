"""LandCover.ai v1: download and inspection.

Source (checked 2026-11-05): https://landcover.ai.linuxpolska.com/ — v1 archive,
CC BY-NC-SA 4.0. See docs/dataset_documentation.md.

This module deliberately *measures* the dataset before any preprocessing code is
written: archive layout, which rasters are images vs. masks, image/mask pairing,
sizes, colour modes, and the actual values present in the masks. Tiling and the
Dataset class are written only after this report has been reviewed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from terravision.data.download import download_verified, extract_once

LANDCOVERAI_URL = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
LANDCOVERAI_ARCHIVE = "landcover.ai.v1.zip"
LANDCOVERAI_SIZE_BYTES = 1_538_212_277  # server Content-Length, 2026-11-05
# The official site publishes no checksum. This SHA-256 is the one TorchGeo
# (microsoft/torchgeo, torchgeo/datasets/landcoverai.py) pins for the same URL:
# an independent check that we get the same file everyone else uses.
LANDCOVERAI_SHA256 = "dd448287ff3a76291f2b159084b8a265f2ba4559bb80b0551fc8c6c0eb149a4b"

# What the official site documents for v1. Used only to *compare* with the
# values measured in the masks, never as ground truth.
DOCUMENTED_MASK_VALUES = {1: "building", 2: "woodland", 3: "water", 4: "road"}

RASTER_EXTENSIONS = {".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".py", ".md", ".csv", ".json"}
TEXT_PREVIEW_LINES = 10
FULL_TEXT_MAX_CHARS = 20_000  # small scripts (e.g. split.py) are returned in full

# Orthophotos are ~9000x9500 px (~86 Mpx). That is just under PIL's default
# decompression-bomb limit (~89 Mpx); raise it for these known, trusted files.
Image.MAX_IMAGE_PIXELS = 200_000_000


def download_landcoverai(raw_dir: Path) -> Path:
    return download_verified(LANDCOVERAI_URL, Path(raw_dir) / LANDCOVERAI_ARCHIVE,
                             LANDCOVERAI_SHA256, "sha256")


def read_raster(path: Path) -> np.ndarray:
    """Read a raster as (H, W) or (H, W, C). PIL first, OpenCV as fallback
    (some compressed GeoTIFF variants are only readable by one of them)."""
    try:
        with Image.open(path) as img:
            return np.asarray(img)
    except Exception:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise
        if arr.ndim == 3:
            arr = arr[..., ::-1]  # OpenCV is BGR; return RGB like PIL
        return arr


def _summarise_text(path: Path) -> dict:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    info: dict = {"n_lines": len(lines), "head": lines[:TEXT_PREVIEW_LINES]}
    if path.suffix == ".py" and len(text) <= FULL_TEXT_MAX_CHARS:
        info["full_text"] = text
    return info


def inspect_landcoverai(raw_dir: Path, data_dir: Path, processed_dir: Path) -> dict:
    """Download (if needed), extract, and measure the dataset. Writes
    ``landcoverai_inspection.json`` to ``processed_dir`` and returns it."""
    archive = download_landcoverai(raw_dir)
    root = extract_once(archive, Path(data_dir) / "landcoverai")
    files = sorted(p for p in root.rglob("*") if p.is_file())

    # 1. Layout: file count per folder and extension.
    layout: dict[str, Counter] = defaultdict(Counter)
    for p in files:
        layout[p.parent.relative_to(root).as_posix() or "."][p.suffix.lower() or "(none)"] += 1

    # 2. Text files and scripts (read, never executed).
    text_files = {p.relative_to(root).as_posix(): _summarise_text(p)
                  for p in files if p.suffix.lower() in TEXT_EXTENSIONS}

    # 3. Rasters: full decode; 1-band rasters are treated as masks.
    rasters = [p for p in files if p.suffix.lower() in RASTER_EXTENSIONS]
    images: dict[str, dict] = {}
    masks: dict[str, dict] = {}
    mask_value_counts: Counter = Counter()
    unreadable: list[str] = []
    for p in tqdm(rasters, desc="Reading rasters"):
        rel = p.relative_to(root).as_posix()
        try:
            arr = read_raster(p)
        except Exception as exc:
            unreadable.append(f"{rel}: {type(exc).__name__}: {exc}")
            continue
        info = {"path": rel, "height": arr.shape[0], "width": arr.shape[1],
                "bands": 1 if arr.ndim == 2 else arr.shape[2], "dtype": str(arr.dtype)}
        if info["bands"] == 1:
            values, counts = np.unique(arr, return_counts=True)
            info["values"] = {int(v): int(c) for v, c in zip(values, counts)}
            mask_value_counts.update(info["values"])
            masks[p.stem] = info
        else:
            images[p.stem] = info
        del arr

    # 4. Pairing by file stem, and size agreement.
    paired = sorted(set(images) & set(masks))
    size_mismatch = [s for s in paired
                     if (images[s]["height"], images[s]["width"]) != (masks[s]["height"], masks[s]["width"])]
    total_mask_px = sum(mask_value_counts.values())

    report = {
        "source": {"url": LANDCOVERAI_URL, "archive": LANDCOVERAI_ARCHIVE,
                   "sha256": LANDCOVERAI_SHA256, "license": "CC BY-NC-SA 4.0"},
        "layout": {k: dict(v) for k, v in sorted(layout.items())},
        "text_files": text_files,
        "n_images": len(images),
        "n_masks": len(masks),
        "n_pairs": len(paired),
        "images_without_mask": sorted(set(images) - set(masks)),
        "masks_without_image": sorted(set(masks) - set(images)),
        "size_mismatch": size_mismatch,
        "unreadable": unreadable,
        "image_sizes": dict(Counter(f"{images[s]['width']}x{images[s]['height']}" for s in images)),
        "image_dtypes": dict(Counter(f"{i['bands']}-band {i['dtype']}" for i in images.values())),
        "mask_dtypes": dict(Counter(m["dtype"] for m in masks.values())),
        "mask_values": {
            str(v): {"pixels": c, "fraction": round(c / total_mask_px, 6),
                     "documented_as": DOCUMENTED_MASK_VALUES.get(v, "(not documented)")}
            for v, c in sorted(mask_value_counts.items())
        },
        "rasters": {"images": list(images.values()), "masks": list(masks.values())},
    }
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "landcoverai_inspection.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return report
