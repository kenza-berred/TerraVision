"""Checksum-verified downloads and safe archive extraction (shared by datasets)."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm


def file_hash(path: Path, algorithm: str = "md5", chunk_size: int = 1 << 20) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, dest: Path, checksum: str, algorithm: str = "md5") -> Path:
    """Download ``url`` to ``dest`` unless a verified copy already exists.

    Writes to ``<dest>.part`` and renames only after the checksum matches, so
    an interrupted or corrupted download never looks complete.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if file_hash(dest, algorithm) == checksum:
            print(f"Archive already present and verified: {dest}")
            return dest
        raise ValueError(
            f"{dest} exists but its {algorithm.upper()} does not match the published one. "
            "Delete it manually and re-run to download a fresh copy."
        )

    partial = dest.with_name(dest.name + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
                bar.update(len(chunk))

    actual = file_hash(partial, algorithm)
    if actual != checksum:
        partial.unlink()
        raise ValueError(f"{algorithm.upper()} mismatch: got {actual}, expected {checksum}. Download discarded.")
    partial.rename(dest)
    print(f"Downloaded and verified: {dest}")
    return dest


def extract_archive(archive: Path, out_dir: Path) -> Path:
    """Unzip ``archive`` into ``out_dir``, refusing entries that escape it."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (out_dir / member).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise ValueError(f"Unsafe path in archive: {member}")
        zf.extractall(out_dir)
    return out_dir


def extract_once(archive: Path, extract_dir: Path) -> Path:
    """Extract unless ``extract_dir`` already exists.

    Extracts to ``<extract_dir>.tmp`` first and renames at the end, so an
    interrupted extraction never looks complete (the .tmp is cleared next time).
    """
    extract_dir = Path(extract_dir)
    if extract_dir.exists():
        return extract_dir
    tmp_dir = extract_dir.with_name(extract_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Extracting to {extract_dir}")
    extract_archive(archive, tmp_dir)
    tmp_dir.rename(extract_dir)
    return extract_dir
