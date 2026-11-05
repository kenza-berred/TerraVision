"""LandCover.ai inspection tests on a small synthetic archive (no download)."""

import zipfile

import numpy as np
import pytest
from PIL import Image

from terravision.data import download, landcoverai


@pytest.fixture
def fake_archive(tmp_path):
    """images/*.tif (RGB) + masks/*.tif (1-band), split files, a script,
    one image without a mask and one pair whose sizes disagree."""
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "masks").mkdir()
    rng = np.random.default_rng(0)

    def save_pair(name, h, w, mask_hw=None):
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(src / "images" / f"{name}.tif")
        mh, mw = mask_hw or (h, w)
        mask = np.zeros((mh, mw), dtype=np.uint8)
        mask[: mh // 2, : mw // 2] = 2
        mask[0, 0] = 4
        Image.fromarray(mask).save(src / "masks" / f"{name}.tif")

    save_pair("A", 40, 30)
    save_pair("B", 20, 20)
    save_pair("C", 20, 20, mask_hw=(10, 10))
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(src / "images" / "orphan.tif")
    (src / "train.txt").write_text("A_0\nA_1\nB_0\n")
    (src / "split.py").write_text("print('tiling script')\n")

    archive = tmp_path / "raw" / landcoverai.LANDCOVERAI_ARCHIVE
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src).as_posix())
    return archive


@pytest.fixture
def report(fake_archive, tmp_path, monkeypatch):
    monkeypatch.setattr(landcoverai, "LANDCOVERAI_SHA256", download.file_hash(fake_archive, "sha256"))
    return landcoverai.inspect_landcoverai(fake_archive.parent, tmp_path / "data", tmp_path / "processed")


def test_layout_and_text_files(report):
    assert report["layout"]["images"] == {".tif": 4}
    assert report["layout"]["masks"] == {".tif": 3}
    assert report["text_files"]["train.txt"]["n_lines"] == 3
    assert "tiling script" in report["text_files"]["split.py"]["full_text"]


def test_pairing_and_problems_are_reported(report):
    assert report["n_images"] == 4 and report["n_masks"] == 3 and report["n_pairs"] == 3
    assert report["images_without_mask"] == ["orphan"]
    assert report["size_mismatch"] == ["C"]
    assert report["unreadable"] == []


def test_mask_values_are_measured(report):
    values = report["mask_values"]
    assert set(values) == {"0", "2", "4"}
    assert values["2"]["documented_as"] == "woodland"
    assert sum(v["pixels"] for v in values.values()) == 40 * 30 + 20 * 20 + 10 * 10
    assert report["image_dtypes"] == {"3-band uint8": 4}
    assert report["mask_dtypes"] == {"uint8": 3}


def test_bad_checksum_is_rejected(fake_archive, monkeypatch):
    monkeypatch.setattr(landcoverai, "LANDCOVERAI_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        landcoverai.download_landcoverai(fake_archive.parent)
