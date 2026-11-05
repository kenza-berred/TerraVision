"""EuroSAT pipeline tests on a small synthetic archive (no download, CPU only)."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from terravision.data import eurosat  # noqa: E402
from terravision.data.classification_dataset import EuroSATDataset  # noqa: E402
from terravision.data.preprocessing import RandomRot90, classification_transforms  # noqa: E402

CLASSES = list(eurosat.EXPECTED_CLASSES)
PER_CLASS = 20


@pytest.fixture
def fake_archive(tmp_path) -> Path:
    """Zip laid out as EuroSAT_RGB/<class>/<class>_<i>.jpg, plus one corrupt
    file and one byte-identical duplicate."""
    src = tmp_path / "src" / "EuroSAT_RGB"
    rng = np.random.default_rng(0)
    for name in CLASSES:
        (src / name).mkdir(parents=True)
        for i in range(PER_CLASS):
            arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(src / name / f"{name}_{i}.jpg", quality=95)
    (src / "Forest" / "Forest_broken.jpg").write_bytes(b"not a jpeg")
    (src / "River" / "River_copy.jpg").write_bytes((src / "River" / "River_0.jpg").read_bytes())

    archive = tmp_path / "raw" / eurosat.EUROSAT_ARCHIVE
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        for p in src.rglob("*.jpg"):
            zf.write(p, p.relative_to(src.parent).as_posix())
    return archive


@pytest.fixture
def prepared(fake_archive, tmp_path, monkeypatch):
    # Treat the fake archive as the "official" one.
    monkeypatch.setattr(eurosat, "EUROSAT_MD5", eurosat.file_md5(fake_archive))
    report = eurosat.prepare_eurosat(
        raw_dir=fake_archive.parent, data_dir=tmp_path / "data", processed_dir=tmp_path / "processed")
    class_root = tmp_path / "data" / "eurosat" / report["class_root_in_archive"]
    split_csv = tmp_path / "processed" / "eurosat_split_seed42.csv"
    return report, class_root, split_csv


def test_report_measures_dataset(prepared):
    report, _, _ = prepared
    assert report["classes"] == sorted(CLASSES)
    assert report["classes_match_paper"]
    assert report["class_root_in_archive"] == "EuroSAT_RGB"
    assert report["n_files_scanned"] == 10 * PER_CLASS + 2
    assert len(report["corrupt_files"]) == 1 and "Forest_broken" in report["corrupt_files"][0]
    assert report["image_sizes"] == {"64x64": 10 * PER_CLASS + 1}
    assert report["duplicate_groups"] == [["River/River_0.jpg", "River/River_copy.jpg"]]
    assert len(report["normalization"]["mean"]) == 3


def test_split_is_stratified_disjoint_and_complete(prepared):
    report, _, split_csv = prepared
    df = pd.read_csv(split_csv)
    assert df["path"].is_unique
    assert len(df) == report["n_readable"]
    assert not df["path"].str.contains("broken").any()  # corrupt file excluded
    fractions = df["split"].value_counts(normalize=True)
    assert fractions["train"] == pytest.approx(0.70, abs=0.02)
    assert fractions["val"] == pytest.approx(0.15, abs=0.02)
    assert fractions["test"] == pytest.approx(0.15, abs=0.02)
    # Every class appears in every split.
    assert (df.groupby(["split", "class_name"]).size().unstack() > 0).all().all()


def test_split_is_deterministic(prepared):
    _, _, split_csv = prepared
    index = pd.read_csv(split_csv)[["path", "class_name", "label"]]
    shuffled = index.sample(frac=1, random_state=1)
    a = eurosat.make_splits(index, seed=42)
    b = eurosat.make_splits(shuffled, seed=42)  # input order must not matter
    c = eurosat.make_splits(index, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a["split"].equals(c["split"])


def test_mean_std_matches_numpy(prepared):
    report, class_root, split_csv = prepared
    df = pd.read_csv(split_csv)
    train = df[df["split"] == "train"]["path"]
    pixels = np.concatenate([
        np.asarray(Image.open(class_root / p).convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255
        for p in train
    ])
    np.testing.assert_allclose(report["normalization"]["mean"], pixels.mean(0), atol=1e-5)
    np.testing.assert_allclose(report["normalization"]["std"], pixels.std(0), atol=1e-5)


def test_prepare_is_rerunnable(prepared, fake_archive, tmp_path):
    report, _, _ = prepared
    again = eurosat.prepare_eurosat(raw_dir=fake_archive.parent, data_dir=tmp_path / "data",
                                    processed_dir=tmp_path / "processed")
    assert again["split"]["counts"] == report["split"]["counts"]
    assert json.loads((tmp_path / "processed" / "eurosat_report.json").read_text()) == again


def test_bad_md5_is_rejected(fake_archive, monkeypatch):
    monkeypatch.setattr(eurosat, "EUROSAT_MD5", "0" * 32)
    with pytest.raises(ValueError, match="MD5"):
        eurosat.download_eurosat(fake_archive.parent)


def test_unsafe_zip_is_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="Unsafe"):
        eurosat.extract_archive(bad, tmp_path / "out")


def test_dataset_and_transforms(prepared):
    report, class_root, split_csv = prepared
    norm = report["normalization"]
    train_ds = EuroSATDataset(split_csv, class_root, "train",
                              classification_transforms(norm["mean"], norm["std"], train=True))
    val_ds = EuroSATDataset(split_csv, class_root, "val",
                            classification_transforms(norm["mean"], norm["std"], train=False))
    assert train_ds.classes == sorted(CLASSES)
    image, label = train_ds[0]
    assert image.shape == (3, 64, 64) and image.dtype == torch.float32
    assert 0 <= label < 10
    loader = torch.utils.data.DataLoader(val_ds, batch_size=8)
    images, labels = next(iter(loader))
    assert images.shape == (8, 3, 64, 64) and labels.shape == (8,)


def test_eval_transform_is_deterministic(prepared):
    report, class_root, split_csv = prepared
    norm = report["normalization"]
    ds = EuroSATDataset(split_csv, class_root, "test",
                        classification_transforms(norm["mean"], norm["std"], train=False))
    assert torch.equal(ds[0][0], ds[0][0])


def test_rot90_rotates_image_and_mask_together():
    image = torch.arange(2 * 4 * 4).reshape(2, 4, 4)
    mask = image[0].clone()
    for _ in range(10):
        img_r, mask_r = RandomRot90()(image, mask)
        assert torch.equal(img_r[0], mask_r)
