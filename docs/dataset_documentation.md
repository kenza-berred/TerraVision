# Dataset documentation

Every fact below is labelled with where it comes from:
- **[source]**: the dataset authors' official pages (checked on the date shown).
- **[measured]**: produced by our own code on the downloaded files (`*_report.json`).
- **[pending]**: not verified yet.

---

## 1. EuroSAT (RGB) — classification

### 1.1 Source and license [source, checked 2026-11-05]

| | |
|---|---|
| Authors | Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth |
| Official repository | https://github.com/phelber/EuroSAT |
| Download | Zenodo record 7711810, version v2, DOI [10.5281/zenodo.7711810](https://doi.org/10.5281/zenodo.7711810). The older `madm.dfki.de` links are marked deprecated by the authors. |
| File used | `EuroSAT_RGB.zip`, 94,658,721 bytes, MD5 `f46e308c4d50d4bf32fedad2d3d62f3b` |
| License | MIT. Underlying imagery: Copernicus Sentinel-2 data, free and open under the Copernicus Sentinel data terms. |
| Description | Sentinel-2 based, 10 land-use/land-cover classes, 27,000 labelled and geo-referenced images. RGB version = red, green, blue bands as JPEG. |

### 1.2 Citation (as requested by the authors)

```bibtex
@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019},
  publisher={IEEE}
}

@inproceedings{helber2018introducing,
  title={Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  booktitle={IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium},
  pages={204--207},
  year={2018},
  organization={IEEE}
}
```

A preprint also exists: arXiv:1709.00029 (2017).

### 1.3 Measured properties [pending]

Filled from `eurosat_report.json` after the Colab run: archive layout, class names, images per class, image size and colour mode, corrupt files, duplicates.

### 1.4 Preparation pipeline

Code: `src/terravision/data/eurosat.py` → `prepare_eurosat()`.

1. **Download** from Zenodo to `MyDrive/TerraVision/datasets/raw/eurosat/`; kept only if the MD5 matches.
2. **Extract** to `/content/data/eurosat` (Colab local disk; faster than reading through the Drive mount).
3. **Scan** every image with a full decode: class, size, mode, byte-level duplicates. Unreadable files are excluded from the split and listed in the report, never silently fixed.
4. **Split** stratified 70 / 15 / 15 (train / val / test), seed 42, via two stratified `train_test_split` calls. Saved as `eurosat_split_seed42.csv`.
5. **Normalisation**: per-channel mean and std of pixels in [0, 1], computed on the **training split only** to avoid leaking validation/test information.

Outputs on Drive (`MyDrive/TerraVision/datasets/processed/eurosat/`): `eurosat_split_seed42.csv`, `eurosat_report.json`.

### 1.5 Augmentation (training only)

Code: `src/terravision/data/preprocessing.py`.

| Transform | Why |
|---|---|
| Horizontal + vertical flip, random 90° rotation | Overhead views have no canonical orientation; these keep pixels exact (no interpolation). |
| Brightness and contrast jitter, ±10% | Illumination and atmospheric differences between acquisitions. |
| No hue or saturation change | Colour is a key cue (water vs. vegetation vs. bare soil). |

Validation, test and inference use only tensor conversion + normalisation.

---

## 2. LandCover.ai — segmentation [pending]

To be verified from the official source and the downloaded files: version, format, class IDs, mask encoding, tiling, official splits, license.

---

## 3. Google Drive layout

See `docs/environment_setup.md` §4.
