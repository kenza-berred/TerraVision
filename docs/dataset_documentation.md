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

### 1.3 Measured properties [measured, Colab run 2026-11-05]

From `eurosat_report.json` (archive MD5 verified):

| Property | Value |
|---|---|
| Archive layout | `EuroSAT_RGB/<ClassName>/*.jpg` |
| Images | 27,000 scanned, 27,000 readable, **0 corrupt**, **0 byte-identical duplicates** |
| Size / mode | All **64×64**, all **RGB** |
| Classes | 10, names identical to the paper |

| Class | Images | | Class | Images |
|---|---:|---|---|---:|
| AnnualCrop | 3,000 | | Pasture | **2,000** |
| Forest | 3,000 | | PermanentCrop | 2,500 |
| HerbaceousVegetation | 3,000 | | Residential | 3,000 |
| Highway | 2,500 | | River | 2,500 |
| Industrial | 2,500 | | SeaLake | 3,000 |

**Class balance:** mild imbalance, largest/smallest = 3,000 / 2,000 = 1.5. Accuracy alone would hide weaker small classes, so evaluation reports macro-averaged and per-class precision/recall/F1. Class-weighted loss is not planned for the baseline; it would only be added if per-class validation results justify it.

**Split (seed 42):** train 18,899 / val 4,051 / test 4,050 (total 27,000; 69.99% / 15.00% / 15.00%). Stratified, so each split keeps the class proportions above.

**Normalisation (train split only, pixels in [0, 1]):**

| | R | G | B |
|---|---|---|---|
| mean | 0.344829 | 0.380553 | 0.408277 |
| std | 0.203084 | 0.137132 | 0.115618 |

These values are stored with every classification checkpoint so inference uses exactly the same preprocessing.

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

## 2. LandCover.ai — segmentation

### 2.1 Source and license [source, checked 2026-11-05]

| | |
|---|---|
| Official site | https://landcover.ai.linuxpolska.com/ |
| Version used | **v1** (adds the road class and refines annotations; v0 is described as not peer reviewed) |
| File | `landcover.ai.v1.zip`, 1,538,212,277 bytes (server Content-Length) |
| Checksum | No official checksum published. We verify SHA-256 `dd448287…149a4b`, the value TorchGeo (`microsoft/torchgeo`) pins for the same URL. |
| License | **CC BY-NC-SA 4.0** (non-commercial, share-alike). Fine for this academic project; the app and report must credit the dataset, and it may not be used commercially. |
| Imagery (documented) | Aerial orthophotos of Poland, RGB GeoTIFF, EPSG:2180. 33 at 25 cm/px (~9000×9500 px) and 8 at 50 cm/px (~4200×4700 px); 216.27 km² total. |
| Masks (documented) | Single-channel GeoTIFF. v1 values: building = 1, woodland = 2, water = 3, road = 4. TorchGeo uses 0 = background. |
| Splits (documented) | Official `train.txt` / `val.txt` / `test.txt` listing 512×512 tiles produced by the included `split.py`. |

Citation (from the official site): Boguszewski, A., Batorski, D., Ziemba-Jankowska, N., Dziedzic, T., Zambrzycka, A. *LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery.* CVPR Workshops 2021, pp. 1102–1110. (Exact BibTeX to be copied from the site into the report.)

### 2.2 Measured properties [pending]

Measured by `inspect_landcoverai()` (`src/terravision/data/landcoverai.py`): archive layout, image vs. mask rasters (by band count), pairing, sizes, dtypes, and the actual values present in the masks. `split.py` is read as text and reviewed, not executed. Tiling and the segmentation Dataset are written only after this inspection.

---

## 3. Google Drive layout

See `docs/environment_setup.md` §4.
