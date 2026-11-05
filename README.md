# TerraVision

Deep Learning for satellite imagery: land-use **classification** with a CNN (EuroSAT RGB) and land-cover **semantic segmentation** with a U-Net (LandCover.ai), served through a local web application.

> **Status:** Phase 1 (project architecture). No dataset has been downloaded and no model has been trained yet.

## Repository layout

```
src/terravision/      Python package (data, classification, segmentation, evaluation)
tests/                pytest suite (runs on CPU, no dataset needed)
docs/                 specification, environment setup, documentation
notebooks/            thin notebooks that call into src/
data/ models/ runs/   local data and checkpoints (gitignored)
results/              small metric JSONs (tracked)
```

## Quick start (local, Windows)

See [docs/environment_setup.md](docs/environment_setup.md) for the full local + Colab setup.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-local.txt
pip install -e . --no-deps
pytest
```

Training runs on Google Colab (GPU); the local machine is used for development and CPU inference.
