# Environment setup

TerraVision runs in two places with the same code:

| | Local (Windows) | Google Colab |
|---|---|---|
| Role | Write code, tests, backend, frontend, docs; CPU inference | Training and evaluation on GPU |
| Python | 3.12.10 in `.venv` | Colab's runtime |
| PyTorch | 2.11.0 **CPU** build | 2.11.0+cu130 (preinstalled) |
| Code | This repository | `git pull` of this repository |
| Large files | `models/` (downloaded checkpoints) | Google Drive |

Local and Colab use the **same PyTorch version (2.11)** so checkpoints trained on Colab load on the laptop's CPU.

---

## 1. Local setup (once)

```powershell
# [LOCAL — POWERSHELL]  from C:\Users\kenza\Desktop\TerraVision
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1. PyTorch CPU build (never the CUDA build: this laptop has no NVIDIA GPU)
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else (shared + notebooks + backend + tests)
pip install -r requirements-local.txt

# 3. The project itself, editable: `import terravision` works everywhere
pip install -e . --no-deps

# 4. Check
pytest
```

In VS Code, select `.venv` as the Python interpreter and as the Jupyter kernel.

## 2. Requirements files

| File | Used by | Contents |
|---|---|---|
| `requirements-base.txt` | both | numpy, pandas, matplotlib, seaborn, scikit-learn, pillow, opencv-python-headless, tqdm, pyyaml |
| `requirements-local.txt` | local | base + jupyter, fastapi, uvicorn, pytest, httpx |
| `requirements-colab.txt` | Colab | base only |
| `requirements.txt` | default | points to `requirements-local.txt` |

Versions use ranges (lower bound + major-version cap) rather than exact pins, so Colab's preinstalled packages already satisfy them and `pip` does not replace them. PyTorch is installed separately and is never listed: replacing Colab's CUDA build with a generic one would lose the GPU.

## 3. Paths and environment variables

All code gets folders from `terravision.config.get_paths()`. Defaults are inside the repo; Colab overrides them:

| Variable | Default (local) | Colab setting | Holds |
|---|---|---|---|
| `TERRAVISION_DATA_DIR` | `data/` | `/content/data` | Extracted datasets |
| `TERRAVISION_RUNS_DIR` | `runs/` | `MyDrive/TerraVision/runs` | One folder per training run |
| `TERRAVISION_OUTPUTS_DIR` | `outputs/` | `MyDrive/TerraVision/outputs` | Figures, prediction images |
| `TERRAVISION_RESULTS_DIR` | `results/` | `MyDrive/TerraVision/results` | Final metric / history JSONs |
| `TERRAVISION_MODELS_DIR` | `models/` | (unused) | Checkpoints served by the backend |

Datasets are extracted to `/content/data` (Colab's local disk), **not read from Drive directly**: reading thousands of small images through the Drive mount is very slow. Archives stay on Drive and are copied + unzipped at the start of a session.

## 4. Google Drive layout

```
MyDrive/TerraVision/
├── TerraVision.ipynb          # launcher notebook (setup cell + calls into src/)
├── datasets/
│   ├── raw/                   # original downloads, never modified
│   │   ├── eurosat/
│   │   └── landcoverai/
│   └── processed/             # e.g. LandCover.ai 512×512 tiles as one zip
├── runs/
│   ├── classification/<run_id>/
│   └── segmentation/<run_id>/
├── results/                   # final metrics JSONs (copied into the repo)
└── outputs/
    ├── figures/
    └── predictions/
```

Each run folder will contain: `config.json`, `history.json`, `checkpoint_last.pt` (model + optimizer + scheduler + epoch, to resume after a disconnect), `model_best.pt` (weights only) and `model_meta.json` (architecture, classes, normalisation stats, input size, torch version). Defined in Phase 4.

## 5. Colab setup cell

Paste this as the **first cell** of `TerraVision.ipynb` and run it at the start of every session (Runtime → Change runtime type → T4 GPU first).

```python
# [COLAB — PYTHON CELL]  TerraVision session setup
import os, sys, subprocess
from google.colab import drive

drive.mount("/content/drive")

REPO_URL = "https://github.com/kenza-berred/TerraVision.git"
REPO_DIR = "/content/TerraVision"
DRIVE_DIR = "/content/drive/MyDrive/TerraVision"

# Get the latest code. --ff-only refuses to merge if the Colab copy was edited,
# so local work (the source of truth) is never silently overwritten.
if os.path.isdir(f"{REPO_DIR}/.git"):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)

# Make `import terravision` work in this already-running kernel.
if f"{REPO_DIR}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_DIR}/src")

# Point heavy folders to fast disk (data) or persistent Drive (everything produced).
os.environ["TERRAVISION_DATA_DIR"] = "/content/data"
os.environ["TERRAVISION_RUNS_DIR"] = f"{DRIVE_DIR}/runs"
os.environ["TERRAVISION_OUTPUTS_DIR"] = f"{DRIVE_DIR}/outputs"
os.environ["TERRAVISION_RESULTS_DIR"] = f"{DRIVE_DIR}/results"

import torch, terravision
from terravision.config import get_paths
from terravision.utils import environment_info

paths = get_paths().ensure()
print("commit:", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
print(environment_info())
print(paths)
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
```

Why `sys.path` instead of `pip install -e .`: an editable install only becomes importable after a kernel restart, which would undo the rest of the setup. Adding `src/` to the path has the same effect immediately.

## 6. Sync workflow (one source of truth)

1. Edit code **locally** → `pytest` → commit → push.
2. In Colab, re-run the setup cell (it pulls).
3. Train/evaluate in Colab; everything produced goes to Drive.
4. Download the final checkpoint into local `models/` and the metrics JSON into local `results/`; commit the JSON (small), never the checkpoint.
5. Don't edit `src/` in Colab. If a quick fix is unavoidable there, copy it back into the local file and commit it locally, then `git checkout -- .` in Colab before the next pull.
