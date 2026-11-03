# TerraVision — Phase 0 Audit

- **Date:** 2026-11-03
- **Scope:** Read-only inspection of the local repository, Git state, local `.venv`, and Claude skill setup. No files were changed except this report.
- **Not inspected:** The Colab notebook `TerraVision.ipynb` and Google Drive contents, which Claude Code cannot access. See §3.3.

Facts below come from commands run during the audit. Dataset properties in §7 are labelled **expected (verify)** where they come from the dataset authors' documentation and not from inspecting downloaded files. Phase 2 must confirm them.

---

## 1. Current project status

The project is at **day zero for code**. The scaffold exists and the specs (`CLAUDE.md`, `docs/PROJECT_SPEC.md`) are complete, but there is no Python code, notebook, backend, frontend, dataset or model yet.

| Item | State |
|---|---|
| Branch | `main`, the only branch, tracking `origin/main` |
| Remote | `origin` → `https://github.com/kenza-berred/TerraVision.git` |
| Local vs remote | In sync: local `HEAD` = remote `main` = `9311a3b` (checked with `git ls-remote`) |
| Commits | 1: `9311a3b Initial TerraVision project structure`, containing 3 **empty** files (`.gitignore`, `README.md`, `requirements.txt`) |
| Uncommitted | `.gitignore` and `requirements.txt` modified; `CLAUDE.md`, `docs/`, `notebooks/`, `src/` untracked |
| Git identity | `kenzaberred06-dot <kenzaberred06@gmail.com>`; the GitHub repo owner is `kenza-berred` |

## 2. Existing files and structure

```
TerraVision/
├── .gitignore            # 24 lines, CRLF, no trailing newline (modified, uncommitted)
├── CLAUDE.md             # 5.3 KB, project rules (untracked)
├── README.md             # EMPTY (0 bytes)
├── requirements.txt      # 10 unpinned packages (modified, uncommitted)
├── data/.gitkeep         # ignored by .gitignore
├── docs/
│   ├── .gitkeep
│   └── PROJECT_SPEC.md   # 10 KB (untracked)
├── models/.gitkeep       # ignored
├── notebooks/.gitkeep    # no notebooks
├── outputs/figures/.gitkeep, outputs/predictions/.gitkeep   # ignored
├── results/.gitkeep      # ignored
└── src/{classification,data,evaluation,segmentation}/.gitkeep   # src/data is ignored (bug, §4.1)
```

There are no `.py` files, no `.ipynb` files, and no `backend/`, `frontend/`, `configs/` or `tests/` folders. `src/` has no `__init__.py` and there is no `pyproject.toml`, so nothing is importable as a package yet.

**`requirements.txt`:** `numpy, pandas, matplotlib, seaborn, scikit-learn, pillow, opencv-python, tqdm, jupyter, ipykernel`. Nothing is pinned, and there is no PyTorch, FastAPI or test tooling.

**Local `.venv`:**
- Python **3.12.10** (CPython from `pythoncore-3.12-64`). Matches CLAUDE.md.
- **PyTorch: not installed.** `torch` and `torchvision` both fail to import, so there is no CPU or CUDA build yet. This is correct for now.
- Installed versions of note: numpy 2.5.4, pandas 3.1.0, opencv-python 5.0.0.93, pillow 13.0.0, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, seaborn 0.13.2, jupyterlab 4.6.5, ipykernel 7.4.0, PyYAML 6.0.3, httpx 0.28.1.
- FastAPI, uvicorn, pytest and python-multipart are not installed.

**Tooling:** Node v24.13.0, npm 11.6.2, pnpm 10.33.2 and Git 2.52 (`core.autocrlf=true`) are available.

**Frontend skill (Open Design `frontend-design`): not installed.**
- `TerraVision/.claude/skills/`: the folder doesn't exist.
- `~/.claude/skills/`: contains only `synced/`, which holds two unrelated synced skills.
- `~/.claude/plugins/installed_plugins.json` lists no plugins. Anthropic's own `frontend-design` plugin sits in the marketplace cache but is **not installed**, and it is a **different** skill from the Open Design one. The spec asks for `nexu-io/open-design`.
- An Open Design MCP server is configured in Claude Code, but its local app was not running (`127.0.0.1:7456` unreachable).
- The skill does exist upstream at `nexu-io/open-design/skills/frontend-design/` (`SKILL.md` + `LICENSE.txt`), checked on GitHub during this audit. Install steps are in §6.

## 3. Done vs. missing

### 3.1 Done
- Project rules and full specification (`CLAUDE.md`, `docs/PROJECT_SPEC.md`).
- Folder skeleton for `src/`, `data/`, `models/`, `results/`, `outputs/`, `notebooks/` and `docs/`.
- GitHub remote connected; local venv on the correct Python version with the scientific stack.
- First draft of `.gitignore` and `requirements.txt` (needs fixes, §4).

### 3.2 Missing (everything in spec §9, Phases 1–10)
- Config system (`configs/`, path resolution for local vs Colab).
- Package setup (`pyproject.toml` / `__init__.py`), so the same imports work locally and in Colab.
- Dataset download, verification and split code; `docs/dataset_documentation.md`.
- EDA notebooks, CNN code, U-Net code, training loops with checkpoint/resume, evaluation code.
- Trained models and real metrics. The app must show "Model not trained yet" until these exist.
- `backend/`, `frontend/`, `tests/`, README content, methodology/results docs, report and presentation.
- Local CPU PyTorch install (needs your approval).
- Open Design `frontend-design` skill (blocks Phase 8 only).

### 3.3 Unknown — needs your input
The Colab notebook `TerraVision.ipynb` on Drive isn't in the repo. **Please paste its cells** (or tell me it is empty or just experiments) so I can tell what has already been tried on Colab and whether any code needs to move into `src/`. Also useful: a listing of what exists in Drive under the project folder, if anything.

## 4. Problems and inconsistencies

### 4.1 `.gitignore` — one critical bug, several gaps

| # | Severity | Issue | Effect |
|---|---|---|---|
| G1 | **Critical** | `data/` has no leading `/`, so it matches **every** folder named `data`, including **`src/data/`**. Confirmed with `git status --ignored` (`!! src/data/`). | All dataset/transform code would never be committed. Colab `git pull` would silently miss it. |
| G2 | High | `models/`, `results/` and `outputs/` have the same unanchored problem. | Any future `backend/models/` or `src/**/results/` would be ignored. |
| G3 | High | `results/` is ignored entirely, but spec §4 says `results/*.json` holds the real metrics the backend serves. | Metrics wouldn't be versioned; a fresh clone has no metrics page data. Small JSON should be tracked. |
| G4 | Medium | Every `.gitkeep` inside an ignored folder is itself ignored (`data/`, `models/`, `results/`, `outputs/*`). | Those folders don't exist after `git clone` (including in Colab). |
| G5 | Medium | Missing patterns: `*.zip`, `*.tar*`, `*.tif`/`*.tiff`, `*.npy`/`*.npz`, `*.onnx`, `*.safetensors`, `*.h5`, `node_modules/`, `frontend/dist/`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`, `runs/` / `lightning_logs/` (TensorBoard), `.env.*`, `Thumbs.db`, `.DS_Store`. | A stray dataset zip, TIFF or `node_modules` could get committed. |
| G6 | Low | No trailing newline. No `.gitattributes`; files are CRLF on Windows while Colab is Linux. | Line-ending noise in diffs between the two environments. |

### 4.2 `requirements.txt` — would cause trouble in Colab and is incomplete locally

| # | Issue | Why it matters |
|---|---|---|
| R1 | **Nothing pinned.** | `pip install -r requirements.txt` in Colab would upgrade Colab's preinstalled numpy/pandas to the newest releases. Local already has numpy 2.5 and pandas 3.1, which may not match what Colab's torch, opencv and other preinstalled packages were built against. That gives broken imports or a forced runtime restart. Local and Colab could also diverge silently. |
| R2 | **`jupyter` + `ipykernel` in a shared file.** | Colab ships its own Jupyter stack. Upgrading `ipykernel` there is a known way to break the Colab kernel. These belong in a local-only file. |
| R3 | **`opencv-python` (GUI build).** | Colab preinstalls `opencv-python-headless`. Having both is a known conflict (two `cv2` builds). Use `opencv-python-headless` everywhere: there is no GUI need. |
| R4 | **No PyTorch.** Correct to keep it out of the shared file, because local must be CPU and Colab already has CUDA torch. But the CPU install needs to be documented. | Local: `torch==2.11.0` + matching `torchvision` from the PyTorch CPU index, to match Colab's 2.11 so checkpoints load. Needs your approval before installing. |
| R5 | **Missing app/test deps.** | `fastapi`, `uvicorn`, `python-multipart` (uploads), `pyyaml` (configs), `pytest`. |
| R6 | One file mixes three roles. | Shared ML code, local-only tooling and backend have different needs. |

### 4.3 Other inconsistencies
- **README is empty** (0 bytes).
- **Nothing meaningful is committed.** `CLAUDE.md` and `docs/` exist only on this laptop, and the remote has three empty files. If this machine is lost, the spec is lost.
- **Not importable.** No package setup. Colab notebooks would need `sys.path` hacks, which conflicts with the "notebooks stay thin, import from `src/`" rule.
- **Git identity vs GitHub account.** Commits are authored as `kenzaberred06-dot`; the repo belongs to `kenza-berred`. This is harmless if that email is verified on the `kenza-berred` account; otherwise commits won't show on your GitHub profile. Worth checking for the portfolio.
- `docs/.gitkeep` is redundant now that `docs/` has files. It can go at the next cleanup (not deleted by this audit).
- Spec §7 lists `configs/`, `backend/`, `frontend/` and `tests/`; none exist yet. This is expected and they get created when needed.

## 5. Recommended final architecture

The spec's §7 structure is sound. Recommended **deviations and additions**, each with the reason:

| # | Change | Reason |
|---|---|---|
| A1 | **Make the code an installable package**: add `pyproject.toml` and move `src/*` into `src/terravision/` (src-layout), installed with `pip install -e .` locally and in Colab. Imports become `from terravision.data import ...`. | One import style everywhere, with no `sys.path` hacks in notebooks or the backend. It also guards against the `src/data` vs `data/` name clash. Alternative if you prefer the spec layout as written: keep `src/` and add `__init__.py` files, but then everything must run from the repo root. |
| A2 | **Split requirements** into `requirements/base.txt` (shared, pinned to Colab-compatible ranges), `requirements/local.txt` (`-r base` + jupyter, fastapi, uvicorn, pytest) and `requirements/colab.txt` (`-r base` minus anything Colab already has). Torch stays out of all of them, with documented install commands. | Fixes R1–R6. Colab only installs what's missing; local gets the CPU torch build. |
| A3 | **Path config via environment variables + YAML.** `TERRAVISION_DATA_DIR`, `TERRAVISION_RUNS_DIR` and similar, with defaults to repo folders locally; Colab sets them to Drive paths in the first cell. Hyperparameters go in `configs/*.yaml`. | Satisfies "no hardcoded absolute paths". The same `train.py` runs in both places. |
| A4 | **Stage datasets onto Colab's local disk.** Keep zips on Drive and copy+unzip them to `/content/data` at session start. Never train by reading thousands of small files straight from the mounted Drive. | Reading many small files from mounted Drive is very slow and can time out. Copying one zip is fast. |
| A5 | **Track small result JSONs and split files in Git**: `results/*.json` and split index CSVs. Keep checkpoints, datasets and figures out. | Reproducibility, and the metrics page works on a fresh clone. Checkpoints stay on Drive and are downloaded into `models/`. |
| A6 | **Run directory convention**, one folder per run on Drive: `config.json`, `history.json`, `metrics.json`, `checkpoint_last.pt` (model+optimizer+scheduler+epoch, for resume), `model_best.pt` (`state_dict` only) and `model_meta.json` (architecture, classes, normalisation stats, input size, torch version). | Covers the resume requirement and gives the backend everything it needs to load on CPU. |
| A7 | **Build a vertical slice early.** Right after the CNN baseline (Phase 4), build a minimal `/classify` endpoint and load the Colab checkpoint on local CPU *before* starting U-Net. | Checks the hardest integration risk (Colab GPU → local CPU loading, matching preprocessing) while it's cheap to fix. The full backend still comes in Phase 7. |
| A8 | **Frontend served by FastAPI in demo mode.** The built static files (`frontend/dist`) are mounted by FastAPI, so one command runs the whole app. During development, use the Vite dev server with a proxy to the API. | No extra infrastructure (no Docker, no separate server), consistent with CLAUDE.md. |
| A9 | Add a `.gitattributes` (`* text=auto eol=lf`, `*.ipynb` text, binaries marked binary). Consider stripping notebook outputs before commit (e.g. `nbstripout`, which is a new dependency, so only with your OK). | Clean diffs between Windows and Linux. Notebook outputs bloat the repo. |

Everything else follows the spec: `src/terravision/{data,classification,segmentation,evaluation}`, separate `train.py` and `predict.py`, `backend/{main.py,api/,services/,schemas/}`, `tests/` with tiny synthetic fixtures (no real data needed for CI-style tests), and no Docker, database or auth.

## 6. Frontend stack recommendation

**Recommendation: Vite + React + TypeScript, styled with a hand-built design-token layer (CSS custom properties + CSS Modules), with React Router for the 5 pages and D3 (selected modules) or Observable Plot for scientific charts.** This is provisional: the Open Design skill comes first and may shape the choice. It needs your agreement (spec §6.1.4).

Why:
- **FastAPI integration:** A static single-page app builds to plain files that FastAPI can serve (A8). Next.js would add a Node server and server-side rendering that this app doesn't need: all inference is on the Python side.
- **Maintainability:** TypeScript types can mirror the FastAPI response schemas (FastAPI's OpenAPI output can generate them), which catches API drift. Five pages with shared upload, result and chart components are past what plain HTML/JS handles cleanly.
- **Design quality:** A token-based CSS system gives full control over the requested identity (navy, earth green, turquoise, off-white, deliberate typography). Utility frameworks and component kits such as Tailwind defaults or MUI pull toward the "stock dashboard" look the spec explicitly forbids.
- **Charts:** Confusion matrices, per-class IoU bars and training curves need precise, custom styling. D3/Plot give that; heavy dashboard chart libraries fight it.
- **Tooling ready:** Node 24 and pnpm are already installed.

**Blocker for Phase 8 — install the Open Design skill** (one-time, project-scoped, two files from the official repo):

```powershell
# [LOCAL — POWERSHELL]  run from C:\Users\kenza\Desktop\TerraVision
New-Item -ItemType Directory -Force .claude\skills\frontend-design | Out-Null
$base = "https://raw.githubusercontent.com/nexu-io/open-design/main/skills/frontend-design"
Invoke-WebRequest "$base/SKILL.md"    -OutFile .claude\skills\frontend-design\SKILL.md
Invoke-WebRequest "$base/LICENSE.txt" -OutFile .claude\skills\frontend-design\LICENSE.txt
```

Then restart Claude Code so it picks up the skill. Not urgent: the frontend is Phase 8.

## 7. Dataset preparation strategy

### 7.1 EuroSAT (RGB) — classification

**Expected (verify in Phase 2):** Helber et al., "EuroSAT", IEEE JSTARS 2019. Sentinel-2 based, RGB variant of 27,000 JPEG images at 64×64 px across 10 classes with 2,000–3,000 images each. Official source is the authors' GitHub (`phelber/EuroSAT`) linking to a Zenodo archive. License to be confirmed from the official source.

Plan:
1. **Download once** (Colab) from the official source → `Drive/TerraVision/datasets/raw/eurosat/EuroSAT_RGB.zip`. Record the URL, date, file size and SHA-256 in `docs/dataset_documentation.md`.
2. **Each session:** copy the zip to `/content/data/` and unzip (A4).
3. **Verify:** class folder names match the 10 expected classes, count per class, every file decodes, and all are 64×64 RGB. Report duplicates and corrupt files; never fix them silently.
4. **Split:** build an index (`relative_path, class, label_id`) and a stratified 70/15/15 split with a fixed seed (scikit-learn `train_test_split` twice). Save as `splits/eurosat_seed42.csv` (committed, A5).
5. **Normalisation:** per-channel mean/std computed on the **train split only**, saved to JSON and later into `model_meta.json`.
6. **Augmentation (training only):** horizontal and vertical flips plus 90° rotations (overhead imagery has no canonical "up") and mild brightness/contrast jitter. No hue shifts that would turn rivers green or fields blue.
7. **Local CPU testing:** a tiny sample (about 10 images per class, copied manually, gitignored) to test the Dataset class and transforms. Unit tests use synthetic images, so they need no real data.
8. **Note for the app:** EuroSAT images are about 10 m/pixel Sentinel-2 crops. User uploads (e.g. Google Maps screenshots) are a different resolution and sensor, so predictions on them will be weaker. The UI states this (spec §5).

### 7.2 LandCover.ai — segmentation

**Expected (verify in Phase 2 against the downloaded files and official docs):** Boguszewski et al., "LandCover.ai", CVPR Workshops 2021. Aerial orthophotos of Poland as GeoTIFF, mixed 25 cm and 50 cm resolution, multi-thousand-pixel images. Single-channel masks. Classes in v1: background, building, woodland, water, road. Ships with a tiling script that cuts 512×512 tiles and with official `train.txt` / `val.txt` / `test.txt` split lists. Website `landcover.ai.linuxpolska.com`. License believed to be non-commercial (CC BY-NC-SA); **must confirm** and cite exactly.

Plan:
1. **Download once** (Colab) from the official site → `Drive/TerraVision/datasets/raw/landcoverai/`. Record version, URL and checksum.
2. **Inspect before anything else:** list the archive contents; read the README; open several image/mask pairs; run `np.unique` on masks to **observe** the actual class IDs (never assume); check image/mask dimensions match pairwise; check CRS/resolution only if needed.
3. **Tile:** run the **official** split script (or a reimplementation proven identical on a sample) to get 512×512 tiles. Then use the **official train/val/test lists**. Zip the tiles → `Drive/.../processed/landcoverai_tiles_512.zip`, so tiling happens once, not every session.
4. **Class statistics:** pixel frequency per class on the train split. The data will very likely be imbalanced (background/woodland dominant, roads and buildings rare). This decides whether CE + Dice or class weights are justified in Phase 5/6.
5. **Training input size:** start with the full 512×512 tiles if the U-Net baseline fits on the T4 at a reasonable batch size (use mixed precision). Otherwise use random 256×256 crops. Measure memory and decide; document the choice.
6. **Transforms:** identical geometric transforms on image and mask (flips, 90° rotations). Resize masks with **nearest-neighbour only**. Colour jitter on the image only.
7. **Local CPU testing:** a handful of tiles for loader tests plus synthetic image/mask pairs in unit tests.

### 7.3 Drive layout (proposal, final in Phase 1)

```
MyDrive/TerraVision/
├── datasets/
│   ├── raw/{eurosat,landcoverai}/          # original archives, untouched
│   └── processed/                          # tile zips, computed stats
├── runs/{classification,segmentation}/<run_id>/   # A6 contents
├── figures/
└── predictions/
```

## 8. Prioritized roadmap

The spec's phases are kept. Changes are marked ★.

| Phase | Priority work | Exit check |
|---|---|---|
| **1 Architecture** | Fix `.gitignore` (G1–G6) → commit CLAUDE.md + docs + skeleton → `pyproject.toml` + package layout (A1) → split requirements (A2) → config/env path module (A3) → Drive layout doc → ★ install CPU torch 2.11 locally (with approval) → ★ install Open Design skill (any time before Phase 8) | Fresh `git clone` + `pip install -e .` works locally **and** in Colab; `import terravision` OK; `torch` CPU imports locally |
| **2 Data prep** | EuroSAT download/verify/split/stats; LandCover.ai download/inspect/tile/official splits/class stats; `dataset_documentation.md`; Dataset classes + tests | Loaders return correct shapes and dtypes on local samples; every dataset fact in the docs is backed by real output |
| **3 EDA** | Notebook 01, thin, calls `src` | Figures saved, reproducible |
| **4 CNN** | Baseline CNN, train loop with resume, eval, export | Real test metrics in `results/` |
| ★ **4b Vertical slice** | Minimal `/classify` loading the Colab checkpoint on local CPU | Same predictions on CPU as on Colab for a few test images |
| **5 U-Net** | Baseline U-Net, CE loss, resume, eval (mIoU, Dice, per-class) | Real metrics + visual panels |
| **6 Improvements** | Only justified ones (e.g. BN/deeper CNN, transfer-learning comparison, CE+Dice) | Documented comparisons on val; test touched once at the end |
| **7 Backend** | All endpoints, validation, "not trained" states, tests | pytest green; manual curl checks |
| **8 Frontend** | Open Design skill → design system doc → stack agreement → pages | All states work against the real API |
| **9 Testing** | End-to-end, responsiveness, a11y, Git sync | Checklist passed |
| **10 Deliverables** | README, docs, report, slides, screenshots | Complete |

Critical path: Phase 1 → EuroSAT part of Phase 2 → Phase 4 → 4b. LandCover.ai preparation (the largest download and the slowest step) can run in Colab in parallel with CNN work.

## 9. Exact first implementation action for Phase 1

**Fix `.gitignore` so `src/data/` is no longer ignored, then verify it before anything gets committed.**

Concretely:
1. Rewrite `.gitignore` with root-anchored patterns (`/data/*`, `/models/*`, `/outputs/**`) plus `!.gitkeep` negations. Track `results/*.json` and ignore everything else under `results/`. Add the missing patterns from G5 and a trailing newline. Add `.gitattributes` (A9).
2. Verify with `git check-ignore -v src/data/.gitkeep` (must print nothing) and `git status --ignored --short`. Expected: `src/data/`, `data/.gitkeep`, `models/.gitkeep` and the other placeholders show as untracked, not ignored, and `.venv/` stays ignored.
3. Then, with your OK, make the first real commit: `.gitignore`, `.gitattributes`, `CLAUDE.md`, `docs/`, and the skeleton `.gitkeep` files. Push it so Colab can clone a correct tree. This protects the spec, which currently exists only on this laptop.

Everything else in Phase 1 (package layout, requirements split, configs, CPU torch) comes after this commit.
