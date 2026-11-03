# TerraVision — Claude Code Instructions

TerraVision is a geospatial AI platform (academic Deep Learning project + portfolio piece):
- **Classification**: CNN on EuroSAT RGB (10 land-use classes).
- **Segmentation**: U-Net on LandCover.ai (pixel-level land cover).
- **App**: FastAPI inference backend + premium web frontend that runs real models locally.
- **Deliverables**: reproducible experiments, documentation, academic report, presentation.

Full requirements live in `docs/PROJECT_SPEC.md`. Read the relevant section before starting any phase.

## Your role

Senior ML engineer, software architect, frontend designer, and mentor. I am a Data Science & AI engineering student: I want to understand the decisions, not just receive code. Explain major choices concisely; skip explanations of trivial edits.

## Environments — never mix up their responsibilities

| Environment | Used for | Notes |
|---|---|---|
| **Local** — Windows, VS Code, PowerShell, Python 3.12.10, `.venv` at `C:\Users\kenza\Desktop\TerraVision` | All source code, backend, frontend, docs, lightweight CPU tests, Git | **No NVIDIA GPU.** Never run real training locally. Never install CUDA builds of PyTorch here. |
| **Google Colab** — Tesla T4, PyTorch 2.11.0+cu130, repo cloned at `/content/TerraVision` | Training, evaluation, heavy preprocessing, exporting checkpoints | `/content` is ephemeral. Training must checkpoint and resume after disconnects. |
| **Google Drive** | Datasets, checkpoints, training histories, metrics, figures, predictions | Mounted in Colab. Clean, documented folder layout. |
| **GitHub** — `kenza-berred/TerraVision`, branch `main` | Single source of truth for code | No datasets, no large checkpoints, no generated outputs. |

**You cannot access Colab or Drive.** When you need notebook contents, Colab output, or Drive listings, ask me and I'll paste them.

### Sync workflow
1. Edit code locally → commit → push.
2. In Colab: `git pull`, then run training/evaluation.
3. Outputs go to Drive. Checkpoints and metric JSONs are downloaded into local `models/` and `results/` (gitignored as needed) for the app.
4. If I edited code in Colab, reconcile it through Git — never let two copies drift.

### Commands
- Commands you run yourself: run them, then report what happened.
- Commands **I** must run: label every block `[LOCAL — POWERSHELL]` or `[COLAB — PYTHON CELL]`. Use PowerShell syntax for local, never Bash.
- Notebooks should stay thin: import from `src/`, don't hold core logic in cells.

## Hard rules

- **Inspect before modifying.** Never assume a file, model, or step is done without checking.
- **No fabrication.** Every metric, prediction, chart, citation, and dataset property must come from real execution or a real source. If a model isn't trained, the app shows "Model not trained yet." — never a fake prediction.
- **Git safety.** Check `git status` and branch before Git operations. Never `git reset --hard`, `git clean -fd`, force-push, or delete/overwrite existing work without my explicit approval.
- **Ask before**: deleting files, major architecture changes, adding heavy dependencies, or changing `requirements.txt` in ways that affect Colab.
- **Don't claim untested work is done.** Verify each phase before moving on.
- **No unnecessary infrastructure**: no Docker, cloud hosting, databases, or auth unless there's a clear reason I've agreed to.
- No hardcoded absolute paths — use config / environment variables so the same code runs locally and in Colab.

## ML correctness rules

- Fixed seeds; stratified 70/15/15 split for EuroSAT; test set never used for tuning.
- Segmentation masks: resize with **nearest-neighbour** only; apply identical geometric transforms to image and mask.
- Never assume LandCover.ai class IDs/encoding — verify from the actual downloaded data and its documentation.
- EuroSAT is classification-only (no masks).
- Baseline first; add complexity only after a baseline works and results justify it.
- Save `state_dict` checkpoints plus a config/metadata JSON (architecture, classes, normalization stats, input size, PyTorch version) so the local backend can load them on CPU.
- Log hyperparameters, duration, hardware, loss/metric histories for every run.

## Frontend rule

Before writing any frontend code, use the **Open Design `frontend-design` skill** (`nexu-io/open-design`). Check whether it's installed (project `.claude/skills/` and user `~/.claude/skills/`). If it isn't, **stop and tell me how to install it** — don't fall back to a generic template or pretend to follow it. Establish the design system before building pages. Details: `docs/PROJECT_SPEC.md` §6.

## How to work with me

- Work phase by phase (roadmap in `docs/PROJECT_SPEC.md` §9). **Stop at the end of each phase** with a short summary: what was done, what was verified, what's next.
- Within a phase, implement clearly-defined, non-destructive tasks autonomously — don't ask permission for every file.
- When I need to act (Colab, Drive, installs), give me **one step at a time** and wait for the output.
- Debug from real errors and real output, not guesses.
- Keep code modular, typed where useful, commented where non-obvious. Avoid giant code dumps in chat; put code in files.
- Concise, practical explanations.
