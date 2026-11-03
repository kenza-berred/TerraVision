# TerraVision — Project Specification

Reference spec for Claude Code. Rules and environment setup are in `CLAUDE.md`; this file holds the detailed requirements per area.

---

## 1. Objectives

TerraVision analyses satellite imagery for land use / land cover with two complementary tasks, served through a polished web app:

| Capability | Model | Dataset | Output |
|---|---|---|---|
| Scene classification | CNN | EuroSAT RGB | One class + confidence per image |
| Semantic segmentation | U-Net | LandCover.ai | Per-pixel class mask |

The result must be **scientifically defensible, technically functional, and visually impressive** — never trade correctness for looks or looks for correctness.

---

## 2. Datasets

### 2.1 EuroSAT RGB — classification
Expected (verify on download): ~27,000 RGB images, 64×64 px, 10 classes:
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

Before implementation:
- Confirm official source and license.
- Inspect directory structure, class counts, corrupted/missing files.
- Visualize samples per class.
- Compute normalization statistics on the **training split only**.

Split: stratified 70 / 15 / 15, fixed seed, split indices saved to disk for reproducibility.

### 2.2 LandCover.ai — segmentation
Verify from the actual dataset version and its official documentation (do not assume):
- Image and mask formats, dimensions, tiling procedure.
- Class list, class IDs, mask encoding.
- Image–mask pairing logic.
- Class imbalance (pixel frequency per class).
- License and usage terms.
- Whether official train/val/test splits exist — use them if so.

Original orthophotos are large; plan tiling/cropping to fit T4 memory and document the choice.

### 2.3 Storage
Raw and processed data live on Google Drive only. Document the Drive layout in `docs/dataset_documentation.md`.

---

## 3. Models

### 3.1 CNN (classification)
- Start with a clear baseline CNN (conv blocks → pooling → classifier with dropout; output logits; cross-entropy).
- Explain: convolutions, filters/feature maps, ReLU, pooling, FC layers, dropout, logits vs softmax, loss, optimizer, LR, batch size, epochs.
- Data augmentation appropriate for overhead imagery (flips, 90° rotations, mild colour jitter — justify choices).
- Monitor over/underfitting; use LR scheduling / early stopping when justified.
- Only after the baseline: consider justified improvements (e.g., deeper net, batch norm, transfer learning as a comparison).

### 3.2 U-Net (segmentation)
- Encoder, conv blocks, downsampling, bottleneck, decoder, upsampling, skip connections, 1×1 output conv to `num_classes`.
- Explain why U-Net suits segmentation, what skip connections recover, how output shape matches mask shape.
- Loss: cross-entropy baseline; consider CE + Dice if class imbalance justifies it.
- Baseline first; no exotic architectures before it works.

---

## 4. Experiments & evaluation

**Every run records**: config, hyperparameters, seed, hardware, duration, per-epoch losses/metrics, best checkpoint path. Saved as JSON alongside checkpoints on Drive.

**Checkpointing**: save model + optimizer + scheduler + epoch so Colab disconnects can resume.

**Classification metrics**: accuracy, macro & per-class precision/recall/F1, confusion matrix, classification report, misclassification gallery + analysis.

**Segmentation metrics**: mean IoU, per-class IoU, Dice, pixel accuracy. Visuals: image / ground truth / prediction / overlay / colour legend. Error analysis (which classes confuse, where boundaries fail).

**Export for the app**: `results/*.json` with real metrics and training histories, consumed by the backend's metrics endpoint.

---

## 5. Backend (FastAPI)

Runs locally on CPU, independent of Colab. Loads real checkpoints from `models/`.

Endpoints (adjust names as appropriate):
- `GET /health`
- `GET /models` — model info, load status, classes
- `POST /classify` — image → class, confidence, full probability distribution
- `POST /segment` — image → mask (encoded image), overlay, per-class pixel share
- `GET /metrics` — real evaluation results from `results/`

Requirements:
- Input validation (type, size, decodability), clear error responses.
- Preprocessing identical to training (same normalization stats, read from checkpoint metadata).
- Uploaded images rarely match training resolution (EuroSAT is 64×64): resize consistently and surface a note in the UI about domain shift.
- If a checkpoint is missing, endpoints return a clear "model not trained" state — never a simulated result.
- Training code and inference code stay separate (`src/*/train.py` vs `src/*/predict.py`).
- Local PyTorch must be the **CPU** build; match the Colab major/minor version for checkpoint compatibility (ask before installing).

---

## 6. Frontend

### 6.1 Mandatory process
1. Locate and read the Open Design `frontend-design` skill (`nexu-io/open-design`). If missing, stop and give me install instructions.
2. Follow it throughout.
3. Define the design system (palette, type scale, spacing, grid, components, motion, responsive breakpoints) **before** building pages, and document it.
4. Recommend a frontend stack with reasoning (maintainability, design quality, FastAPI integration) and get my agreement.

### 6.2 Visual identity
Premium geospatial-intelligence feel: Earth observation, environmental science, AI research, refined data visualization.

Direction: deep navy / midnight blue, earth green, muted turquoise, warm off-white; sophisticated typography; strong hierarchy; deliberate whitespace; real satellite imagery; subtle purposeful motion.

Avoid: purple AI gradients, heavy glassmorphism, decorative blobs, repetitive rounded-card grids, stock dashboard templates, gratuitous animation, fake stats.

### 6.3 Pages
1. **Landing** — brand, concept, real imagery, the two capabilities, real dataset facts, CTAs to the tools.
2. **Classification** — upload, preview, run, predicted class, confidence, distribution across all 10 classes, model info, real metrics.
3. **Segmentation** — upload, preview, run, mask, side-by-side, adjustable overlay, legend, real IoU/Dice.
4. **Performance** — training/validation curves, confusion matrix, per-class metrics, IoU/Dice per class; comparisons only where scientifically meaningful.
5. **Methodology** — objectives, datasets, architectures, preprocessing, training, metrics, limitations, future work.

### 6.4 Functional requirements
Real uploads and inference, loading states, error states, empty states, "model not trained yet" state, responsive layout, keyboard-accessible controls, working navigation. No mock data anywhere.

---

## 7. Target repository structure

Adapt to what exists — don't create empty files for the sake of it. Use `.gitkeep` for folders that must be tracked.

```
TerraVision/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                 # paths, hyperparameters (local vs Colab)
├── data/                    # gitignored; placeholders only
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classification_training.ipynb
│   ├── 03_segmentation_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/                # datasets, preprocessing, transforms
│   ├── classification/      # model.py, train.py, predict.py
│   ├── segmentation/        # unet.py, train.py, predict.py
│   └── evaluation/          # metrics, visualization
├── backend/                 # main.py, api/, services/, schemas/
├── frontend/
├── models/                  # gitignored checkpoints
├── results/                 # metric/history JSONs
├── outputs/                 # figures/, predictions/ (gitignored)
├── tests/
└── docs/
    ├── PROJECT_SPEC.md
    ├── dataset_documentation.md
    ├── methodology.md
    └── results_analysis.md
```

---

## 8. Documentation & academic deliverables

Final deliverables: source code, both models with checkpoints, data pipelines, reproducible experiments, evaluation scripts, prediction visuals, working app, professional README (incl. reproduction steps), technical docs, academic report, presentation, screenshots.

**Report outline**
1. Introduction — context, problem, motivation, objectives, contributions.
2. Background — satellite image analysis, CV, deep learning, CNNs, semantic segmentation, U-Net, related work.
3. Data — EuroSAT, LandCover.ai, exploration, preprocessing, augmentation, splits.
4. Methodology — architectures, training config, losses, optimizers, metrics.
5. Implementation — environments, architecture, backend, frontend, integration.
6. Results & discussion — classification, segmentation, visuals, error analysis, limitations.
7. Conclusion — summary, contributions, limitations, future work.

Use real academic and official sources only. Never invent citations.

---

## 9. Roadmap

Stop for review at the end of each phase.

| Phase | Goal | Exit criteria |
|---|---|---|
| 0 | Audit | Written audit delivered (see kickoff prompt) |
| 1 | Architecture | Structure, configs, dependency plan, Drive layout, sync workflow agreed |
| 2 | Data prep | Both datasets verified, documented; loaders tested on small samples locally |
| 3 | EDA | Reproducible exploration notebooks + figures |
| 4 | CNN | Baseline trained on Colab, evaluated, results exported |
| 5 | U-Net | Baseline trained on Colab, evaluated, results exported |
| 6 | Improvements | Justified experiments, documented comparisons |
| 7 | Backend | All endpoints working with real checkpoints, tested |
| 8 | Frontend | Open Design system + all pages integrated and tested |
| 9 | Testing | Pipelines, inference, API, UI states, responsiveness, Git sync verified |
| 10 | Deliverables | README, docs, report, presentation, screenshots |
