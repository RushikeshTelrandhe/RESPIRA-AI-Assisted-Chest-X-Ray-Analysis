# RESPIRA — AI-Assisted Chest X-Ray Analysis

<p align="center">

**An AI-assisted chest X-ray analysis platform combining high-resolution deep learning, model fusion, and explainable AI.**

</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python 3.13">
<img src="https://img.shields.io/badge/PyTorch-2.10-red?logo=pytorch" alt="PyTorch 2.10">
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" alt="FastAPI">
<img src="https://img.shields.io/badge/React-19-Frontend-61DAFB?logo=react" alt="React 19">
<img src="https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript" alt="TypeScript 5">
<img src="https://img.shields.io/badge/Vite-Frontend-646CFF?logo=vite" alt="Vite">
<img src="https://img.shields.io/badge/Computer%20Vision-Chest%20X--Ray-orange" alt="Chest X-Ray">
<img src="https://img.shields.io/badge/XAI-Grad--CAM%20%7C%20Attention-purple" alt="XAI">
<img src="https://img.shields.io/badge/Models-Hugging%20Face-yellow?logo=huggingface" alt="Hugging Face">

</p>

---

## 📌 Overview

**RESPIRA** is an AI-assisted chest X-ray analysis system developed as a Final Year Project.

The system combines:

- High-resolution chest X-ray preprocessing (CLAHE, lung masking, 224×224 and 512×512)
- Deep learning classification (EfficientNet-B0, ViT-B/16, DenseNet-121)
- Probability-level model fusion (weighted 512 fusion + advanced 224 fusion pipeline)
- Explainable AI (Grad-CAM, ViT attention visualization)
- FastAPI backend with JWT auth, patients, analyses, history, and PDF reports
- React 19 + TypeScript + Vite + Tailwind frontend
- Hugging Face-hosted model checkpoints
- One-click Windows startup (`start_respira.bat`)
- Desktop shell (`electron/`), pytest + vitest suites, and `docs/`

> **Important:** RESPIRA is a research/academic system and is **not a medical diagnostic device**. Predictions must be reviewed by qualified healthcare professionals and must not be used as a substitute for clinical evaluation.

---

## ⚡ Quickstart (Windows)

```powershell
git clone https://github.com/RushikeshTelrandhe/RESPIRA-AI-Assisted-Chest-X-Ray-Analysis.git
cd RESPIRA-AI-Assisted-Chest-X-Ray-Analysis

python -m venv env
.\env\Scripts\activate

pip install -r requirements-app.txt -r requirements-backend.txt

copy .env.example .env   # set JWT_SECRET

cd frontend; npm install; cd ..
```

Then run:

```powershell
.\start_respira.bat
```

Access:

- Frontend: <http://localhost:5173>
- Backend: <http://127.0.0.1:8000>
- Health: <http://127.0.0.1:8000/api/v1/health>

The startup script validates Python/venv/Node, installs frontend deps if needed, downloads missing 512×512 checkpoints from Hugging Face, starts FastAPI, waits for health, then starts Vite.

---

## ✨ Key Features

### 🧠 AI Models

| Pipeline | Models | Input | Notes |
|---|---|---|---|
| Legacy / baseline | EfficientNet-B0, ViT-B/16, DenseNet-121 | 224×224 | Research baselines |
| High-resolution | EfficientNet-B0, ViT-B/16, weighted fusion | 512×512 | Preserves more spatial detail |
| Advanced fusion | EfficientNet-B0 + ViT-B/16 with cross-attention, disease attention, adaptive fusion | 224×224 features | Full pipeline under `outputs/fusion/` + `outputs/final_prediction/` |

### 🔍 Explainability

- Grad-CAM for the EfficientNet branch
- ViT attention visualization for the transformer branch
- Paired XAI comparison and artifact audit under `outputs/xai_comparison/`

### 🖥️ Application

- Auth (JWT + bcrypt), patients, doctors, X-ray analysis, history, explainability, reports
- SQLite by default (`backend_storage/`), Postgres supported via `DATABASE_URL`
- PDF report generation, audit log, model status endpoints

---

## 🩻 Supported Classes

Six-class classification:

1. Atelectasis
2. Bacterial Pneumonia
3. Normal
4. Pulmonary Edema
5. Tuberculosis
6. Viral Pneumonia

---

## 🔬 Pipelines

### High-Resolution 512×512 Pipeline

```text
Original Chest X-Ray
        │
        ▼
Image Acquisition
        │
        ▼
Quality / Dataset Validation
        │
        ▼
Class Balancing
        │
        ▼
CLAHE Enhancement
        │
        ▼
Lung Region Processing / Masking
        │
        ▼
512 × 512 RGB Image
        │
        ├──────────────────────┐
        ▼                      ▼
EfficientNet-B0             ViT-B/16
        │                      │
        ▼                      ▼
Class Probabilities       Class Probabilities
        │                      │
        └──────────┬───────────┘
                   ▼
          Weighted Probability
                Fusion
                   │
                   ▼
             Final Prediction
                   │
                   ▼
            Explainable AI
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Grad-CAM       ViT Attention
          │                 │
          └────────┬────────┘
                   ▼
             Visualization
```

### Advanced 224×224 Fusion Pipeline

The repository also contains the full research fusion stack (used by the live backend `RespiraPipeline`):

```text
224×224 RGB
    │
    ├──────────────┐
    ▼              ▼
EfficientNet-B0  ViT-B/16
    │              │
    ▼              ▼
Projected features (outputs/fusion/projected_features/)
    │
    ▼
Bidirectional cross-attention (outputs/fusion/cross_attention/)
    │
    ▼
Disease-conditioned attention (outputs/fusion/disease_attention/)
    │
    ▼
Disease relationship modeling (outputs/fusion/disease_relationship/)
    │
    ▼
Adaptive fusion + multi-label classification heads
(outputs/fusion/adaptive_fusion/, outputs/fusion/classification/)
    │
    ▼
Final prediction + uncertainty (outputs/final_prediction/)
```

> Both pipelines are real and versioned in `outputs/`. The 512 pipeline is exposed as `efficientnet-512` / `vit-512` / `fusion-512`; the advanced 224 pipeline is exposed as `fusion` / `efficientnet` / `vit` (plus `densenet` baseline).

---

## 📊 Dataset

The project uses a chest X-ray dataset cleaned specifically for this research. The original dataset is **not** included in this Git repository (size + distribution restrictions).

Preprocessing workflow:

1. Dataset acquisition
2. Image validation (file validity, readability, format, missing/corrupted files, class organization)
3. Class organization
4. Dataset balancing
5. Contrast enhancement (CLAHE)
6. Lung-region processing
7. Image resizing (224×224 and 512×512)
8. RGB conversion
9. Train/validation/test split

Final outputs:

- `512 × 512 × 3` RGB images (high-resolution pipeline)
- `224 × 224 × 3` RGB images (standard pipeline)

---

## 🧹 Preprocessing Pipeline

Pipeline order: acquisition → validation → balancing → CLAHE → lung masking → resize → RGB.

1. **Image acquisition** — X-rays collected and organized by disease class.
2. **Dataset validation** — checks file validity, readability, format, missing/corrupted files, class layout.
3. **Class balancing** — underrepresented classes are upsampled during dataset preparation.
4. **CLAHE enhancement** — Contrast Limited Adaptive Histogram Equalization (clip limit 2.0, tile grid 8×8) improves local contrast.

   ```text
   Original X-Ray
         │
         ▼
       CLAHE
         │
         ▼
   Enhanced X-Ray
   ```

5. **Lung region processing** — Otsu thresholding + morphology mask reduces background and focuses on lung fields.
6. **Image resizing** — 512×512 for the high-resolution branch; 224×224 for the standard branch.

Implementation: `app/backend/preprocessing.py` (`apply_clahe`, `create_lung_mask`, `apply_lung_mask`, `process_image_array`, `preprocess_pil`). Default `IMAGE_SIZE = 224` in `app/backend/config.py`; pass an explicit size (e.g. `process_image_array(image, image_size=512)`) for the high-resolution path.

---

## 🧠 Model Architecture

### EfficientNet-B0

- Efficient parameter usage, strong local feature extraction
- High-resolution config: input `512 × 512 × 3` → 6 disease classes
- Test accuracy (512): **92.40%** (1947 samples)

### 🤖 Vision Transformer — ViT-B/16

High-resolution configuration:

| Setting | Value |
|---|---|
| Input | 512 × 512 |
| Patch size | 16 × 16 |
| Patch grid | 32 × 32 |
| Number of patches | 1024 |
| Embedding dim | 768 |

Supports both 224×224 (14×14 = 196 patches) and 512×512 (32×32 = 1024 patches); positional embeddings are interpolated for the larger grid.

- Test accuracy (512): **91.32%** (1947 samples)

### 🔀 Probability Fusion (512)

Weighted fusion of the two 512 backbones:

```text
P_fusion = 0.35 × P_EfficientNet + 0.65 × P_ViT
```

Final class = `argmax(P_fusion)`. Weights come from `outputs/fusion/fusion_config.json` (validation accuracy 92.97%).

---

## 📈 Model Performance

All numbers below are read directly from the checked-in `outputs/` metrics files — experimental research results, not clinical claims.

### 512×512 weighted fusion (`outputs/fusion/test/metrics.json`)

| Model | Test Accuracy |
|---|---|
| EfficientNet-B0 (512) | 92.40% |
| ViT-B/16 (512) | 91.32% |
| Fusion 512 (0.35 / 0.65) | **92.45%** |

| Metric | Result |
|---|---|
| Test samples | 1947 |
| Accuracy | 92.45% |
| Macro precision | 93.86% |
| Macro recall | 93.85% |
| Macro F1 | 93.85% |
| Macro ROC-AUC | 0.9910 |

Best validation config: EfficientNet 0.35 / ViT 0.65, validation accuracy 92.97%.

### Advanced fusion + final prediction (`outputs/final_prediction/reports/final_metrics.json`)

| Metric | Result |
|---|---|
| Test samples | 2072 |
| Model | MultiLabelClassificationHeads (`outputs/fusion/classification/checkpoints/best_model.pth`) |
| Accuracy | 88.08% |
| Macro precision | 87.88% |
| Macro recall | 89.50% |
| Macro F1 | 88.62% |
| Macro ROC-AUC | 0.9794 |

### 🧪 XAI Comparison (`outputs/xai_comparison/`)

Recorded agreement on the 512 test split (1947 images):

| Group | Count |
|---|---|
| Images compared | 1947 |
| Prediction agreement | 93.73% |
| Both correct | 1729 |
| EfficientNet correct / ViT wrong | 70 |
| ViT correct / EfficientNet wrong | 49 |
| Both wrong | 99 |

---

## 🔍 Explainable AI

### Grad-CAM

Used for the EfficientNet branch — class-specific activation map over the input X-ray.

```text
Chest X-Ray → EfficientNet → Prediction → Gradients → Activation Map → Heatmap Overlay
```

### 👁️ Vision Transformer Attention

The ViT branch is visualized via patch attention:

```text
512 × 512 → 16 × 16 patches → 32 × 32 grid → 1024 tokens → Attention map
```

This complements Grad-CAM: CNN localization vs. transformer patch attention.

Artifacts, contact sheets, and representative figures live under `outputs/xai_comparison/` and `outputs/final_analysis/`.

---

## 🏗️ System Architecture

```text
                     RESPIRA
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
Frontend             Backend             AI Layer
    │                   │                   │
    ▼                   ▼                   ▼
React 19 + TS      FastAPI           PyTorch Models
+ Vite + Tailwind      │             (224 + 512)
    │             ┌─────┴──────┐         │
    │             ▼            ▼         ▼
    │         Database     Services → Inference
    │         (SQLite /           ┌──────┼──────┐
    │          Postgres)          ▼      ▼      ▼
    │                       EfficientNet ViT  Fusion
    ▼
Clinical / Research UI
```

### 🖥️ Frontend

Stack: React 19, TypeScript, Vite, Tailwind CSS, shadcn-style UI, lucide-react, framer-motion, recharts, REST API integration.

Covers: auth, dashboard, patient management, X-ray analysis, history, explainability, reports, patient details, profile, model/system info.

Pages (`frontend/src/pages/`):

```text
AnalysisDetail.tsx
Dashboard.tsx
Explainability.tsx
History.tsx
Landing.tsx
Login.tsx
Misc.tsx          # model status / system pages
PatientDetail.tsx
Patients.tsx
Profile.tsx
Reports.tsx
Signup.tsx
XrayTest.tsx
```

### ⚙️ Backend

Stack: Python, FastAPI, Uvicorn, PyTorch, Torchvision, SQLAlchemy (SQLite → Postgres), JWT + bcrypt, reportlab PDFs.

Entry point: `backend/app/main.py`

APIs (`backend/app/api/`): `auth`, `patients`, `doctors`, `analyses`, `history`, `explainability`, `reports`, `system`.

Services (`backend/app/services/`): `inference_service`, `explainability_service`, `model_manager`, `report_service`.

### 🧠 Model Manager

`backend/app/services/model_manager.py` exposes (`MODEL_REGISTRY`):

| Key | Description |
|---|---|
| `fusion` | Full 224 EfficientNet + ViT fusion (recommended) |
| `efficientnet` | 224 CNN baseline |
| `vit` | 224 transformer baseline |
| `densenet` | DenseNet-121 baseline (`models/densenet/best_model.pth`) |
| `efficientnet-512` | 512×512 CNN (~92.4% test) |
| `vit-512` | 512×512 transformer (~91.3% test) |
| `fusion-512` | Weighted 512 fusion (~92.9% val) |

Handles registration, loading, device selection (`cuda` if available), checkpoints, and availability via `/api/v1/models/status`.

---

## 📁 Project Structure

```text
RESPIRA/
├── app/backend/            # research inference (pipeline.py, gradcam.py,
│                           # vit_attention.py, preprocessing.py, config.py)
├── backend/app/            # clinical backend (api/, services/, main.py,
│                           # database.py, models_db.py, schemas.py, security.py)
├── frontend/src/           # React 19 + TS + Vite (components/, context/,
│                           # pages/, services/, utils/)
├── src/                    # research pipeline (models/, training/,
│                           # evaluation/, fusion/, explainability/,
│                           # inference/, analysis/, features/, datasets/)
├── electron/               # desktop shell (main.ts, preload.ts)
├── tests/                  # pytest backend suite (incl. real-model e2e)
├── docs/                   # architecture, setup, backend, frontend, database,
│                           # ml-inference, explainability, security, electron, deployment
├── scripts/download_models.py
├── configs/  data/  deployment/  notebooks/  preprocessing/  experiments/
├── models/densenet/        # densenet checkpoint (deployment location)
├── outputs/                # metrics, fusion, final_prediction, xai (git-ignored pth)
├── backend_storage/        # SQLite DB + stored studies/overlays/reports (git-ignored)
├── .env.example  docker-compose.yml  requirements*.txt  start_respira.bat
└── README.md
```

---

## 🤗 Hugging Face Models

Large checkpoints are **not** stored in Git. Trained files live at:

- Repo: `rushikeshtelrandhe/respira-models`
- URL: <https://huggingface.co/rushikeshtelrandhe/respira-models>

Contains `efficientnet_b0_512/best_model.pth` + `metrics.json`, `vit_512/best_model.pth` + `metrics.json`, `fusion_512/fusion_config.json`, plus legacy baselines.

### ⬇️ Automatic Model Download

`scripts/download_models.py` fetches the five 512 files:

```text
efficientnet_b0_512/best_model.pth
vit_512/best_model.pth
efficientnet_b0_512/metrics.json
vit_512/metrics.json
fusion_512/fusion_config.json
```

Placed under:

```text
outputs/
├── efficientnet_b0_512/
│   ├── checkpoints/best_model.pth
│   └── evaluation/metrics.json
├── vit_512/
│   ├── checkpoints/best_model.pth
│   └── evaluation/metrics.json
└── fusion/fusion_config.json
```

Manual download:

```powershell
python scripts/download_models.py
```

Verify:

```powershell
Test-Path outputs/efficientnet_b0_512/checkpoints/best_model.pth
Test-Path outputs/vit_512/checkpoints/best_model.pth
Test-Path outputs/fusion/fusion_config.json
```

Check the registry:

```powershell
$env:PYTHONPATH="."
python -c "from backend.app.services.model_manager import MODEL_KEYS; print(MODEL_KEYS)"
# ['fusion', 'efficientnet', 'vit', 'densenet',
#  'efficientnet-512', 'vit-512', 'fusion-512']
```

---

## 💻 Installation

Requirements: Windows 10/11, Python 3.13+, Node.js 18+, npm, Git. For GPU: NVIDIA GPU + CUDA PyTorch build. Dev environment used Python 3.13.7, PyTorch 2.10 + cu128, CUDA 12.8, RTX 4060 Laptop (8 GB).

1. **Clone**

   ```powershell
   git clone https://github.com/RushikeshTelrandhe/RESPIRA-AI-Assisted-Chest-X-Ray-Analysis.git
   cd RESPIRA-AI-Assisted-Chest-X-Ray-Analysis
   ```

2. **Python environment**

   ```powershell
   python -m venv env
   .\env\Scripts\Activate.ps1
   # if restricted: .\env\Scripts\activate
   ```

3. **Python dependencies**

   ```powershell
   pip install -r requirements-app.txt -r requirements-backend.txt
   # full research deps: pip install -r requirements.txt
   ```

4. **Frontend**

   ```powershell
   cd frontend; npm install; cd ..
   ```

5. **Environment**

   ```powershell
   copy .env.example .env
   ```

   Set `JWT_SECRET` (required in production). Never commit `.env`.

---

## 🚀 Running RESPIRA

### Recommended — one-click

```powershell
.\start_respira.bat
```

Steps: locate root → validate Python/venv/Node → install frontend deps if needed → download missing models → start FastAPI → wait for `/api/v1/health` → start Vite.

### Manual backend

```powershell
$env:PYTHONPATH="."
.\env\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

### Manual frontend

```powershell
cd frontend
npm install
npm run dev   # http://localhost:5173
```

### Tests

```powershell
python -m pytest tests/ -q        # auth, ownership, real inference/Grad-CAM/ViT/reports
cd frontend; npm run test         # vitest run
```

---

## 🔬 Research Scripts

Organized under `src/`:

| Area | Path | Contents |
|---|---|---|
| Training | `src/training/` | EfficientNet-B0, ViT-B/16 trainers |
| Evaluation | `src/evaluation/` | metrics + predictions |
| Fusion | `src/fusion/` | `probability_fusion.py`, `evaluate_fusion.py`, cross/disease/adaptive fusion |
| Explainability | `src/explainability/` | `grad_cam.py`, `vit_attention.py`, `xai_comparison.py`, artifact audit + contact sheets |
| Analysis | `src/analysis/` | model comparison, final system analysis |
| Inference | `src/inference/` | batched inference helpers |
| Features | `src/features/` | projected / fused feature dumps |

---

## 🧠 Model Inference

Selectable via `POST /analyze` (`fusion` default):

```text
Input → Preprocessing → 512×512 RGB → EfficientNet-B0 → 6-class prediction
Input → Preprocessing → 512×512 RGB → 1024 patches → ViT-B/16 → 6-class prediction
Input → Preprocessing ─┬─→ EfficientNet ─→ Probabilities ─┐
                       └─→ ViT ──────────→ Probabilities ─┴─→ Weighted Fusion → Final
```

### Why 512×512?

| Config | Patches (ViT-B/16) |
|---|---|
| 224×224 | 14 × 14 = 196 |
| 512×512 | 32 × 32 = 1024 |

More spatial tokens for the transformer, at the cost of memory/compute.

### ⚡ GPU Considerations

- ViT-512 checkpoint ≈ 1 GB; EfficientNet-512 ≈ 49 MB
- Fusion needs both resident → GPU strongly recommended; CPU inference is much slower
- Device auto-selected (`cuda` if available, else `cpu`); surfaced at `/api/v1/models/status`

---

## 🧾 Generated Outputs

```text
outputs/
├── efficientnet_b0_512/   # checkpoints/, evaluation/
├── vit_512/               # checkpoints/, evaluation/, explainability/
├── fusion/                # fusion_config.json, classification/, cross_attention/,
│                          # disease_attention/, adaptive_fusion/, test/, validation/
├── final_prediction/      # predictions/, reports/final_metrics.json, uncertainty/
├── final_analysis/  xai_comparison/  vit/  densenet/  efficientnet_b0/
```

Large `*.pth` files are git-ignored and pulled from Hugging Face.

---

## 🔐 Git / Large File Strategy

`.gitignore` excludes `*.pth`, `*.pt`, `*.ckpt`, `*.onnx`, `*.safetensors`, `*.bin`, datasets, and runtime files.

```text
GitHub: source, backend, frontend, configs, scripts, docs
   │
   ▼
Hugging Face: checkpoints, metrics, fusion configs
```

---

## 👥 Running on Another Computer

```powershell
git clone https://github.com/RushikeshTelrandhe/RESPIRA-AI-Assisted-Chest-X-Ray-Analysis.git
cd RESPIRA-AI-Assisted-Chest-X-Ray-Analysis
python -m venv env
.\env\Scripts\activate
pip install -r requirements-app.txt -r requirements-backend.txt
cd frontend; npm install; cd ..
```

Double-click `start_respira.bat` — 512 models auto-download from Hugging Face.

---

## 🧰 Troubleshooting

| Problem | Fix |
|---|---|
| Python/venv not found | `python -m venv env`, reinstall deps |
| `npm` not found | Install Node.js 18+, check `node --version` / `npm --version` |
| Frontend deps missing | `cd frontend; npm install` |
| Models missing | `python scripts/download_models.py` |
| Backend won't start | `.\env\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`, check `/api/v1/health` |
| Frontend won't start | `cd frontend; npm install; npm run dev` |
| Port 8000 in use | `netstat -ano \| findstr :8000`, stop the process |
| Port 5173 in use | `netstat -ano \| findstr :5173`, or let Vite pick another port |

---

## 🧪 Development Workflow

```text
Modify preprocessing / model / backend / frontend
        │
        ▼
Run local tests (pytest + vitest)
        │
        ▼
Validate inference + XAI outputs
        │
        ▼
Run frontend + backend
        │
        ▼
Verify end-to-end
        │
        ▼
Commit source to GitHub / upload large checkpoints to Hugging Face
```

Detailed guides: `docs/architecture.md`, `docs/setup.md`, `docs/backend.md`, `docs/frontend.md`, `docs/ml-inference.md`, `docs/explainability.md`, `docs/security.md`, `docs/deployment.md`, `docs/electron.md`, `docs/database.md`.

---

## 📚 Main Technologies

- **ML:** Python, PyTorch, Torchvision, NumPy, Pandas, scikit-learn
- **Vision:** OpenCV, CLAHE, lung masking, Grad-CAM, ViT attention
- **Models:** EfficientNet-B0, ViT-B/16, DenseNet-121, probability + cross-attention fusion
- **Backend:** FastAPI, Uvicorn, SQLAlchemy, JWT, reportlab
- **Frontend:** React 19, TypeScript, Vite, Tailwind, framer-motion, lucide-react, recharts
- **Hosting:** Hugging Face Hub; **Dev:** Git, GitHub, VS Code, Windows; **Desktop:** Electron

---

## 📖 Research Contribution

High-resolution X-ray analysis + CNN features + transformer features + probability/cross-attention fusion + XAI, with explicit evaluation of model agreement and explanation behavior — not just accuracy.

---

## ⚠️ Medical Disclaimer

For academic research, ML/CV/XAI experimentation, and Final Year Project demonstration only. Not for diagnosis, treatment, or clinical decisions. System outputs are not medical advice — a qualified professional must review every X-ray.

---

## 👨‍💻 Author

**Rushikesh Telrandhe** — B.Tech Artificial Intelligence, G. H. Raisoni College of Engineering, Nagpur

## 🔗 Project Links

- GitHub: <https://github.com/RushikeshTelrandhe/RESPIRA-AI-Assisted-Chest-X-Ray-Analysis>
- Hugging Face: <https://huggingface.co/rushikeshtelrandhe/respira-models>

## ⭐ Acknowledgement

Final Year Project spanning AI, deep learning, computer vision, medical image analysis, XAI, vision transformers, and model fusion.

## 📜 License

Academic/research use. Verify dataset and pretrained-model licenses before redistribution or commercial use.
