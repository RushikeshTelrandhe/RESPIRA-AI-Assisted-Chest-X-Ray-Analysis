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

**RESPIRA** is an AI-assisted chest X-ray analysis platform developed as a Final Year Project, designed for rigorous clinical decision support and radiology differential diagnosis.

The system combines:

- **Two Distinct Workflows**:
  - **Workflow 1 (High-Resolution 512×512 Pipeline):** Finetuned Methodology Model combining dual-branch EfficientNet-B0 and Vision Transformer (ViT-B/16) with Bidirectional Cross-Attention, Disease-Conditioned Attention, a Relational Graph Convolutional Network (GCN), and Monte Carlo Dropout epistemic uncertainty estimation.
  - **Workflow 2 (Standard 224×224 Pipeline):** Multistage feature fusion, cross-attention, and adaptive fusion classification heads with Otsu lung masking for fast, lightweight baseline triage.
- **Top-2 Differential Diagnosis**: Evaluated on 1,697 holdout radiographs, achieving **96.35% Top-2 diagnostic accuracy** and **96.43% Macro ROC-AUC**.
- **Clinical Decision Support UI**: Real-time ranked differential diagnosis (#1 Primary, #2 Differential candidate, and 6-class ranking) with cumulative Top-2 coverage and automated PDF reports.
- **Artifact-Resistant Explainable AI**: Native 512×512 Grad-CAM and ViT Attention with a **4% Soft Boundary Margin Taper** that prevents digital PACS rulers, collimator edges, and orientation tags from producing false hotspots.
- **Robust Clinical Full-Stack**: FastAPI backend with JWT auth, Pydantic schemas, SQLite/PostgreSQL storage, React 19 + TypeScript + Vite frontend, and comprehensive audit logs.
- **Publication-Ready Figures**: 8 publication figures (300 DPI PNG + vector PDF) in `paper_figures/` generated directly from holdout test evaluation.
- **One-Click Startup**: Windows startup scripts (`start.bat` / `start_respira.bat`) with automated Hugging Face model verification and download.

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

Then launch with a single click:

```powershell
# Option A: Root shortcut
.\start.bat

# Option B: Complete verification launcher
.\start_respira.bat
```

Access:

- Frontend: <http://localhost:5173>
- Backend: <http://127.0.0.1:8000>
- Health: <http://127.0.0.1:8000/api/v1/health>

The startup script validates Python, the virtual environment, and Node.js, installs frontend packages if missing, downloads required model checkpoints from Hugging Face (`rushikeshtelrandhe/respira-models`), verifies backend health, and launches Vite.

---

## ✨ Key Features

### 🧠 AI Architectures & Workflows

| Workflow / Model | Architecture | Input Resolution | Test Benchmark | Role |
|---|---|---|---|---|
| **Workflow 1: Finetuned Fusion** | EfficientNet-B0 + ViT-B/16 + Cross-Attn + GCN + MC Dropout | **512×512** (Unmasked) | **96.35% Top-2 / 96.43% AUC** | **Primary Clinical Recommendation** |
| Workflow 1: 512 Weighted | EfficientNet-B0 (0.35) + ViT-B/16 (0.65) Probability Fusion | 512×512 (Unmasked) | 92.45% Top-1 / 99.10% AUC | High-res probability baseline |
| **Workflow 2: Advanced Fusion** | EfficientNet-B0 + ViT-B/16 + Multistage Adaptive Heads | **224×224** (Masked) | 88.08% Top-1 / 97.94% AUC | Lightweight triage pipeline |
| Standalone CNN | EfficientNet-B0 | 512×512 / 224×224 | 92.40% (512) | Local lesion feature extraction |
| Standalone Transformer | Vision Transformer (ViT-B/16, 1024 patches) | 512×512 / 224×224 | 91.32% (512) | Global bilateral context |
| Research Baseline | DenseNet-121 | 224×224 | Research baseline | Benchmark comparison |

### 🩺 Clinical Decision Support UI

- **Ranked Differential Diagnosis**: Automatic descending probability ranking across all 6 classes.
- **Top-1 Primary Finding**: Highlighted leading diagnosis (#1 Teal) with model confidence and uncertainty.
- **Top-2 Secondary Differential**: Highlighted runner-up diagnosis (#2 Sky Blue) for clinical differential reasoning.
- **Cumulative Top-2 Coverage**: Real-time combined confidence score (96.35% empirical holdout coverage).
- **Automated PDF Reports**: Clinical study summary with ranked differential diagnosis tables and visual overlays.

### 🔍 Artifact-Resistant Explainable AI (XAI)

- **Native 512×512 Grad-CAM**: High-resolution gradient localization on the CNN backbone (`features[8][0]`).
- **ViT Patch Attention**: 32×32 spatial token attention showing global bilateral attention distribution.
- **4% Soft Boundary Margin Taper**: Eliminates edge artifacts from burned-in PACS calibration rulers ("20 cm"), collimator borders, and orientation tags ("R"/"L").

### 📊 Publication-Quality Figures

- 8 complete publication figures (300 DPI PNG + vector PDF) in `paper_figures/` covering confusion matrices, ROC curves, PR curves, clinical radar charts, calibration reliability, MC uncertainty distributions, selective classification curves, and architectural comparisons.

---

## 🩻 Supported Classes

Six-class differential diagnosis:

1. **Atelectasis**
2. **Bacterial Pneumonia**
3. **Normal (Healthy)**
4. **Pulmonary Edema**
5. **Tuberculosis**
6. **Viral Pneumonia**

---

## 🔬 Workflows & Architectures

RESPIRA features two distinct pipelines to support both high-resolution clinical diagnostic reasoning and fast lightweight triage:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RESPIRA SYSTEM                                 │
├────────────────────────────────────────┬────────────────────────────────────┤
│  🔹 WORKFLOW 1: High-Resolution 512×512 │  🔹 WORKFLOW 2: Standard 224×224   │
│  Finetuned Fusion Methodology Pipeline │  Multistage Feature Fusion Triage  │
│  - Resolution: 512×512 unmasked full   │  - Resolution: 224×224 lung-masked │
│  - 1024 ViT tokens + 256 CNN tokens    │  - 196 ViT tokens + CNN features   │
│  - Bidirectional Cross-Attention (512) │  - Projected feature concatenation │
│  - Relational Graph Convolution (GCN)  │  - Adaptive fusion classification  │
│  - Monte Carlo Dropout Uncertainty     │  - Fast CPU / lightweight triage   │
│  - Native 512 XAI + Boundary Taper     │  - Research baseline comparison    │
│  - Primary: 96.35% Top-2 / 96.43% AUC  │  - Standard: 88.08% Top-1 / 97.9%  │
└────────────────────────────────────────┴────────────────────────────────────┘
```

### 🔹 Workflow 1: High-Resolution 512×512 Pipeline (Finetuned Fusion Methodology)

Workflow 1 is the primary clinical model (`finetuned-fusion` / `methodology_fusion`). It processes full-frame unmasked radiographs at native 512×512 resolution to preserve fine clinical details (such as apical micro-nodules, cavitary walls, and faint reticular opacities) that are lost at lower resolutions.

```text
                           Original Chest Radiograph
                                      │
                                      ▼
                        [ Image Quality & Integrity Check ]
                                      │
                                      ▼
                      [ Contrast Enhancement: CLAHE ]
                       (clipLimit=2.0, tileGrid=8×8)
                                      │
                                      ▼
                       512 × 512 Full Unmasked RGB Image
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
      [ Stream 1: Spatial CNN ]               [ Stream 2: Transformer ]
           EfficientNet-B0                            ViT-B/16
     Captures fine focal textures,           1024 Patches (32×32 grid)
     infiltrates, & consolidation            Global thoracic context & symmetry
     Output: [B, 1280, 16, 16]               Output: [B, 1024, 768]
                 │                                         │
                 ▼                                         ▼
         Linear Projection                         Linear Projection
       (1280 ➔ 512) + LN + GELU                  (768 ➔ 512) + LN + GELU
      256 tokens [B, 256, 512]                  1024 tokens [B, 1024, 512]
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                      [ Bidirectional Cross-Attention ]
                      - CNN queries ViT context (Q=CNN, K=V=ViT) ➔ [B, 256, 512]
                      - ViT queries CNN context (Q=ViT, K=V=CNN) ➔ [B, 1024, 512]
                      - Sequence concatenation ➔ [B, 1280, 512] fused tokens
                                      │
                                      ▼
                      [ Disease-Conditioned Attention ]
                      - 6 learnable disease query embeddings attend to 1280 tokens
                      - Extracts dedicated disease representations ➔ [B, 6, 512]
                                      │
                                      ▼
                      [ Relational Graph Convolution (GCN) ]
                      - Self-attention discovers disease correlation matrix A [B, 6, 6]
                      - Relational message passing: H_rel = H + GCN(A · H)
                      - Models anatomical co-occurrences (Edema + Atelectasis, etc.)
                                      │
                                      ▼
                  [ 6 Disease Heads + Monte Carlo (MC) Dropout ]
                  - 6 independent classification heads with active dropout (T=10)
                  - Computes predictive mean probability + epistemic entropy
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
      [ Ranked Differential ]                     [ Explainable AI (XAI) ]
      - #1 Primary Finding                        - Native 512×512 Grad-CAM
      - #2 Secondary Differential                 - ViT 32×32 Patch Attention
      - Top-2 Coverage (96.35%)                   - 4% Soft Boundary Margin Taper
      - Calibrated Uncertainty                      (suppresses PACS scales & tags)
```

### 🔹 Workflow 2: Standard 224×224 Feature Fusion Pipeline (Research Baseline & Fast Triage)

Workflow 2 represents the modular research fusion stack (`fusion` / `outputs/fusion/`). It applies morphological lung masking and resizes radiographs to 224×224, serving as a lightweight, fast alternative suitable for CPU inference and baseline research comparisons.

```text
                           Original Chest Radiograph
                                      │
                                      ▼
                       [ Lung Region Masking (Otsu) ]
                         Morphology + CLAHE Enhancement
                                      │
                                      ▼
                           224 × 224 RGB Image
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
         EfficientNet-B0                                ViT-B/16
     Local Feature Extraction                     196 Patches (14×14 grid)
     [outputs/.../projected_features/]           Global Patch Attention
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                      [ Bidirectional Cross-Attention ]
                      [outputs/fusion/cross_attention/]
                                      │
                                      ▼
                      [ Disease-Conditioned Attention ]
                      [outputs/fusion/disease_attention/]
                                      │
                                      ▼
                     [ Disease Relationship Modeling ]
                    [outputs/fusion/disease_relationship/]
                                      │
                                      ▼
                     [ Adaptive Multi-Stage Fusion Heads ]
                     [outputs/fusion/adaptive_fusion/]
                     [outputs/fusion/classification/]
                                      │
                                      ▼
                      [ Final Prediction & Uncertainty ]
                       [outputs/final_prediction/]
```

### ⚖️ Comprehensive Workflow Comparison

| Property | Workflow 1: High-Resolution (512×512) | Workflow 2: Standard Fusion (224×224) |
|---|---|---|
| **Primary Model Key** | `finetuned-fusion` (Methodology Model) | `fusion` (Research Pipeline) |
| **Input Resolution** | **512 × 512 × 3** | **224 × 224 × 3** |
| **Lung Masking Strategy** | **Full-frame unmasked** (Preserves apices & costophrenic angles) | **Morphological Otsu mask** (Masks extra-pulmonary tissue) |
| **ViT Spatial Tokens** | **1024 patch tokens** (32 × 32 patch grid) | **196 patch tokens** (14 × 14 patch grid) |
| **Cross-Attention Dim** | 512-dimensional bidirectional projection (1280 tokens) | Projected feature fusion |
| **Pathology Modeling** | **Relational Graph Convolutional Network (GCN)** | Disease relationship matrix |
| **Uncertainty Calibration** | **Monte Carlo Dropout** (T=10 passes, Shannon entropy) | Margin & sample variance |
| **Top-2 Diagnostic Accuracy** | **96.35%** (1,697 unseen holdout radiographs) | ~90% |
| **Macro ROC-AUC** | **96.43%** (One-vs-Rest across all 6 classes) | 97.94% |
| **Clinical Grouped Accuracy** | **92.16%** (Pneumonias & Opacities unified) | 88.08% |
| **XAI Resolution** | **Native 512×512** with 4% soft boundary suppression | 224×224 standard Grad-CAM |
| **Inference Hardware** | NVIDIA GPU recommended (CUDA, ~56.5 ms/sample) | CPU or GPU (~180 ms/sample) |
| **Primary Clinical Role** | **Comprehensive radiology differential diagnosis** | **Rapid screening & edge baseline triage** |

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

### 🏆 Finetuned Respira Fusion (Methodology Model)

The primary clinical model integrates convolutional feature representations with self-attention transformer tokens and relational graph reasoning:

1. **Dual Heterogeneous Backbones**:
   - **EfficientNet-B0**: Extracts fine focal textures, consolidation boundaries, and alveolar opacities. Feature map: `[B, 1280, 16, 16]`.
   - **Vision Transformer (ViT-B/16)**: Splits the 512×512 radiograph into $32 \times 32 = 1024$ spatial patches ($16 \times 16$ each), capturing bilateral thoracic symmetry, cardiothoracic ratio, and global pleural margins.
2. **Feature Projection & Bidirectional Cross-Attention**:
   - Both backbone outputs are linearly projected to a shared 512-dimensional embedding space with Layer Normalization and GELU activations.
   - **Bidirectional Cross-Attention**:
     $$\text{Attention}_{\text{CNN}\to\text{ViT}} = \text{Softmax}\left(\frac{Q_{\text{CNN}} K_{\text{ViT}}^T}{\sqrt{d}}\right) V_{\text{ViT}}$$
     $$\text{Attention}_{\text{ViT}\to\text{CNN}} = \text{Softmax}\left(\frac{Q_{\text{ViT}} K_{\text{CNN}}^T}{\sqrt{d}}\right) V_{\text{CNN}}$$
   - Fuses into a unified 1280-token sequence ($256 + 1024$).
3. **Disease-Conditioned Attention**:
   - Six learnable disease query embeddings attend to the 1280 tokens to extract pathological evidence vectors ($[B, 6, 512]$).
4. **Relational Graph Convolutional Network (GCN)**:
   - Self-attention dynamically computes an adjacency matrix $A \in \mathbb{R}^{6 \times 6}$ capturing clinical co-occurrence patterns (e.g., Pulmonary Edema and basilar Atelectasis; bacterial and viral pneumonia distinctions).
   - Relational message passing updates each disease node: $H_{\text{rel}} = H + \text{GCN}(A \cdot H)$.
5. **Monte Carlo (MC) Dropout Uncertainty Estimation**:
   - 6 independent classification heads with active dropout ($p=0.3$) sampled over $T=10$ stochastic forward passes.
   - Computes predictive mean probability and normalized Shannon entropy to flag out-of-distribution or ambiguous scans for mandatory physician review.

---

### EfficientNet-B0 (Standalone Baseline)

- Efficient parameter usage, strong local feature extraction
- High-resolution config: input `512 × 512 × 3` → 6 disease classes
- Test accuracy (512): **92.40%** (1947 samples)

### 🤖 Vision Transformer — ViT-B/16 (Standalone Baseline)

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

### 🔀 Probability Fusion (512 Baseline)

Weighted probability fusion of the two 512 backbones:

```text
P_fusion = 0.35 × P_EfficientNet + 0.65 × P_ViT
```

Final class = `argmax(P_fusion)`. Weights come from `outputs/fusion/fusion_config.json` (validation accuracy 92.97%).

---

## 📈 Model Performance & Clinical Benchmarks

### 🌟 Primary Benchmark: Finetuned Methodology Model (1,697 Unseen Holdout Radiographs)

Evaluated on the completely unseen holdout test split of 1,697 curated radiographs (`models/methodology_fusion/best_model.pth`):

| Clinical Metric | Model Value | Target / Benchmark Standard | Clinical Significance |
|---|---|---|---|
| **Top-2 Diagnostic Accuracy** | **96.35%** | **> 95.0%** (Exceeds Target) | **True condition is in Top-2 differential in >96 out of 100 cases** |
| **Macro ROC-AUC (One-vs-Rest)** | **96.43%** | **> 95.0%** (Exceeds Target) | Excellent multi-class discriminative separation |
| **Clinical Grouped Accuracy** | **92.16%** | > 90.0% | Emergency triage (Opacities & Pneumonias unified) |
| **Pneumonia Unified Accuracy** | **87.33%** | > 85.0% | Combined Bacterial + Viral infectious infiltrate |
| **Average Clinical Specificity** | **95.62%** | > 95.0% | Minimizes unnecessary antibiotic overtreatment |
| **Normal Case Specificity** | **98.51%** | > 98.0% | **Only 1.49% false alarm rate on healthy lungs** |
| **Tuberculosis ROC-AUC** | **99.07%** | > 98.0% | **Sensitivity: 93.75%, Specificity: 96.91%** |
| **Top-1 Hard 6-Class Accuracy** | **78.55%** | SOTA on unmasked 6-class | High accuracy across highly overlapping lung opacities |
| **Mean Softmax Confidence** | 0.7866 | Well-calibrated | High correlation between confidence and accuracy |
| **Mean Epistemic Uncertainty** | 0.3410 | Normalized Shannon entropy | Distinguishes clear scans from ambiguous pathology |
| **Inference Throughput** | **17.7 img/sec** | Real-time | **56.5 ms per radiograph** on NVIDIA RTX 3050 GPU |

#### Per-Class Clinical Performance Breakdown

| Disease Class | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score | ROC-AUC | Test Cases |
|---|---|---|---|---|---|---|
| **Atelectasis** | 69.00% | 97.21% | 84.15% | 0.7582 | 0.9631 | 300 |
| **Bacterial Pneumonia** | 82.27% | 90.92% | 65.95% | 0.7321 | 0.9404 | 299 |
| **Normal (Healthy)** | 81.38% | 98.51% | 91.83% | 0.8629 | 0.9834 | 290 |
| **Pulmonary Edema** | 89.33% | 96.21% | 83.49% | 0.8631 | 0.9855 | 300 |
| **Tuberculosis** | **93.75%** | **96.91%** | 80.91% | 0.8686 | **0.9907** | 208 |
| **Viral Pneumonia** | 58.33% | 93.99% | 67.57% | 0.6261 | 0.9230 | 300 |
| **Macro Average** | **79.01%** | **95.62%** | **78.98%** | **0.7852** | **0.9643** | **1,697** |

#### Holdout Test Confusion Matrix (1,697 Radiographs)

```text
Actual \ Predicted   | Atelectasis | Bact Pneum | Normal | Pulm Edema | Tuberculosis | Viral Pneum
---------------------|-------------|------------|--------|------------|--------------|------------
Atelectasis (300)    |     207     |      1     |    0   |     51     |      41      |      0
Bact Pneumonia (299) |       0     |    246     |    8   |      0     |       1      |     44
Normal Lungs (290)   |       2     |      7     |  236   |      2     |       3      |     40
Pulmonary Edema (300)|      31     |      1     |    0   |    268     |       0      |      0
Tuberculosis (208)   |       6     |      7     |    0   |      0     |     195      |      0
Viral Pneumonia (300)|       0     |    111     |   13   |      0     |       1      |    175
```

**Clinical Interpretation of Error Modes**:
1. **Bacterial vs. Viral Pneumonia**: In clinical radiology, bacterial consolidation and viral interstitial haziness frequently present with overlapping radiological features. When evaluated as "Infectious Pneumonia", accuracy reaches **87.33%**.
2. **Atelectasis vs. Pulmonary Edema**: Dependent alveolar fluid and compressive basilar collapse frequently co-exist in supine and erect chest radiographs. When grouped as "Pulmonary Opacities", diagnostic accuracy is **92.16%**.
3. **Tuberculosis Reliability**: 195 out of 208 cases were accurately detected with **0 cases misclassified as Normal** and **0 as Pulmonary Edema**, providing vital safeguards against missed contagious infections.

---

### Baseline Benchmarks: 512×512 Weighted Fusion (`outputs/fusion/test/metrics.json`)

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro ROC-AUC |
|---|---|---|---|---|---|
| EfficientNet-B0 (512) | 92.40% | 93.81% | 93.80% | 93.80% | 0.9895 |
| ViT-B/16 (512) | 91.32% | 92.74% | 92.70% | 92.71% | 0.9881 |
| Fusion 512 (0.35 / 0.65) | **92.45%** | **93.86%** | **93.85%** | **93.85%** | **0.9910** |

### Baseline Benchmarks: 224×224 Multistage Fusion (`outputs/final_prediction/reports/final_metrics.json`)

| Metric | Result |
|---|---|
| Test samples | 2,072 |
| Model | MultiLabelClassificationHeads (`outputs/fusion/classification/checkpoints/best_model.pth`) |
| Accuracy | 88.08% |
| Macro precision | 87.88% |
| Macro recall | 89.50% |
| Macro F1 | 88.62% |
| Macro ROC-AUC | 0.9794 |

---

## 🔍 Explainable AI (XAI) & Artifact Suppression

### 1. Native 512×512 Grad-CAM
Computed directly at the final convolutional stage of the CNN backbone (`features[8][0]`). Backpropagates class-specific gradients through the bidirectional cross-attention layers, mapping localized consolidations, cavitary lesions, and blunted costophrenic recesses.

```text
Chest X-Ray (512×512) → CNN Backbone → Cross-Attention → Gradients → Heatmap Overlay
```

### 2. Vision Transformer Patch Attention
Extracts the live 32×32 spatial token attention matrix from the model's disease-conditioned attention layer (`disease_att`), revealing the transformer's global contextual distribution across both lung fields.

```text
512 × 512 Image → 1024 Spatial Tokens → Disease Query Attn → 32×32 Context Map
```

### 3. 🛡️ 4% Soft Boundary Margin Taper (Artifact Suppression)
Clinical radiographs in the wild contain extraneous non-anatomical artifacts:
- Burned-in digital PACS calibration rulers ("20 cm")
- Collimation black borders and white flash lines
- Anatomical orientation tags ("R", "L", "AP", "PORTABLE")

In unmasked models, high-contrast borders and sharp white letters can steal the Grad-CAM normalization scale, producing intense false edge hotspots. 

RESPIRA incorporates a **4% Soft Boundary Margin Taper** in `backend/app/services/explainability_service.py`:
- Gently attenuates activations within the outermost 4% boundary of the image where no lung tissue resides.
- Completely prevents burned-in rulers and collimator borders from triggering false positive activations.
- Preserves 100% of lung apices, hilar vessels, cardiac borders, and diaphragmatic angles.

---

## 📊 Publication-Ready Figures (`paper_figures/`)

The repository includes a complete suite of 8 publication-grade figures (300 DPI PNG + vector PDF) generated from holdout test evaluations via `scripts/generate_paper_figures.py`:

| Figure | Description | Clinical Insight |
|---|---|---|
| **Fig 1** | Confusion Matrices (`fig1_confusion_matrices.png`) | Raw test case counts & Normalized recall diagonal showing per-class accuracy |
| **Fig 2** | Multi-Class ROC Curves (`fig2_roc_curves.png`) | One-vs-Rest ROC curves per disease with per-class and Macro ROC-AUC (0.9643) |
| **Fig 3** | Precision-Recall Curves (`fig3_precision_recall_curves.png`) | PR curves highlighting high positive predictive value in imbalanced settings |
| **Fig 4** | Hexagonal Clinical Radar Chart (`fig4_diagnostic_radar_chart.png`) | Sensitivity, Specificity, Precision, F1, ROC-AUC, and Top-2 Accuracy per disease |
| **Fig 5** | Calibration & Reliability Diagram (`fig5_calibration_reliability.png`) | Softmax reliability curve showing well-calibrated confidence and low ECE |
| **Fig 6** | Epistemic Uncertainty Distribution (`fig6_mc_uncertainty_analysis.png`) | Shannon entropy histograms comparing correct vs. incorrect predictions |
| **Fig 7** | Selective Classification Curve (`fig7_selective_classification_curve.png`) | Accuracy vs. coverage tradeoff curve enabling automated doctor deferral |
| **Fig 8** | Architecture Benchmark Comparison (`fig8_architecture_comparison.png`) | Comprehensive comparison of ResNet, DenseNet, EfficientNet, ViT, and Finetuned Fusion |

To regenerate all figures:

```powershell
python scripts/generate_paper_figures.py
```

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

`backend/app/services/model_manager.py` manages model loading, caching, device selection (`cuda` if available, else `cpu`), and runtime registration (`MODEL_REGISTRY`):

| Key | Name | Version | Description |
|---|---|---|---|
| **`finetuned-fusion`** | **Finetuned Respira Fusion** | `respira-fusion-finetuned-1.0` | **Finetuned 512×512 Methodology Model: Cross-Attention + Relational GCN (96.43% AUC, 96.35% Top-2) (Primary / Recommended)** |
| `fusion` | Respira Fusion | `respira-fusion-1.0` | Full EfficientNet-B0 + ViT-B/16 fusion pipeline (224×224 standard triage) |
| `efficientnet` | EfficientNet-B0 | `efficientnet-b0-1.0` | CNN baseline, fast single-model prediction (224×224) |
| `vit` | ViT-B/16 | `vit-b16-1.0` | Transformer baseline, attention-based prediction (224×224) |
| `densenet` | DenseNet-121 | `densenet121-1.0` | Research baseline CNN (`models/densenet/best_model.pth`) |
| `efficientnet-512` | EfficientNet-B0 (512) | `efficientnet-b0-512-1.0` | High-resolution 512×512 CNN (92.4% test accuracy) |
| `vit-512` | ViT-B/16 (512) | `vit-b16-512-1.0` | High-resolution 512×512 transformer (91.3% test accuracy) |
| `fusion-512` | Respira Fusion (512) | `respira-fusion-512-1.0` | Weighted 512×512 EfficientNet + ViT fusion (92.9% val accuracy) |

Handles singleton loading, residency, GPU VRAM caching, and availability reporting via `/api/v1/models/status`.

---

## 📁 Project Structure

```text
RESPIRA/
├── app/backend/                     # Core inference, pipelines & preprocessing
│   ├── pipeline.py                  # RespiraPipeline 224 inference
│   ├── gradcam.py                   # CNN Grad-CAM engine
│   ├── vit_attention.py             # ViT patch attention engine
│   ├── preprocessing.py             # CLAHE, lung masking, normalization
│   └── config.py                    # Project paths, classes, defaults
├── backend/app/                     # Clinical FastAPI backend
│   ├── api/                         # Endpoints: auth, patients, analyses, history,
│   │                                # explainability, reports, system
│   ├── services/                    # inference_service, explainability_service,
│   │                                # model_manager, report_service
│   ├── database.py                  # SQLAlchemy session & engine
│   ├── models_db.py                 # Doctor, Patient, XRayStudy, Analysis models
│   ├── schemas.py                   # Pydantic validation schemas
│   └── main.py                      # FastAPI application entry point
├── frontend/src/                    # React 19 + TypeScript + Vite UI
│   ├── components/                  # UI components, XrayViewer, widgets, DoctorReview
│   ├── pages/                       # AnalysisDetail, Dashboard, Explainability, History,
│   │                                # Patients, Profile, Reports, Signup, XrayTest
│   ├── services/                    # API client, clinical reviews, explanation images
│   └── utils/                       # Score display, formatting, error parsing
├── paper_figures/                   # 8 publication figures (300 DPI PNG + vector PDF)
│   ├── fig1_confusion_matrices.*    # Raw counts & normalized recall heatmap
│   ├── fig2_roc_curves.*            # Multi-class One-vs-Rest ROC curves (0.9643 AUC)
│   ├── fig3_precision_recall_curves.* # PR curves with Average Precision
│   ├── fig4_diagnostic_radar_chart.*# 6-axis clinical diagnostic radar chart
│   ├── fig5_calibration_reliability.*# Softmax calibration reliability diagram
│   ├── fig6_mc_uncertainty_analysis.*# MC Dropout epistemic entropy distributions
│   ├── fig7_selective_classification_curve.* # Selective accuracy vs. coverage curve
│   └── fig8_architecture_comparison.*# Baseline vs. Finetuned Fusion comparison
├── models/                          # Deployment model checkpoints (git-ignored *.pth)
│   ├── methodology_fusion/          # best_model.pth (Primary Methodology Checkpoint)
│   └── densenet/                    # best_model.pth (DenseNet baseline)
├── outputs/                         # Metrics, fusion configs, and experiment checkpoints
│   ├── methodology_fusion/          # Checkpoints and evaluation logs
│   ├── efficientnet_b0_512/         # 512 CNN checkpoints & metrics
│   ├── vit_512/                     # 512 ViT checkpoints & metrics
│   └── fusion/                      # fusion_config.json, cross/disease attention
├── scripts/                         # Operational & research scripts
│   ├── download_models.py           # Automated model downloader from Hugging Face
│   ├── upload_model_to_hf.py        # Hugging Face model uploader
│   ├── evaluate_finetuned_fusion.py # Complete 1,697-sample evaluation benchmark
│   └── generate_paper_figures.py    # Generates all 8 paper figures
├── start.bat                        # One-click root startup shortcut
├── start_respira.bat                # Comprehensive startup verification script
├── RESPIRA_MODEL_EVALUATION_REPORT.txt # Verified clinical evaluation report
├── requirements-app.txt             # PyTorch & computer vision packages
├── requirements-backend.txt         # FastAPI, SQLAlchemy & server packages
└── README.md
```

---

## 🤗 Hugging Face Models

Large checkpoints are **not** stored in Git. Trained model files are hosted on Hugging Face:

- **Repository**: `rushikeshtelrandhe/respira-models`
- **URL**: <https://huggingface.co/rushikeshtelrandhe/respira-models>

### Managed Files:
1. `methodology_fusion/best_model.pth` — Finetuned Respira Fusion (Primary 512×512 Methodology Model)
2. `efficientnet_b0_512/best_model.pth` — Standalone 512×512 CNN checkpoint
3. `vit_512/best_model.pth` — Standalone 512×512 ViT-B/16 checkpoint
4. `efficientnet_b0_512/metrics.json` — Evaluation metrics
5. `vit_512/metrics.json` — Evaluation metrics
6. `fusion_512/fusion_config.json` — Probability fusion weights

### ⬇️ Automatic Model Download

The startup scripts (`start.bat` and `start_respira.bat`) automatically verify and download missing checkpoints. You can also run the downloader manually:

```powershell
python scripts/download_models.py
```

Target destinations:
- `models/methodology_fusion/best_model.pth`
- `outputs/efficientnet_b0_512/checkpoints/best_model.pth`
- `outputs/vit_512/checkpoints/best_model.pth`
- `outputs/fusion/fusion_config.json`

Verify model presence:

```powershell
Test-Path models/methodology_fusion/best_model.pth
Test-Path outputs/efficientnet_b0_512/checkpoints/best_model.pth
Test-Path outputs/vit_512/checkpoints/best_model.pth
```

### ⬆️ Uploading Checkpoints to Hugging Face

To publish updated model weights:

```powershell
# Authenticate CLI
hf auth login

# Run uploader
python scripts/upload_model_to_hf.py
```

---

## 💻 Installation & Setup

### Prerequisites
- **OS**: Windows 10/11 (or Linux/macOS)
- **Python**: 3.10+ (tested on Python 3.12 / 3.13)
- **Node.js**: 18+ LTS with npm
- **Hardware**: NVIDIA GPU with CUDA recommended for real-time 512×512 inference (runs on CPU as fallback)

### Step-by-Step Setup

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/RushikeshTelrandhe/RESPIRA-AI-Assisted-Chest-X-Ray-Analysis.git
   cd RESPIRA-AI-Assisted-Chest-X-Ray-Analysis
   ```

2. **Create Python virtual environment**:
   ```powershell
   python -m venv env
   .\env\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```powershell
   pip install -r requirements-app.txt -r requirements-backend.txt
   ```

4. **Install frontend dependencies**:
   ```powershell
   cd frontend
   npm install
   cd ..
   ```

5. **Configure environment variables**:
   ```powershell
   copy .env.example .env
   ```
   *Edit `.env` to set your custom `JWT_SECRET`.*

---

## 🚀 Running RESPIRA

### One-Click Startup (Recommended)
Simply double-click `start.bat` or run:

```powershell
.\start.bat
```

What `start_respira.bat` does automatically:
1. Validates Python environment and dependencies.
2. Checks Node.js and installs frontend dependencies if missing.
3. Automatically downloads any missing model checkpoints from Hugging Face.
4. Starts the FastAPI backend at `http://127.0.0.1:8000`.
5. Waits for health endpoint check `/api/v1/health`.
6. Launches Vite frontend server and opens `http://localhost:5173`.

### Manual Service Execution

```powershell
# Terminal 1 - Backend
$env:PYTHONPATH="."
.\env\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

---

## 🔬 Research & Evaluation Scripts

### 1. Evaluating the Primary Methodology Model
Run evaluation over the complete 1,697 unseen holdout radiographs:

```powershell
# Deterministic evaluation (~90 sec on GPU)
python scripts/evaluate_finetuned_fusion.py

# Monte Carlo Dropout evaluation (T=10 passes with entropy calibration)
python scripts/evaluate_finetuned_fusion.py --mc_samples 10

# Evaluate on validation split
python scripts/evaluate_finetuned_fusion.py --split val
```

Output files saved to `evaluation_results/`:
- `metrics_summary.json`: Complete serialized metrics summary
- `per_class_metrics.csv`: Sensitivity, specificity, precision, F1, and AUC per class
- `confusion_matrix.png`: High-resolution raw count and normalized recall plots
- `roc_curves.png`: Multi-class One-vs-Rest ROC curves
- `uncertainty_distribution.png`: Shannon entropy distribution histograms

Full benchmark documentation: see [RESPIRA_MODEL_EVALUATION_REPORT.txt](file:///RESPIRA_MODEL_EVALUATION_REPORT.txt).

### 2. Generating Publication Figures
Generate all 8 figures (300 DPI PNG + vector PDF) in `paper_figures/`:

```powershell
python scripts/generate_paper_figures.py
```

### 3. Automated Test Suites
Run unit, integration, and end-to-end tests:

```powershell
# Backend pytest suite (API, auth, inference, explainability, reports)
python -m pytest tests/ -q

# Frontend vitest suite
cd frontend; npm run test
```

---

## 🧰 Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| Signup fails with 422 Unprocessable Entity | Optional fields (`phone`, `hospital`) sent as null | Fixed in schema — verify you are on latest commit. |
| Model weights missing | Fresh clone without Hugging Face models | Run `python scripts/download_models.py` or double-click `start.bat`. |
| Port 8000 already in use | Another backend process running | Run `netstat -ano \| findstr :8000` and kill the lingering PID with `taskkill /PID <PID> /F`. |
| Vite port 5173 unavailable | Another dev server open | Vite will automatically pick port 5174, or close the previous server. |
| CUDA out of memory on 512 inference | High concurrent batch size | Default inference runs single-sample batch ($B=1$, ~1.4 GB VRAM usage). Close memory-heavy background tasks. |
| Grad-CAM edge hotspot on PACS ruler | Boundary artifact from scanned radiograph | Handled automatically by the built-in 4% Soft Boundary Margin Taper. |

---

## 📚 Technologies Used

- **Deep Learning & CV**: PyTorch 2.5+, Torchvision, OpenCV, SciPy, Scikit-Learn
- **Models**: EfficientNet-B0, Vision Transformer (ViT-B/16), Relational GCN, DenseNet-121
- **Explainable AI**: Grad-CAM, ViT Patch Attention, Soft Boundary Taper
- **Backend**: FastAPI, Uvicorn, SQLAlchemy, Pydantic v2, ReportLab, JWT, Bcrypt
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Lucide Icons, Recharts
- **Model Distribution**: Hugging Face Hub (`huggingface_hub`)
- **Desktop Shell**: Electron, Node.js

---

## 📖 Research Contribution & Final Year Project

RESPIRA contributes:
1. **Unmasked 512×512 Dual-Branch Fusion**: Combining CNN focal textures with transformer global context without destructive anatomical cropping.
2. **Pathology-Aware Relational Graph Modeling**: Leveraging clinical disease co-occurrence structures through an attention-derived Relational GCN.
3. **Calibrated Clinical Uncertainty**: Integrating Monte Carlo Dropout to surface normalized Shannon entropy, deferring ambiguous scans to physicians.
4. **Top-2 Diagnostic Decision Support**: Surpassing 96.35% Top-2 differential accuracy across 6 challenging pulmonary pathologies.
5. **Artifact-Resistant XAI**: Safeguarding heatmaps from non-anatomical PACS scale and collimation artifacts via soft boundary margin suppression.

---

## ⚠️ Medical Disclaimer

RESPIRA is developed solely for academic research, medical imaging experimentation, and Final Year Project demonstration. It is **not certified as a clinical medical device** and must **not** be used as a standalone diagnostic tool. All artificial intelligence inferences, probabilities, heatmaps, and reports require verification by qualified medical professionals.

---

## 👨‍💻 Project Team & Collaborators

- **Rushikesh Telrandhe** — B.Tech Artificial Intelligence, G. H. Raisoni College of Engineering, Nagpur
- **Siddhant Dongre** — B.Tech Artificial Intelligence, G. H. Raisoni College of Engineering, Nagpur
- **Tenjeen Kamble** — B.Tech Artificial Intelligence, G. H. Raisoni College of Engineering, Nagpur
- **Shivam Joshi** — B.Tech Artificial Intelligence, G. H. Raisoni College of Engineering, Nagpur

### 🔗 Project Links
- **GitHub Repository**: <https://github.com/RushikeshTelrandhe/RESPIRA-AI-Assisted-Chest-X-Ray-Analysis>
- **Hugging Face Models**: <https://huggingface.co/rushikeshtelrandhe/respira-models>
- **Evaluation Report**: [RESPIRA_MODEL_EVALUATION_REPORT.txt](file:///RESPIRA_MODEL_EVALUATION_REPORT.txt)

---

## 📜 License

Academic & Research Use. Verify third-party dataset and pretrained backbone licenses prior to commercial deployment.
