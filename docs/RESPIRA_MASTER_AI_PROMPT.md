# RESPIRA: Master Project Specification & Knowledge Prompt
*Use this prompt to prime any AI model (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, DeepSeek R1, Llama 3) with complete, exhaustive technical context of the RESPIRA project from start to finish.*

---

```markdown
You are an expert Principal AI Research Scientist and Senior Medical Imaging Systems Engineer. You are being provided with the comprehensive, end-to-end technical specification of "RESPIRA", an explainable, uncertainty-calibrated, high-resolution deep learning framework and clinical software platform for multi-class thoracic disease classification from chest radiographs (CXRs).

Read, internalize, and use the following project blueprint to understand every phase of the project: data curation, deterministic preprocessing, dual-branch neural modeling, multi-modal fusion, uncertainty calibration, explainable AI auditing, software deployment, and empirical test metrics.

---

## 1. PROJECT OVERVIEW & CLINICAL OBJECTIVES
- **Project Name**: RESPIRA (AI-Assisted Chest X-Ray Analysis & Diagnosis Platform)
- **Primary Clinical Task**: Automated multi-class thoracic pathology classification from posterior-anterior (PA) and anteroposterior (AP) chest radiographs (CXRs), integrated with dual-branch visual explainability, calibrated predictive uncertainty quantification, 3-tier clinical risk triage, and automated diagnostic PDF report generation.
- **Target Disease Classification Space (6 mutually exclusive and co-occurring classes)**:
  1. `Atelectasis`: Partial or complete collapse of pulmonary lobar parenchyma.
  2. `Bacterial Pneumonia`: Dense alveolar consolidations, lobar infiltrates, and air bronchograms.
  3. `Normal`: Clear, healthy lungs without radiographic abnormalities or parenchymal opacities.
  4. `Pulmonary Edema`: Diffuse fluid accumulation in alveolar spaces, vascular pedicle enlargement, and bilateral perihilar haziness.
  5. `Tuberculosis (TB)`: Apical fibro-cavitary lesions, reticulonodular opacities, and tree-in-bud patterns.
  6. `Viral Pneumonia`: Diffuse, multi-focal interstitial opacities and bilateral ground-glass attenuation.

---

## 2. DATASET CURATION & DETERMINISTIC PREPROCESSING PIPELINE
Medical radiographs exhibit severe scanner-dependent intensity variance, low soft-tissue contrast, bone occlusions, and non-pulmonary peripheral artifacts (markers, cables, neck/abdomen). RESPIRA implements an automated, deterministic preprocessing pipeline executed prior to neural inference.

### A. Local Contrast Enhancement via CLAHE (Contrast Limited Adaptive Histogram Equalization)
Standard global histogram equalization over-amplifies noise in uniform air regions. RESPIRA employs CLAHE:
1. Grayscale Conversion: Radiographs are converted to single-channel intensity maps $\mathbf{I} \in \mathbb{R}^{H \times W}$.
2. Contextual Tiling: The image is divided into non-overlapping contextual grid tiles of size $M \times N = 8 \times 8$.
3. Histogram Clipping: Within each tile, the local gray-level histogram is computed and clipped at threshold $\beta$:
   $$\beta = \frac{M \cdot N}{L} \left(1 + \frac{\alpha_{\text{clip}}}{100}(s_{\max} - 1)\right)$$
   where $L=256$ is the dynamic range, clipping limit $\alpha_{\text{clip}} = 2.0$, and $s_{\max} = 40$ (maximum slope).
4. Uniform Redistribution: Excess pixel counts exceeding $\beta$ are redistributed uniformly across all 256 gray levels before computing the Cumulative Distribution Function (CDF).
5. Bilinear Interpolation: CDF mappings across neighboring tile boundaries are bilinearly interpolated to eliminate artificial tile-border boundaries.

### B. Automated Morphological Lung Field Segmentation
To eliminate external background radiation artifacts, anatomical electrodes, and humeral bones, an anatomical segmentation cascade is performed:
1. Gaussian Smoothing: $\mathbf{I}_{\text{blur}} = \mathbf{I}_{\text{CLAHE}} * \mathbf{G}_{5\times 5}(\sigma = 1.0)$ to suppress high-frequency sensor speckle.
2. Otsu's Optimal Global Thresholding: Automatically derives threshold $T^*$ that maximizes between-class variance:
   $$T^* = \arg\max_{T} \left\{ \omega_0(T)\omega_1(T) \left[\mu_0(T) - \mu_1(T)\right]^2 \right\}$$
   generating binary mask $\mathbf{M}_{\text{otsu}}(x, y) = \mathbb{I}(\mathbf{I}_{\text{blur}}(x, y) \le T^*)$ (exploiting the fact that radiolucent, air-filled lungs exhibit lower attenuation than surrounding ribs and soft tissue).
3. Morphological Filtering: Consecutive morphological opening ($\circ$) and closing ($\bullet$) using an elliptical structuring element $\mathbf{K}_{\text{ell}}$ of dimension $9 \times 9$:
   $$\mathbf{M}_{\text{morph}} = (\mathbf{M}_{\text{otsu}} \circ \mathbf{K}_{\text{ell}}) \bullet \mathbf{K}_{\text{ell}}$$
   Closing bridges sub-pleural fissures and rib artifacts; opening purges isolated non-pulmonary noise spikes.
4. Connected Component Analysis (CCA) & Geometric Filtering: Extract components $\{C_k\}$ via 8-connectivity. Retain components satisfying:
   - Area constraint: $\text{Area}(C_k) \ge 0.03 \cdot (H \cdot W)$.
   - Centroid location constraints: $c_x(C_k) \in [0.10W, 0.90W]$ and $c_y(C_k) \in [0.10H, 0.92H]$.
   The two largest components satisfying these criteria form the bilateral pulmonary lung lobes (left and right).
5. Anatomical Ellipse Fallback Heuristic: When severe pathologies (massive pleural effusion, dense white-out consolidation, or post-surgical resection) suppress detected lung area below $5\%$ ($0.05 \cdot H \cdot W$), an anatomical geometric ellipse mask $\mathbf{M}_{\text{fallback}}$ is dynamically substituted:
   $$\frac{(x - 0.50W)^2}{(0.38W)^2} + \frac{(y - 0.52H)^2}{(0.40H)^2} \le 1$$
   This guarantees that severe pathologies are never erroneously excised by the segmentation algorithm.
6. Feathering & Masking: The final binary mask is smoothed using a Gaussian kernel ($7 \times 7$) to prevent sharp border steps, yielding feather mask $\mathbf{M}_{\text{feather}} \in [0.0, 1.0]$. The segmented radiograph is obtained via element-wise multiplication:
   $$\mathbf{I}_{\text{segmented}} = \mathbf{I}_{\text{CLAHE}} \odot \mathbf{M}_{\text{feather}}$$

### C. Resolution Schemes & Tensor Formatting
- **Pipeline A (Production High-Resolution)**: Processed at $512 \times 512 \times 3$, preserving fine reticular lines, micronodules, and subtle apical parenchymal details that are lost in standard $224 \times 224$ pipelines.
- **Pipeline B (Multi-Stage Latent Cross-Attention)**: Processed at $224 \times 224 \times 3$.
- **Tensor Normalization**: Normalized with standard ImageNet statistics: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

---

## 3. DUAL-BACKBONE ARCHITECTURES & INDUCTIVE BIAS COMPLEMENTARITY
RESPIRA bridges the fundamental trade-off between convolutional local inductive bias and transformer global contextual modeling by utilizing two heterogeneous, pre-trained neural vision backbones:

### A. Branch 1: EfficientNet-B0 (Convolutional Local Feature Extractor)
- **Role**: Extracts high-frequency spatial gradients, localized parenchymal textures, focal air bronchograms, and boundary opacities.
- **Core Mechanism**: MBConv blocks (Mobile Inverted Bottleneck Convolutions) utilizing depthwise separable convolutions, squeeze-and-excitation (SE) channel re-weighting, and swish activations.
- **Feature Representations**:
  - High-resolution spatial feature map: $\mathbf{F}_{\text{cnn}} \in \mathbb{R}^{B \times 1280 \times 16 \times 16}$ (for $512 \times 512$ inputs).
  - Global average pooled representation: $\mathbf{v}_{\text{cnn}} = \text{GAP}(\mathbf{F}_{\text{cnn}}) \in \mathbb{R}^{B \times 1280}$.
  - Classification head: Linear projection $\mathbf{W}_{\text{eff}} \in \mathbb{R}^{1280 \times 6} \implies$ Logits $\mathbf{z}_{\text{eff}} \in \mathbb{R}^6 \implies$ Probabilities $\mathbf{p}_{\text{eff}} = \text{Softmax}(\mathbf{z}_{\text{eff}})$.

### B. Branch 2: Vision Transformer ViT-B/16 (Global Thoracic Context Extractor)
- **Role**: Captures long-range spatial correlations, bilateral thoracic symmetry, cardiothoracic ratio, and diffuse bilateral pathologies (e.g., cardiomegaly, pulmonary edema).
- **Core Mechanism**:
  - Patch Partitioning: Non-overlapping patches of size $P \times P = 16 \times 16$. For $S=512$, sequence length $N_p = (512/16)^2 = 1024$ visual tokens (or $N_p = 196$ for $S=224$).
  - Linear Patch Embedding: Project each patch $\mathbf{x}_p \in \mathbb{R}^{16 \times 16 \times 3} \to \mathbb{R}^{D}$ where $D=768$.
  - Position Embedding: Prepend learnable `[CLS]` token $\mathbf{x}_{\text{cls}} \in \mathbb{R}^{768}$. Add 1D learnable position embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N_p + 1) \times 768}$ with bicubic interpolation for resolution scaling.
  - Transformer Layers: 12 Transformer encoder blocks, each with Multi-Head Self-Attention (12 heads, head dimension $d_h = 64$) and MLP blocks (expansion 3072) with GELU activation and LayerNorm.
  - Representation: Final `[CLS]` token $\mathbf{v}_{\text{vit}} \in \mathbb{R}^{B \times 768}$.
  - Classification head: Linear projection $\mathbf{W}_{\text{vit}} \in \mathbb{R}^{768 \times 6} \implies$ Logits $\mathbf{z}_{\text{vit}} \in \mathbb{R}^6 \implies$ Probabilities $\mathbf{p}_{\text{vit}} = \text{Softmax}(\mathbf{z}_{\text{vit}})$.

---

## 4. MULTI-MODAL FUSION ENGINES

RESPIRA investigates and deploys two multi-modal fusion mechanisms:

### Engine 1: High-Resolution Weighted Probability Fusion (Deployed Production Model)
- Directly combines softmax class probabilities from high-resolution ($512 \times 512$) fine-tuned backbones:
  $$\mathbf{p}_{\text{fused}} = \alpha^* \cdot \mathbf{p}_{\text{eff}} + (1 - \alpha^*) \cdot \mathbf{p}_{\text{vit}}$$
- **Optimal Grid-Search Weight**: $\alpha^* = 0.35$ (EfficientNet weight = 0.35, ViT-B/16 weight = 0.65).
- Giving $65\%$ weight to the Vision Transformer leverages the superior long-range global context of ViT-B/16 while allowing EfficientNet-B0 ($35\%$) to anchor localized textural opacities.
- **Empirical Test Performance (1,947 Held-Out CXRs)**:
  - Accuracy: **92.45%**
  - Macro ROC-AUC: **0.9910**
  - Macro Precision: **93.86%**, Macro Recall: **93.85%**, Macro F1: **93.85%**
  - Per-class ROC-AUC: Atelectasis (0.9999), Bacterial Pneumonia (0.9782), Normal (0.9991), Pulmonary Edema (1.0000), Tuberculosis (1.0000), Viral Pneumonia (0.9714).

### Engine 2: Advanced 5-Stage Latent Feature Cross-Attention Network (Pipeline B)
For dense multi-label representations, RESPIRA implements a deep cross-attention latent network:
1. **Stage 1 (Feature Spatial Alignment)**: CNN feature map $\mathbf{F}_{\text{cnn}} \in \mathbb{R}^{B \times 1280 \times 7 \times 7}$ is projected via $1\times 1$ convolution to dimension $d_k=512$, unflattened into 49 spatial tokens $\mathbf{T}_{\text{cnn}} \in \mathbb{R}^{B \times 49 \times 512}$. ViT tokens $\mathbf{T}_{\text{vit}} \in \mathbb{R}^{B \times 197 \times 768}$ are projected to $\mathbb{R}^{B \times 197 \times 512}$.
2. **Stage 2 (Bidirectional Cross-Attention)**:
   - CNN queries ViT: $\mathbf{A}_{\text{cnn}\leftarrow\text{vit}} = \text{Softmax}\left(\frac{\mathbf{Q}_{\text{cnn}}\mathbf{K}_{\text{vit}}^T}{\sqrt{d_k}}\right)\mathbf{V}_{\text{vit}} \in \mathbb{R}^{B \times 49 \times 512}$.
   - ViT queries CNN: $\mathbf{A}_{\text{vit}\leftarrow\text{cnn}} = \text{Softmax}\left(\frac{\mathbf{Q}_{\text{vit}}\mathbf{K}_{\text{cnn}}^T}{\sqrt{d_k}}\right)\mathbf{V}_{\text{cnn}} \in \mathbb{R}^{B \times 197 \times 512}$.
3. **Stage 3 (Gated Multi-Modal Fusion)**: Computes dynamic gating coefficients $\mathbf{g}_{\text{cnn}} = \sigma(\mathbf{W}_{g,c}[\mathbf{T}_{\text{cnn}} \,\|\, \mathbf{A}_{\text{cnn}\leftarrow\text{vit}}])$, producing fused visual tokens $\mathbf{F}_{\text{fused}} \in \mathbb{R}^{B \times 49 \times 512}$.
4. **Stage 4 (Disease-Conditioned Query Attention & Relational Modeling)**:
   - Six learnable disease queries $\mathbf{Q}_{\text{dis}} \in \mathbb{R}^{6 \times 512}$ probe $\mathbf{F}_{\text{fused}}$ to extract pathology-specific feature vectors $\mathbf{D}_{\text{spec}} \in \mathbb{R}^{B \times 6 \times 512}$.
   - Relational Self-Attention: A multi-head self-attention layer models pairwise clinical disease dependencies $\mathbf{A}_{\text{rel}} \in \mathbb{R}^{6 \times 6}$ (e.g., co-occurrence of effusion secondary to pneumonia).
5. **Stage 5 (Independent Multi-Label Heads)**: Projected through 6 independent classification heads with GELU and Dropout (0.3) yielding independent disease logits.

---

## 5. CALIBRATED UNCERTAINTY QUANTIFICATION & 3-TIER CLINICAL TRIAGE
Standard neural networks output uncalibrated, overconfident predictions on corrupted or out-of-distribution radiographs. RESPIRA calculates dual uncertainty metrics:

### Mathematical Formulation
1. **Normalized Binary Entropy**: Quantifies predictive dispersion across each disease head $c \in \{1, \dots, 6\}$:
   $$H_n(p_c) = -\frac{1}{\ln 2} \left[ p_c \ln p_c + (1 - p_c) \ln (1 - p_c) \right] \in [0.0, 1.0]$$
   Composite Entropy Score: $\bar{H} = \frac{1}{6} \sum_{c=1}^6 H_n(p_c)$.
2. **Decision Margin Uncertainty**: Evaluates ambiguity between the top two competing diagnoses:
   $$U_{\text{margin}} = 1 - (p_{(1)} - p_{(2)})$$
   where $p_{(1)}$ and $p_{(2)}$ denote the highest and second-highest sorted probabilities. If $p_{(1)} \approx p_{(2)}$, $U_{\text{margin}} \to 1.0$.

### 3-Tier Clinical Triage Protocol
Cases are automatically stratified into operational clinical tiers based on $\bar{H}$:
- **Low Risk ($\bar{H} < 0.20$)** $\implies$ **High Confidence**: Sharp decision boundaries. Eligible for automated preliminary reporting and fast-track routine queue.
- **Moderate Risk ($0.20 \le \bar{H} < 0.40$)** $\implies$ **Secondary Review**: Borderline confidence. Routed for expedited secondary radiologist verification with highlighted competing differential diagnoses.
- **High Risk ($\bar{H} \ge 0.40$)** $\implies$ **Mandatory Senior Audit**: Significant diagnostic ambiguity or out-of-distribution radiograph. Automated report is withheld; case is escalated for mandatory senior radiologist audit.

---

## 6. EXPLAINABLE AI (XAI) & DUAL-BRANCH QUANTITATIVE AUDITING
To eliminate black-box opacity, RESPIRA computes visual explanations from both architectures:

1. **EfficientNet Grad-CAM (Gradient-Weighted Class Activation Mapping)**:
   - Target layer: `features.7` (last convolutional stage, 1280 channels).
   - Importance weights: $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$.
   - Heatmap: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$, bilinearly upsampled to $512 \times 512$ and overlaid with Jet colormap.
2. **ViT Cross-Attention Mapping (Layer 11 Attention Rollout)**:
   - Target layer: Block 11 multi-head self-attention.
   - Extracts attention weights $\mathbf{A}_{\text{vit}} \in \mathbb{R}^{H \times (N+1) \times (N+1)}$.
   - Average across heads, extract row corresponding to `[CLS]` token attending to the $1024$ spatial patches $\mathbf{a}_{\text{cls}} \in \mathbb{R}^{1024}$.
   - Reshape into $32 \times 32$ spatial grid and bilinearly upsample to $512 \times 512$.
3. **Quantitative Cross-Modality Explainability Audit (Evaluated on 1,947 Test CXRs)**:
   - Pearson Correlation: $r = 0.684$
   - Cosine Similarity: $0.742$
   - Saliency IoU at top 20% attention energy ($\text{IoU}_{0.20}$): $0.518$
   - Cross-Modality Prediction Agreement Rate: **93.73%** (1,729 cases where both backbones agreed and were correct; 70 cases where EfficientNet was correct while ViT erred; 49 cases where ViT was correct while EfficientNet erred).
   - This proves the ensemble captures complementary localized vs. diffuse clinical signals.

---

## 7. TRAINING METHODOLOGY & REPRODUCIBILITY HYPERPARAMETERS
- **Optimization Strategy**: Two-stage progressive fine-tuning (backbones initialized with ImageNet pre-trained weights; classification heads trained first, followed by end-to-end unfreezing with differential learning rates).
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $\lambda = 1 \times 10^{-2}$).
- **Base Learning Rates**:
  - EfficientNet-B0: $1 \times 10^{-4}$
  - ViT-B/16: $5 \times 10^{-5}$
  - Multi-Stage Feature Fusion Heads: $3 \times 10^{-4}$
- **Batch Size**: 16 (high-resolution $512 \times 512$), 32 (feature fusion $224 \times 224$).
- **Learning Rate Schedule**: ReduceLROnPlateau (monitor validation loss, patience 3 epochs, decay factor $\gamma = 0.5$, minimum LR $1 \times 10^{-6}$).
- **Regularization**: Label Smoothing ($\epsilon = 0.05$), Dropout (0.2 in backbones, 0.3 in heads), Early Stopping (patience 6 epochs).
- **Loss Functions**:
  - Multi-Class Cross-Entropy with Label Smoothing:
    $$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^C y_{\text{smooth}, c} \log p_c$$
  - Multi-Label Binary Cross-Entropy with Asymmetric Focal Loss for tail classes:
    $$\mathcal{L}_{\text{Focal}} = -\sum_{c=1}^6 \left[ \alpha_c (1 - p_c)^\gamma y_c \log p_c + (1 - \alpha_c) p_c^\gamma (1 - y_c) \log (1 - p_c) \right]$$

---

## 8. CLINICAL DEPLOYMENT & SOFTWARE IMPLEMENTATION
The machine learning pipeline is fully integrated into a deployable clinical application:
1. **Asynchronous Backend Engine**: FastAPI (Python 3.13) under Uvicorn ASGI. Model checkpoints are instantiated during startup via a singleton `ModelManager`, eliminating per-request cold-start latency. Inferences execute inside `torch.inference_mode()`.
2. **Relational Database & Data Governance**: SQLAlchemy 2.0 ORM manages SQLite/PostgreSQL schemas with strict foreign-key relations tracking `Doctor`, `Patient`, `XRayStudy`, `Analysis`, and `Report` entities. Row-level physician ownership validation protects medical record confidentiality.
3. **Security Screening**: HS256-signed JWT tokens (8-hour expiry), bcrypt salted passwords. Uploaded CXR images undergo a 3-tier security gate: file extension verification, MIME magic-byte header validation, and PIL decompression checking capped at 15 MB.
4. **Automated Clinical PDF Reporting**: Built using ReportLab, compiling patient metadata, disease probability distribution charts, quantified uncertainty badges, side-by-side Grad-CAM and ViT attention overlays, and mandatory clinical disclaimers.
5. **Front-End & Desktop Client**: Built with React 19, TypeScript 5, Vite, and Tailwind CSS; packaged for offline hospital workstation deployment via an Electron desktop container with context isolation.

---

## 9. CODEBASE FILE & COMPONENT MAP
- `app/backend/preprocessing.py`: CLAHE, Otsu thresholding, morphological opening/closing, CCA filtering, anatomical ellipse fallback, feathering.
- `app/backend/networks.py` & `src/models/`: EfficientNet-B0 and ViT-B/16 network definitions, classification heads.
- `src/fusion/`: Weighted probability fusion engine and 5-stage bidirectional cross-attention latent network (`HierarchicalCXRFeatureFusion`).
- `app/backend/gradcam.py`: Layer-wise Grad-CAM gradient hooks for EfficientNet convolutional layers.
- `app/backend/vit_attention.py`: ViT Layer 11 attention rollout and spatial patch unflattening.
- `src/explainability/xai_comparison.py`: Quantitative XAI metrics (Pearson $r$, Cosine similarity, Saliency IoU).
- `backend/app/main.py`: FastAPI routes, auth endpoints, study uploads, asynchronous inference dispatch.
- `backend/app/services/pdf_report.py`: ReportLab automated clinical PDF generator.
- `docs/COMPLETE_PAPER_IEEE.tex` & `docs/METHODOLOGY_IEEE.tex`: Full IEEEtran LaTeX publications.
```
