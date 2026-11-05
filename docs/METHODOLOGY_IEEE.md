# Section III: METHODOLOGY

The proposed **RESPIRA** diagnostic framework addresses the challenges of multi-class thoracic pathology classification by integrating high-resolution representation learning, a hybrid convolutional-transformer backbone, multi-modal probability and feature fusion, calibrated uncertainty estimation, and dual-modality explainability. The classification space comprises six diagnostic categories: **Atelectasis**, **Bacterial Pneumonia**, **Normal (Healthy)**, **Pulmonary Edema**, **Tuberculosis**, and **Viral Pneumonia**.

---

## A. Multi-Stage Radiograph Preprocessing and Normalization

Chest radiographs (CXRs) exhibit high inter-scanner variance, low local soft-tissue contrast, and non-pulmonary anatomical occlusions. To standardize inputs while eliminating non-thoracic artifacts, an automated, deterministic preprocessing pipeline is applied prior to feature extraction.

### 1) Local Contrast Enhancement via CLAHE
Standard global histogram equalization over-amplifies background high-frequency noise in uniform air regions. Contrast Limited Adaptive Histogram Equalization (CLAHE) partitions the input grayscale radiograph $\mathbf{I} \in \mathbb{R}^{H \times W}$ into non-overlapping contextual grids of dimension $M \times N = 8 \times 8$. Within each tile, the local grayscale histogram is computed and clipped at threshold $\beta$:

$$\beta = \frac{M \cdot N}{L} \left(1 + \frac{\alpha_{\text{clip}}}{100}(s_{\max} - 1)\right) \tag{1}$$

where $L=256$ is the dynamic range, $\alpha_{\text{clip}} = 2.0$ denotes the clipping factor, and $s_{\max}$ is the maximum slope. Excess pixels exceeding $\beta$ are redistributed uniformly across all gray-level bins before computing the cumulative distribution function (CDF). Bilinear interpolation across tile boundaries eliminates artificial border discontinuities.

### 2) Heuristic Morphological Lung Field Segmentation
To force representations to focus on parenchymal abnormalities rather than diaphragmatic, humeral, or external radiopaque markers, a morphological segmentation cascade is executed:

1. **Dynamic Normalization and Gaussian Denoising**: The enhanced radiograph is linearly scaled to $\mathbf{I}_{\text{norm}} \in [0, 255]$ and convolved with an isotropic Gaussian kernel $\mathbf{G}_\sigma$ ($\text{kernel size} = 5 \times 5, \sigma = 0$):
   $$\mathbf{I}_{\text{blur}}(x, y) = (\mathbf{I}_{\text{norm}} * \mathbf{G}_\sigma)(x, y) \tag{2}$$

2. **Otsu Binarization**: An inverted global threshold $T^*$ maximizes the inter-class variance $\sigma_B^2(T)$ between radiolucent air fields and dense bone/mediastinum:
   $$T^* = \arg\max_{T} \left\{ \omega_0(T)\omega_1(T) \left[\mu_0(T) - \mu_1(T)\right]^2 \right\} \tag{3}$$
   yielding binary mask $\mathbf{M}_{\text{otsu}}(x, y) = \mathbb{I}(\mathbf{I}_{\text{blur}}(x, y) \le T^*)$.

3. **Morphological Filtering**: Using an elliptical structuring element $\mathbf{K}_{\text{ell}}$ of dimension $9 \times 9$, consecutive morphological opening ($\circ$) and closing ($\bullet$) operations eliminate sub-pleural noise and bridge internal pulmonary fissures:
   $$\mathbf{M}_{\text{morph}} = (\mathbf{M}_{\text{otsu}} \circ \mathbf{K}_{\text{ell}}) \bullet \mathbf{K}_{\text{ell}} \tag{4}$$

4. **Connected Component Analysis and Anatomic Filtering**: Connected components $\{C_k\}_{k=1}^K$ are extracted via 8-connectivity. Components satisfying area criterion $\text{Area}(C_k) \ge 0.03 \cdot H \cdot W$ and spatial centroid constraints:
   $$c_x(C_k) \in [0.10W, 0.90W], \quad c_y(C_k) \in [0.10H, 0.92H] \tag{5}$$
   are retained. The two largest components represent the left and right lung lobes.

5. **Anatomical Ellipse Fallback**: If severe consolidation or pleural effusion suppresses total detected lung area below $0.05 \cdot H \cdot W$, a geometric fallback mask $\mathbf{M}_{\text{fallback}}$ is computed:
   $$\frac{(x - 0.50W)^2}{(0.38W)^2} + \frac{(y - 0.52H)^2}{(0.40H)^2} \le 1 \tag{6}$$

6. **Feathering and Radiograph Masking**: The segmented binary mask is smoothed using a Gaussian filter ($7 \times 7$) to prevent edge step artifacts, normalized to $\mathbf{M}_{\text{feather}} \in [0.0, 1.0]$, and multiplied element-wise:
   $$\mathbf{I}_{\text{segmented}} = \mathbf{I}_{\text{CLAHE}} \odot \mathbf{M}_{\text{feather}} \tag{7}$$

### 3) Rescaling and Standardization
The segmented radiograph is interpolated using pixel-area resampling (`cv2.INTER_AREA`) to target dimensions $S \times S$ (where $S \in \{224, 512\}$), replicated into three identical channels to align with pretrained RGB weight tensors, and normalized:

$$\mathbf{X}_c = \frac{\frac{\mathbf{I}_c}{255} - \mu_c}{\sigma_c}, \quad c \in \{R, G, B\} \tag{8}$$

with ImageNet channel means $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and standard deviations $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.

---

## B. Deep Feature Extraction Backbones

Medical diagnosis requires simultaneously detecting local micro-textural abnormalities (e.g., fine reticular markings, localized tuberculous cavities) and broad contextual patterns (e.g., cardiomegaly, bilateral fluid redistribution). We deploy two complementary architectures:

### 1) Convolutional Inductive Bias: EfficientNet-B0
EfficientNet-B0 employs mobile inverted bottleneck convolutions (MBConv) reinforced with squeeze-and-excitation (SE) residual optimization. Depth ($d$), width ($w$), and input resolution ($r$) are scaled proportionally via a compound coefficient $\phi$:

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi \tag{9}$$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ with $\alpha \ge 1, \beta \ge 1, \gamma \ge 1$.

The final convolutional stage outputs activation tensor $\mathbf{A} \in \mathbb{R}^{B \times 1280 \times H' \times W'}$. For input resolution $224 \times 224$, $H'=W'=7$, producing 49 spatial feature tokens:

$$\mathbf{T}_{\text{eff}} \in \mathbb{R}^{B \times 49 \times 1280} \tag{10}$$

### 2) Global Self-Attention Modeling: Vision Transformer (ViT-B/16)
ViT-B/16 treats the radiograph as a sequence of discrete visual tokens. An input image $\mathbf{X} \in \mathbb{R}^{S \times S \times C}$ is partitioned into non-overlapping patches $\mathbf{x}_p \in \mathbb{R}^{P \times P \times C}$ with patch size $P=16$. The sequence length is:

$$N_p = \frac{S^2}{P^2} = \begin{cases} 196, & S = 224 \\ 1024, & S = 512 \end{cases} \tag{11}$$

Patches are flattened and projected into a latent embedding space of dimension $D = 768$ via learnable projection matrix $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$. A learnable classification token $\mathbf{x}_{\text{class}} \in \mathbb{R}^{1 \times D}$ and 1D position embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N_p + 1) \times D}$ are incorporated:

$$\mathbf{Z}_0 = [\mathbf{x}_{\text{class}} \,\|\, \mathbf{x}_p^1\mathbf{E} \,\|\, \dots \,\|\, \mathbf{x}_p^{N_p}\mathbf{E}] + \mathbf{E}_{\text{pos}} \tag{12}$$

When evaluating at $S=512$, the position embeddings are spatially interpolated using bicubic spline interpolation from grid $14 \times 14$ to $32 \times 32$. The token sequence is processed through 12 stacked transformer encoder layers, each computing multi-head self-attention (MHSA) across $h=12$ heads:

$$\text{MHSA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = [\text{head}_1 \,\|\, \dots \,\|\, \text{head}_h]\mathbf{W}^O \tag{13}$$

$$\text{head}_i = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{W}_i^Q (\mathbf{K}\mathbf{W}_i^K)^T}{\sqrt{d_k}}\right) \mathbf{V}\mathbf{W}_i^V \tag{14}$$

where $d_k = D / h = 64$. The resulting patch tokens represent sequence $\mathbf{T}_{\text{vit}} \in \mathbb{R}^{B \times N_p \times 768}$.

---

## C. Multi-Modal Model Fusion Architectures

RESPIRA implements two distinct fusion paradigms:

### 1) Pipeline A: High-Resolution (512×512) Probability Fusion
At $512 \times 512$, cross-attention over 1,024 patches imposes high quadratic compute and memory costs. Thus, Pipeline A performs probability-level ensembling of independent $512 \times 512$ backbones. Let $\mathbf{z}_{\text{eff}}, \mathbf{z}_{\text{vit}} \in \mathbb{R}^{C}$ represent the output logits of the respective networks. Normalized class probability vectors are derived via Softmax:

$$\mathbf{P}_{\text{eff}} = \text{softmax}(\mathbf{z}_{\text{eff}}), \quad \mathbf{P}_{\text{vit}} = \text{softmax}(\mathbf{z}_{\text{vit}}) \tag{15}$$

The ensemble prediction is defined by parameterized convex combination:

$$\mathbf{P}_{\text{fusion}} = \alpha \cdot \mathbf{P}_{\text{eff}} + (1 - \alpha) \cdot \mathbf{P}_{\text{vit}}, \quad \alpha \in [0.0, 1.0] \tag{16}$$

Weight parameter $\alpha$ was optimized through a systematic grid search on the held-out validation set ($N_{\text{val}} = 1003$):

$$\alpha^* = \arg\max_{\alpha \in [0, 1]} \text{Acc}_{\text{val}}(\mathbf{P}_{\text{fusion}}(\alpha)) \tag{17}$$

Empirical search yields $\alpha^* = 0.35$ ($\text{Weight}_{\text{eff}} = 0.35$, $\text{Weight}_{\text{vit}} = 0.65$), producing a validation accuracy of **92.97%** and a validation Macro-F1 of **93.93%**. Evaluated on 1,947 unseen test samples, this achieved **92.45%** accuracy and a Macro ROC-AUC of **0.9910**.

### 2) Pipeline B: Advanced Multi-Stage Latent Feature Fusion (224×224)
Pipeline B implements an intermediate feature fusion mechanism structured across five stages:

#### a) Latent Feature Projection
EfficientNet spatial tokens ($\mathbf{T}_{\text{eff}} \in \mathbb{R}^{B \times 49 \times 1280}$) and ViT patch tokens ($\mathbf{T}_{\text{vit}} \in \mathbb{R}^{B \times 196 \times 768}$) are projected into a shared latent metric space of dimension $d = 512$:

$$\mathbf{H}_{\text{eff}} = \text{GELU}(\text{LayerNorm}(\mathbf{T}_{\text{eff}}\mathbf{W}_{\text{proj\_eff}} + \mathbf{b}_{\text{proj\_eff}})) \tag{18}$$

$$\mathbf{H}_{\text{vit}} = \text{GELU}(\text{LayerNorm}(\mathbf{T}_{\text{vit}}\mathbf{W}_{\text{proj\_vit}} + \mathbf{b}_{\text{proj\_vit}})) \tag{19}$$

with $\mathbf{H}_{\text{eff}} \in \mathbb{R}^{B \times 49 \times 512}$ and $\mathbf{H}_{\text{vit}} \in \mathbb{R}^{B \times 196 \times 512}$.

#### b) Bidirectional Cross-Attention
Two asymmetric cross-attention modules with $h=8$ heads ($d_k = 64$) and dropout $p=0.1$ permit bidirectional information exchange:

$$\mathbf{H}'_{\text{eff}} = \text{LayerNorm}\left(\mathbf{H}_{\text{eff}} + \text{MHA}(\mathbf{Q}=\mathbf{H}_{\text{eff}}, \mathbf{K}=\mathbf{H}_{\text{vit}}, \mathbf{V}=\mathbf{H}_{\text{vit}})\right) \tag{20}$$

$$\mathbf{H}'_{\text{vit}} = \text{LayerNorm}\left(\mathbf{H}_{\text{vit}} + \text{MHA}(\mathbf{Q}=\mathbf{H}_{\text{vit}}, \mathbf{K}=\mathbf{H}_{\text{eff}}, \mathbf{V}=\mathbf{H}_{\text{eff}})\right) \tag{21}$$

Each sub-layer is followed by a feed-forward network ($\text{FFN}: 512 \rightarrow 2048 \rightarrow 512$) with GELU activations. The cross-attended representations are concatenated along the token sequence dimension:

$$\mathbf{H}_{\text{fused}} = [\mathbf{H}''_{\text{eff}} \,\|\, \mathbf{H}''_{\text{vit}}] \in \mathbb{R}^{B \times 245 \times 512} \tag{22}$$

#### c) Disease-Conditioned Attention
Six learnable disease query embeddings $\mathbf{Q}_d \in \mathbb{R}^{6 \times 512}$ are instantiated ($\mathbf{Q}_d \sim \mathcal{N}(0, 0.02^2)$) to extract disease-specific patterns from the multimodal tokens:

$$\mathbf{D} = \text{softmax}\left(\frac{\mathbf{Q}_d \mathbf{W}_Q (\mathbf{H}_{\text{fused}}\mathbf{W}_K)^T}{\sqrt{d_k}}\right) (\mathbf{H}_{\text{fused}}\mathbf{W}_V) \tag{23}$$

After residual addition, LayerNorm, and FFN refinement ($512 \rightarrow 1024 \rightarrow 512$), the disease-specific feature matrix is:

$$\mathbf{D}_{\text{refined}} \in \mathbb{R}^{B \times 6 \times 512} \tag{24}$$

#### d) Disease Relationship Modeling
Inter-pathology co-occurrences are explicitly modeled via an 8-head self-attention module operating across the 6 disease embeddings:

$$\mathbf{A}_{\text{rel}} = \frac{1}{h}\sum_{i=1}^h \text{softmax}\left(\frac{\mathbf{D}_{\text{refined}}\mathbf{W}_i^Q (\mathbf{D}_{\text{refined}}\mathbf{W}_i^K)^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times 6 \times 6} \tag{25}$$

Matrix $\mathbf{A}_{\text{rel}}[i, j]$ denotes the learned relational dependence between disease class $i$ and class $j$.

#### e) Relational Propagation and Multi-Label Decision Heads
1. Disease representations are projected: $\mathbf{F}_{\text{proj}} = \text{GELU}(\text{LayerNorm}(\mathbf{D}_{\text{refined}}\mathbf{W}_p)) \in \mathbb{R}^{B \times 6 \times 256}$.
2. Relational feature propagation is performed via batch matrix multiplication:
   $$\mathbf{F}_{\text{rel}} = \mathbf{A}_{\text{rel}} \mathbf{F}_{\text{proj}}, \quad \mathbf{F}_{\text{enh}} = \mathbf{F}_{\text{proj}} + \mathbf{F}_{\text{rel}} \tag{26}$$
3. Six independent sub-networks compute the output logit $z_c$ for each disease condition $c \in \{1, \dots, 6\}$:
   $$z_c = \mathbf{W}_{c, 2} \left( \text{Dropout}_{0.3}(\text{GELU}(\mathbf{W}_{c, 1} \mathbf{F}_{\text{enh}}[c] + \mathbf{b}_{c, 1})) \right) + b_{c, 2} \tag{27}$$
   where $\mathbf{W}_{c, 1} \in \mathbb{R}^{256 \times 128}$ and $\mathbf{W}_{c, 2} \in \mathbb{R}^{128 \times 1}$.
4. Final probabilities are generated via Sigmoid activation: $p_c = \sigma(z_c) = \frac{1}{1 + e^{-z_c}}$.

---

## D. Calibrated Uncertainty Quantification

To prevent silent diagnostic failures on atypical or ambiguous radiographs, RESPIRA computes dual uncertainty metrics:

### 1) Normalized Binary Entropy
For each disease class $c$, binary entropy measures predictive dispersion:

$$H(p_c) = -\frac{1}{\ln(2)} \left[ p_c \ln(p_c) + (1 - p_c)\ln(1 - p_c) \right] \tag{28}$$

where $p_c$ is clamped to $[\epsilon, 1 - \epsilon]$ ($\epsilon = 10^{-8}$) for numerical stability. The composite sample uncertainty is the arithmetic mean across all classes:

$$\bar{H} = \frac{1}{C}\sum_{c=1}^C H(p_c) \quad \in [0.0, 1.0] \tag{29}$$

### 2) Top-1 / Top-2 Margin Uncertainty
Margin uncertainty quantifies ambiguity at the decision boundary between the two most probable classes:

$$U_{\text{margin}} = 1.0 - \left( p_{(1)} - p_{(2)} \right) \tag{30}$$

where $p_{(1)}$ and $p_{(2)}$ represent the highest and second-highest sorted probabilities.

### 3) Clinical Risk Triage Stratification
Based on composite uncertainty $\bar{H}$, cases are automatically stratified into three operational clinical tiers:

$$\text{Tier} = \begin{cases}
\textbf{Low Risk} \, (\text{High Confidence}), & \bar{H} < 0.20 \\
\textbf{Moderate Risk} \, (\text{Secondary Review}), & 0.20 \le \bar{H} < 0.40 \\
\textbf{High Risk} \, (\text{Mandatory Senior Audit}), & \bar{H} \ge 0.40
\end{cases} \tag{31}$$

---

## E. Explainable AI (XAI) and Visual Interpretability

To provide clinicians with verifiable visual evidence, RESPIRA generates dual-branch saliency maps that capture both local activation patterns and global transformer token interactions.

### 1) Convolutional Branch: Grad-CAM
Grad-CAM is coupled to the final convolutional layer of EfficientNet-B0 (`backbone.features.8.2`). Gradients of the target class score $y^c$ with respect to spatial feature map $A^k \in \mathbb{R}^{H' \times W'}$ are computed via backpropagation:

$$\alpha_k^c = \frac{1}{H' \cdot W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A_{i, j}^k} \tag{32}$$

Feature activations are linearly combined and rectified:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k \right) \tag{33}$$

The map is bilinearly upsampled to the original input resolution, min-max normalized, and blended with the radiograph using a Jet colormap ($\alpha_{\text{blend}} = 0.5$).

### 2) Transformer Branch: Cross-Attention Patch Saliency
Transformer interpretability is obtained directly from the ViT-to-CNN cross-attention tensor $\mathbf{A}_{\text{vit}\to\text{cnn}} \in \mathbb{R}^{196 \times 49}$. The importance of ViT patch token $p$ is derived by marginalizing across the 49 spatial CNN tokens:

$$s_p = \frac{1}{49} \sum_{j=1}^{49} \mathbf{A}_{\text{vit}\to\text{cnn}}[p, j], \quad \forall p \in \{1, \dots, 196\} \tag{34}$$

Vector $\mathbf{s} \in \mathbb{R}^{196}$ is reshaped into a $14 \times 14$ spatial grid, normalized to $[0, 1]$, bilinearly interpolated to the radiograph's native resolution, and rendered as a heatmap overlay.

### 3) Quantitative XAI Evaluation Metrics
To measure concordance between CNN and Transformer visual explanations, three quantitative metrics are computed over all 1,947 test images:
1. **Pearson Correlation Coefficient ($r$)**:
   $$r = \frac{\sum_{i=1}^{HW} (M_{\text{cnn}, i} - \bar{M}_{\text{cnn}})(M_{\text{vit}, i} - \bar{M}_{\text{vit}})}{\sqrt{\sum_{i=1}^{HW} (M_{\text{cnn}, i} - \bar{M}_{\text{cnn}})^2 \sum_{i=1}^{HW} (M_{\text{vit}, i} - \bar{M}_{\text{vit}})^2}} \tag{35}$$
2. **Cosine Similarity**:
   $$\cos(\theta) = \frac{\mathbf{m}_{\text{cnn}} \cdot \mathbf{m}_{\text{vit}}}{\|\mathbf{m}_{\text{cnn}}\|_2 \|\mathbf{m}_{\text{vit}}\|_2} \tag{36}$$
3. **Attention Intersection over Union ($\text{IoU}_{20\%}$)**:
   Binary masks $B_{\text{cnn}}$ and $B_{\text{vit}}$ isolate the top 20% most salient pixels ($p \ge \text{Percentile}_{80}(M)$):
   $$\text{IoU}_{20\%} = \frac{|B_{\text{cnn}} \cap B_{\text{vit}}|}{|B_{\text{cnn}} \cup B_{\text{vit}}|} \tag{37}$$

---

## F. Training Protocol and Optimization Specifications

Models were implemented in PyTorch 2.10 and trained on an NVIDIA GeForce RTX 4060 GPU with CUDA 12.8 using Automatic Mixed Precision (AMP).

### 1) Two-Stage Transfer Learning Protocol
To stabilize optimization and prevent catastrophic disruption of pretrained ImageNet representations:
- **Stage 1: Classifier Warm-up ($E_1 = 3$ epochs)**: Feature extraction backbones are frozen (`requires_grad = False`; Batch Normalization maintained in evaluation mode). Only classification heads are updated using AdamW ($\eta_1 = 1.0 \times 10^{-3}$, weight decay $\lambda = 1.0 \times 10^{-4}$).
- **Stage 2: Full Network Fine-Tuning ($E_2 = 25 - 30$ epochs)**: All layers are unfrozen. AdamW learning rate is reduced to $\eta_2 = 1.0 \times 10^{-4}$ for EfficientNet-B0 and $\eta_2 = 1.0 \times 10^{-5}$ for ViT-B/16.

### 2) Loss Formulations and Regularization
Multi-class models are trained using Categorical Cross-Entropy:
$$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^C y_c \log \hat{y}_c \tag{38}$$

Multi-label relational classification heads are trained using Binary Cross-Entropy with Logits:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{C}\sum_{c=1}^C \left[ y_c \log \sigma(z_c) + (1 - y_c)\log(1 - \sigma(z_c)) \right] \tag{39}$$

Optimization incorporates a `ReduceLROnPlateau` scheduler ($\gamma = 0.5$, patience $= 2$ epochs) monitoring validation loss, with early stopping triggered after 6 consecutive non-improving epochs.

### Table I: Hyperparameter Specifications for Model Training and Fusion

| Configuration Parameter | EfficientNet-B0 | ViT-B/16 | Multi-Stage Fusion |
|---|---|---|---|
| Input Spatial Resolution | $512 \times 512$ | $512 \times 512$ | $224 \times 224$ |
| Input Patch / Token Count | 49 spatial tokens | 1024 patches | 245 multimodal |
| Latent Embedding Dimension | 1280 | 768 | 512 |
| Attention Heads | N/A | 12 heads | 8 heads |
| Warmup Epochs / LR ($\eta_1$) | 3 / $1 \times 10^{-3}$ | 3 / $1 \times 10^{-3}$ | N/A |
| Fine-Tuning LR ($\eta_2$) | $1 \times 10^{-4}$ | $1 \times 10^{-5}$ | $1 \times 10^{-4}$ |
| Optimizer | AdamW | AdamW | AdamW |
| Weight Decay ($\lambda$) | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| Batch Size | 16 | 8 | 256 (features) |
| Dropout Probability | 0.3 | 0.1 | 0.2 -- 0.3 |
| LR Scheduler Decay Factor ($\gamma$) | 0.5 | 0.5 | 0.5 |
| Early Stopping Patience | 6 epochs | 6 epochs | 7 epochs |
| Best Checkpoint Epoch | Epoch 20 | Epoch 15 | Epoch 18 |

---

## G. Clinical Deployment Architecture and Software Implementation

The validated deep learning framework is integrated into a multi-tier clinical software suite:

1. **Asynchronous Backend Engine**: Built on FastAPI (Python 3.13) under Uvicorn ASGI. Model checkpoints are instantiated during service startup via a singleton `ModelManager`, avoiding per-request initialization latency. Inferences execute inside `torch.inference_mode()`.
2. **Persistence and Data Governance**: SQLAlchemy 2.0 ORM manages relational data schemas across SQLite and PostgreSQL, with strict foreign key constraints tracking `Doctor`, `Patient`, `XRayStudy`, `Analysis`, and `Report` entities. Row-level physician ownership validation protects medical record confidentiality.
3. **Security Controls**: Transport authentication utilizes HS256-signed JSON Web Tokens (JWT, 8-hour expiry). Passwords are salted and hashed via bcrypt. Ingested images undergo three-tier security screening: file extension validation, MIME magic-byte verification, and PIL decompression checking capped at 15 MB.
4. **Automated Clinical PDF Reporting**: A ReportLab-based reporting engine compiles patient demographics, predicted class distributions, quantified uncertainty, dual-modality side-by-side XAI heatmaps, and a mandatory clinical disclaimer into an auditable document.
5. **Front-End Clinical User Interface**: Implemented using React 19, TypeScript 5, Vite, and Tailwind CSS, packaged for desktop workstations via an Electron container with strict context isolation.

---

### Table II: Empirical Test Set Performance Across Individual Backbones and Fusion Architectures

| Model Architecture | Resolution | Test Accuracy (%) | Macro Prec. (%) | Macro Rec. (%) | Macro F1 (%) | Macro ROC-AUC |
|---|---|---|---|---|---|---|
| EfficientNet-B0 (Single Branch) | $512 \times 512$ | 92.40 | 93.48 | 93.77 | 93.61 | 0.9904 |
| ViT-B/16 (Single Branch) | $512 \times 512$ | 91.32 | 92.93 | 92.94 | 92.93 | 0.9896 |
| **RESPIRA Weighted Fusion (0.35 / 0.65)** | **512 × 512** | **92.45** | **93.86** | **93.85** | **93.85** | **0.9910** |
| Multi-Stage Feature Fusion (Pipeline B) | $224 \times 224$ | 88.08 | 87.88 | 89.50 | 88.62 | 0.9794 |
