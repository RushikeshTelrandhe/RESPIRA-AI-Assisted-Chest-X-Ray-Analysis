# Section II: RELATED WORK

Deep learning has emerged as a cornerstone for computer-aided diagnosis in thoracic radiology, with convolutional neural networks (CNNs) and Vision Transformers (ViTs) being extensively explored for automated chest X-ray (CXR) interpretation. However, achieving clinically viable diagnostic systems requires addressing not only raw classification accuracy, but also rigorous dataset quality control, multi-scale feature representations, inter-disease relational dependencies, calibrated uncertainty estimation, and faithful visual interpretability.

---

### A. Convolutional Architectures and Image Quality Enhancement
Convolutional architectures have historically dominated medical image classification due to their inherent translation invariance and strong localized inductive bias. Pandit et al. [1] presented an explainable deep learning comparative study for multi-class pulmonary disease classification (evaluating normal, lung opacity, and viral pneumonia) using multiple pretrained CNN backbones, including NASNetMobile, Xception, InceptionV3, DenseNet-121, ResNet-50, and VGG-16. Their framework incorporated class balancing, standard data augmentation, and Grad-CAM visualizations, demonstrating that deep CNN backbones capture discriminative textural patterns. In a parallel investigation, Shukla et al. [7] emphasized that radiograph classification accuracy is fundamentally constrained by input image fidelity; they proposed a deep learning pipeline combining contrast enhancement techniques with CNN classifiers to mitigate artifactual noise and enhance subtle parenchymal abnormalities. 

To improve sensitivity toward fine-grained lesions without explosive parameter scaling, attention-augmented CNNs have gained substantial traction. Dey [8] proposed CBAM-DenseNet121, integrating Channel and Spatial Convolutional Block Attention Modules (CBAM) into DenseNet-121 for differentiating normal radiographs from bacterial and viral pneumonia. This demonstrated that spatial and channel re-weighting significantly boosts diagnostic sensitivity in low-contrast zones. Further extending dense connectivity, Mažonienė and Šešok [9] developed DualPool-DenseNet121 for multi-label chest disease detection, combining dual pooling mechanisms with Grad-CAM saliency mapping to preserve both peak activation signals and average context. Yaligar et al. [10] conducted a benchmark of state-of-the-art CNN strategies across multi-label thoracic disorders, highlighting that while deep residual and densely connected networks achieve high diagnostic AUCs, pure CNN architectures struggle to model bilateral thoracic symmetry and long-range spatial correlations across distant pulmonary lobes.

---

### B. Vision Transformers in Radiographic Image Analysis
To overcome the constrained receptive fields of convolutional filters, self-attention mechanisms and Vision Transformers have been actively adapted to radiographic imaging. Kansal, Chandra, and Singh [2] proposed CXR-ViT, a pure transformer framework demonstrating that patch-based self-attention effectively captures widespread radiographic patterns. Similarly, Ghosh et al. [3] established that Vision Transformers excel in multi-label chest radiograph analysis on large-scale benchmarks (such as NIH ChestX-ray14), showing that all-to-all patch attention layers can identify diffuse thoracic abnormalities (such as cardiomegaly and pulmonary edema) more robustly than localized convolutional filters. 

Addressing domain-specific adaptation, Fu, Tantithamthavorn, and Le [11] formulated DAViT, a domain-adapted vision transformer tailored for automated pneumonia detection and localized visual explanations, proving that pretraining and fine-tuning strategies tailored to medical radiographs yield superior token representations. More recently, Ba et al. [12] introduced a monotonic multi-expert Vision Transformer for clinical chest X-ray diagnosis, integrating multiple specialized attention experts to resolve localized pathological ambiguities. While these investigations substantiate the power of Vision Transformers for radiographic analysis, pure transformer models typically exhibit weak inductive bias, demand extensive training corpora, and occasionally neglect high-frequency textural boundaries essential for detecting early-stage infiltrates.

---

### C. Hybrid CNN-Transformer Architectures and Multi-Modal Feature Fusion
Recognizing the complementary strengths of CNNs (local feature extraction) and ViTs (global context aggregation), recent literature has shifted toward hybrid fusion frameworks. Rafi et al. [4] performed a comparative and integrative analysis of CNNs and Vision Transformers for pneumonia and tuberculosis detection, reporting accuracies of 89.61% and 90.12% respectively, while demonstrating that combining both structural paradigms yields superior sensitivity over single-model baselines. Altun and Candemir [13] proposed a differential evolution-based CNN-transformer fusion framework, demonstrating that metaheuristic-guided feature combinations between convolutional and self-attention representations achieve superior thoracic pathology discrimination compared to standalone architectures.

Concurrently, Naghash, Ling, and Li [14] developed HyCoViT, a hybrid convolution-vision transformer featuring dynamic dropout to stabilize the fusion of dense local representations with global contextual embeddings. Agarwal and Verma [15] explored cross-model attention mechanisms that align transformer-based descriptors with localized convolutional feature maps, enhancing feature saliency across complex chest radiographs. Bahram et al. [16] designed CheX-DFuseNet, combining EfficientNetV2 backbones with ViT modules to maximize feature richness in multi-label settings. Despite these architectural advances, existing fusion frameworks frequently operate at reduced spatial resolutions ($224 \times 224$) or employ unweighted feature concatenations that fail to explicitly account for pathological interdependencies.

---

### D. Label Dependencies, Co-Occurrence Modeling, and Cross-Attention
Chest X-ray diagnosis is inherently complex because thoracic pathologies frequently co-occur (e.g., pleural effusion secondary to pneumonia, or atelectasis accompanying severe consolidation). To address label correlations, Shi et al. [17] developed a visual-semantic framework employing graph-guided multiscale cross-attention, demonstrating that coupling heterogeneous CNN-ViT visual encoders with graph convolutional reasoning over label dependencies substantially improves classification performance across ambiguous thoracic lesions. 

Addressing long-tailed clinical distributions, Shao et al. [18] introduced TRCGL-Net, which combines generative data augmentation with explicit label co-occurrence modeling to prevent majority-class bias from suppressing rare pulmonary conditions. These studies highlight that multi-label chest radiograph diagnosis cannot treat diagnostic classes as mutually independent Bernoulli variables; instead, intermediate feature representations must be explicitly conditioned on disease-specific queries and inter-pathology relationships.

---

### E. Model Explainability and Visual Interpretability (XAI)
In medical artificial intelligence, clinical acceptance requires transparent, verifiable visual rationales for automated predictions. Kim, Choi, and Seo [5] proposed a hierarchical explainability framework using layer-wise Grad-CAM on U-Net feature representations, extracting multi-scale activation maps across 14 radiological observations on the MIMIC-CXR dataset to improve radiological trust. Barra et al. [6] presented a comparative evaluation of XAI techniques in medical imaging, analyzing Grad-CAM, LIME, and SHAP. Their findings revealed that while Grad-CAM provides computationally efficient and anatomically intuitive gradient heatmaps, perturbation methods like LIME and SHAP incur prohibitive computational overhead.

Pabba, Samudrala, and Chevuturi [19] demonstrated the clinical utility of saliency mappings in identifying focal pulmonary consolidations, while Qi, Chen, and Cai [20] provided an extensive synthesis of attention mechanisms across medical computer vision, arguing that visualizing self-attention weights provides an authentic window into transformer decision logic. However, existing clinical XAI implementations predominantly evaluate heatmaps qualitatively on select anecdotal samples, leaving a critical gap in systematic, quantitative auditing of explanation concordance across CNN and transformer modalities.

---

### F. Uncertainty Estimation, Model Calibration, and Clinical Triage
Standard deep neural networks tend to yield uncalibrated, overconfident predictions on out-of-distribution or noisy radiographs. Xu and Rajpurkar [21] analyzed calibration and uncertainty estimation in chest X-ray classification models, revealing that even high-performing self-supervised models exhibit severe miscalibration on external clinical datasets, with Expected Calibration Error (ECE) scores ranging from 0.12 to 0.41. Their findings underscore that unquantified predictive confidence poses severe risks in clinical radiology.

To prevent diagnostic hazards on ambiguous scans, Aperstein et al. [22] proposed selective multi-pathology classification via automated rejection mechanisms, showing that referring high-uncertainty radiographs to human radiologists markedly improves system-wide diagnostic reliability. Consequently, integrating information-theoretic uncertainty metrics directly into automated classification pipelines is essential for clinical risk triage.

---

### G. Identified Research Gaps and Contributions of the Present Work
A critical review of the recent literature ([1]–[22]) reveals several unresolved research gaps:
1. **Resolution and Inductive Bias Disconnect**: Most transformer-based frameworks downsample radiographs to $224 \times 224$ pixels, discarding fine textural markers (such as small apical nodules, subtle reticular patterns, or early tuberculous cavities). Conversely, high-resolution models rarely leverage balanced dual-stream hybrid fusion.
2. **Heuristic Fusion vs. Relational Conditioning**: Existing fusion approaches predominantly employ naive concatenation or late averaging, failing to model inter-disease clinical co-occurrences through learned disease-conditioned queries.
3. **Absence of Calibrated Uncertainty in Diagnostic Workflows**: High-accuracy models routinely omit predictive uncertainty bounds, treating confident and ambiguous predictions identically without clinical risk stratification.
4. **Lack of Dual-Modality Quantitative XAI Auditing**: Most XAI studies evaluate Grad-CAM on CNNs or attention rollout on ViTs independently, without conducting paired quantitative concordance audits (e.g., via Pearson correlation, Cosine similarity, and salient attention IoU) across identical test cohorts.
5. **Separation of Machine Learning Models from Deployable Software**: Research models are frequently evaluated in isolation as research scripts, lacking authenticated, end-to-end integration into asynchronous backends with automated clinical report generation.

**The Present Work (RESPIRA)** systematically resolves these limitations through a unified, end-to-end clinical framework. RESPIRA introduces:
- A rigorous, deterministic preprocessing and segmentation pipeline incorporating CLAHE, Otsu morphological lung field masking, and anatomical fallback heuristics.
- A dual-backbone architecture coupling **EfficientNet-B0** (local textural inductive bias) with **ViT-B/16** (global thoracic context).
- A high-resolution ($512 \times 512$) weighted probability fusion pipeline alongside an advanced 5-stage latent feature fusion pipeline featuring bidirectional cross-attention, 6 disease-conditioned query embeddings, and inter-disease relational modeling via self-attention.
- Calibrated uncertainty quantification combining normalized binary entropy and decision margin uncertainty into a 3-tier clinical triage stratification protocol.
- A paired, quantitative dual-modality explainability audit (Grad-CAM vs. ViT Cross-Attention Mapping) evaluated across 1,947 test images.
- A fully realized clinical platform featuring an asynchronous FastAPI backend, SQLAlchemy ORM with row-level physician ownership, automated ReportLab PDF diagnostic reporting, and a React 19 / Electron desktop suite.

---

## References

```bibtex
[1] S. K. Pandit, S. S. Arjon, and T. Yasmin, "Explainable deep learning for lung disease detection: A comparative study using Grad-CAM," in Proc. 28th Int. Conf. Comput. Inf. Technol. (ICCIT), Cox's Bazar, Bangladesh, Dec. 2025, pp. 1-6.

[2] K. Kansal, T. B. Chandra, and A. Singh, "CXR-ViT: A transformer-based framework for enhanced chest X-ray classification," in Proc. IEEE Conf. Comput. Vision Pattern Recognit. Workshops / IEEE Int. Conf., 2025, pp. 1-6.

[3] S. Ghosh, A. Bandyopadhyay, M. Bose, and K. C. Santosh, "Vision Transformers excel in chest X-ray analysis," in Proc. IEEE Conf. Artif. Intell. (CAI), Santa Clara, CA, USA, 2025, pp. 1-6.

[4] M. S. M. Rafi, M. S. Ali, M. M. Hossen, S. Z. R. Sneha, and M. S. Arefin, "Chest X-ray analysis for pneumonia and tuberculosis detection using CNNs and Vision Transformers with explainable AI," in Proc. IEEE 2nd Int. Conf. Comput., Appl. Syst. (COMPAS), Kushtia, Bangladesh, Oct. 2025, pp. 1-6.

[5] K. Kim, Y. Choi, and J. Seo, "Explainable AI for multi-label chest X-ray diagnosis: Layer-wise Grad-CAM with hierarchical feature extraction," IEEE Trans. Med. Imaging / IEEE Access, 2025.

[6] P. Barra, A. Della Greca, I. Amaro, and A. Tortora, "A comparative analysis of XAI techniques for medical imaging: Challenges and opportunities," in Proc. IEEE Int. Conf. Bioinf. Biomed. (BIBM), Lisbon, Portugal, Dec. 2024, pp. 6782-6787.

[7] P. Shukla, M. Raj, D. Joshi, and S. M. B. S, "Chest X-ray classification using deep learning framework with image enhancement," in Proc. IEEE Int. Conf., 2026, pp. 1-6.

[8] U. K. Dey, "CBAM-Enhanced DenseNet121 for multi-class chest X-ray classification with Grad-CAM explainability," IEEE Access, vol. 13, pp. 1-14, 2025.

[9] E. Mažonienė and D. Šešok, "Explainable multi-label chest X-ray disease classification using DualPool-DenseNet121 and Grad-CAM visualization," Symmetry, vol. 18, no. 7, Art. no. 1294, 2026.

[10] N. Yaligar, S. G, K. Patil, P. Kulkarni, and P. Bhajantri, "State-of-the-art deep learning strategies for multi-label classification of chest X-ray thoracic disorders," in Proc. IEEE Int. Conf., 2024, pp. 1-6.

[11] M. Fu, C. Tantithamthavorn, and T. Le, "DAViT: A domain-adapted vision transformer for automated pneumonia detection and explanation using chest X-ray images," IEEE Access, vol. 13, pp. 1-12, 2024/2025.

[12] T. Ba, C. Bi, J. Yu, M. Z. Jhandir, H. M. Raza ur Rehman, and G. S. Choi, "A monotonic multi-expert Vision Transformer for clinical chest X-ray diagnosis," Frontiers in Medicine, vol. 13, Art. no. 1847086, Aug. 2026.

[13] S. Altun and S. Candemir, "Differential evolution-based CNN-transformer fusion for thoracic disease classification from chest X-rays," IEEE Access, vol. 14, pp. 1-14, 2026.

[14] O. A. Naghash, N. Ling, and X. Li, "HyCoViT: Hybrid convolution vision transformer with dynamic dropout for enhanced medical chest X-ray classification," IEEE Access, vol. 13, pp. 1-13, Jun. 2025.

[15] L. Agarwal and B. Verma, "Advanced chest X-ray analysis via transformer-based image descriptors and cross-model attention mechanism," Journal of Imaging Systems and Technology / Wiley, 2025.

[16] A. M. Bahram, S. M. Omer, H. M. Mohammed, and S. A. Aula, "Enhanced chest disease classification using an improved CheXNet framework with EfficientNetV2-M and optimization-driven learning (CheX-DFuseNet)," J. Comput. Med. Imaging, 2025/2026.

[17] G. Shi, Z. Wang, Y. Shi, J. Pan, L. Sun, F. Fang, and L. Jin, "Graph guided multiscale cross attention for multilabel chest X ray classification," Scientific Reports, vol. 16, Art. no. 22761, 2026.

[18] T. Shao, H. Ling, L. Zhang, J. Wu, J. Wang, Y. Gao, and F. Wang, "TRCGL-Net: A long-tailed multi-label chest X-ray classification framework with generative data augmentation and label co-occurrence modeling," Comput. Biol. Med., 2025/2026.

[19] P. Pabba, U. Samudrala, V. S. S. Chevuturi, and M. P. R. S. K, "Explainable artificial intelligence for pulmonary disease detection from chest radiographs," in Proc. IEEE Int. Conf., 2025, pp. 1-6.

[20] Y. Qi, R. Chen, and J. Cai, "Attention mechanisms in medical image analysis: Advance, challenge, and future direction," in Proc. IEEE Int. Conf. Information Engineering, 2025, pp. 1-6.

[21] J. Xu and P. Rajpurkar, "Calibration and uncertainty estimation challenges in self-supervised chest X-ray pathology classification models," in Proc. Med. Imaging Deep Learn. (MIDL), Paris, France, Jul. 2024, pp. 1-6.

[22] Y. Aperstein, A. Tzahar, A. Gottlib, T. Verber, R. S. Damti, and A. Apartsin, "Selective multipathology chest X-ray classification via rejection mechanisms," Scientific Reports, 2026, doi: 10.1038/s41598-026-66294-7.
```
