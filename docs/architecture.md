# Respira — Architecture

## Research pipeline (existing, untouched)

```
Image (224×224 RGB, CLAHE + lung mask, ImageNet norm)
 ├─ EfficientNet-B0 → spatial features (B,1280,7,7) → 49 tokens
 └─ ViT-B/16       → patch tokens (B,196,768)
      → FeatureProjection → (B,49,512) + (B,196,512)
      → BidirectionalCrossAttention → fused (B,245,512)
      → DiseaseConditionedAttention (6 learned queries) → (B,6,512)
      → DiseaseRelationshipModel (8-head self-attn) → (B,6,6)
      → MultiLabelClassificationHeads → logits (B,6) → sigmoid
      → binary-entropy uncertainty + Grad-CAM (EffNet branch) + ViT cross-attention maps
```

Code: `src/` (training/analysis), `app/backend/` (live pipeline:
`config.py`, `preprocessing.py`, `networks.py`, `pipeline.py`,
`gradcam.py`, `vit_attention.py`, `main.py` — the original research API).

Checkpoints (canonical): `outputs/efficientnet_b0/checkpoints/best_model.pth`,
`outputs/vit/checkpoints/best_model.pth` (~1 GB),
`outputs/fusion/adaptive_fusion/checkpoints/best_model.pth`,
`outputs/fusion/classification/checkpoints/best_model.pth` (final head).
Class order (everywhere): Atelectasis, Bacterial Pneumonia, Normal,
Pulmonary Edema, Tuberculosis, Viral Pneumonia.
Verified test metrics: accuracy 88.08%, macro F1 88.62%, ROC-AUC 97.94%
(`outputs/final_prediction/reports/final_metrics.json`).

## Clinical application (new)

```
Electron (electron/main.ts, preload.ts, contextIsolation=true)
  └─ React+TS frontend (frontend/) — REST + multipart to FastAPI
       └─ Clinical backend (backend/app/main.py, prefix /api/v1)
            ├─ api/: auth, doctors, patients, analyses, history,
            │        explainability, reports, system
            ├─ services/: model_manager (loads checkpoints ONCE),
            │             inference_service, explainability_service, report_service
            │             → all delegate to app.backend.* / src.* (adapters, no duplication)
            ├─ database.py / models_db.py (Doctor→Patient→XRayStudy→Analysis→Report, AuditLog)
            └─ security.py (bcrypt + JWT, ownership checks per row)
```

No mock AI anywhere: every probability/heatmap/uncertainty value is
computed live by the backend from the trained checkpoints.
