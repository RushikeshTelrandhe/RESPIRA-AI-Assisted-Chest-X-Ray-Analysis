# Respira — Explainability

- **Grad-CAM** (`backend/app/services/explainability_service.py::gradcam_for` →
  `app.backend.gradcam`): last Conv2d of the EfficientNet-B0 branch, per-disease
  `target_class` 0–5, ReLU + min-max norm, 224×224 heatmap + overlay.
  Matches `src/explainability/grad_cam.py` methodology.
- **ViT attention** (`vit_attention_for` → `pipeline.capture_cross_attention` +
  `app.backend.vit_attention.create_vit_attention_map`): live `vit_to_cnn`
  cross-attention, mean over CNN tokens → 196 patch scores → 14×14 map →
  bilinear upscale + overlay. Matches STEP 8.8
  (`src/analysis/vit_attention_visualization.py`).
- Frontend `Explainability` page: Overview / Grad-CAM / ViT tabs, disease
  selector (re-queries backend), opacity slider, overlay vs side-by-side,
  comparison note: “Grad-CAM highlights spatial regions contributing to the
  CNN prediction, while ViT attention visualizes attention within the
  transformer representation.”
