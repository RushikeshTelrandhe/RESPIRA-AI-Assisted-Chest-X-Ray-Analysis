# Respira — ML Inference

The clinical backend **imports** the research implementation; it does not
re-implement it:

- `backend/app/services/model_manager.py` → loads `app.backend.pipeline.RespiraPipeline`
  + `app.backend.gradcam.GradCAMGenerator` **once** at startup (CUDA if available).
- `backend/app/services/inference_service.py` → `validate_image()` (PNG/JPG magic
  bytes, extension, 15 MB, PIL verify) then `pipeline.predict_image()` inside
  `torch.inference_mode()` with `model.eval()` (set at load). Returns real
  per-class sigmoid probabilities, binary-entropy uncertainties, margin, and
  **measured** preprocessing/inference timings.
- Preprocessing is exactly the research chain (`app.backend.preprocessing`):
  RGB→gray → CLAHE (clip 2.0, 8×8) → OTSU lung mask + morphology + fallback
  ellipse → resize 224 → RGB → ImageNet normalize.
- Uncertainty = normalized binary entropy per class (÷ ln 2), sample mean,
  margin = 1 − (top1 − top2); levels Low <0.20 / Moderate <0.40 / High.
  Same formula as `src/inference/final_prediction.py`.
- Intermediate fusion modules (projection/cross-attention/disease/relationship)
  have no persisted weights by design → recreated deterministically (seed 42),
  exactly as the research app does.

See also: `docs/explainability.md`.
