# RESPIRA — AI-Assisted Chest X-Ray Analysis

A professional AI-assisted radiology workstation for doctors, built on top of
the existing Respira research pipeline (EfficientNet-B0 + ViT-B/16 fusion with
bidirectional cross-attention, disease-conditioned attention, adaptive fusion,
relationship modeling, and multi-label classification heads).

**No mock AI.** Every prediction, confidence, uncertainty value, Grad-CAM
heatmap, and ViT attention map is computed live by the backend from the real
trained checkpoints under `outputs/`. The research code in `src/` and
`app/backend/` is imported as-is via thin service adapters.

Verified research test-set metrics (read live from
`outputs/final_prediction/reports/final_metrics.json`, shown on the landing
page as research results, not clinical guarantees): accuracy ≈ 88.08%,
macro precision ≈ 87.88%, macro recall ≈ 89.50%, macro F1 ≈ 88.62%,
ROC-AUC ≈ 97.94%.

## Repository layout

- `src/` — research pipeline (models, fusion, inference, analysis, explainability)
- `app/backend/` — original research inference service (imported, not rewritten)
- `backend/app/` — clinical backend: auth (JWT+bcrypt), patients, real analysis,
  history, explainability, PDF reports, audit log, SQLite→Postgres DB layer
- `frontend/` — React 19 + TypeScript + Vite + Tailwind + shadcn-style UI +
  lucide + framer-motion + recharts
- `electron/` — desktop shell (`contextIsolation=true`, `nodeIntegration=false`)
- `tests/` — backend pytest suite incl. REAL model end-to-end test
- `docs/` — architecture, setup, backend, frontend, database, ml-inference,
  explainability, security, electron, deployment
- `backend_storage/` — SQLite DB + stored studies/overlays/reports (git-ignored data)

## Quickstart

```powershell
copy .env.example .env   # set JWT_SECRET
pip install -r requirements-app.txt -r requirements-backend.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

```powershell
cd frontend; npm install; npm run dev   # http://localhost:5173
```

Desktop: build frontend (`npm run build`), run backend, then
`cd electron; npm install; npm run build; npx electron ./dist/main.js`.

## Tests

```powershell
python -m pytest tests/ -q        # auth, ownership, REAL inference/Grad-CAM/ViT/reports
cd frontend; npx vitest run       # frontend unit tests
```

## Clinical disclaimer

Respira provides AI-assisted analysis for research and clinical decision
support. Results should be reviewed by a qualified medical professional and
should not be considered a definitive diagnosis.
