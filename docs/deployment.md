# Respira — Deployment

- Single host: uvicorn backend + `vite build` static served by any web server
  (or Electron wrapper). `docker-compose.yml` provides Postgres + backend
  (mount `outputs/` read-only, persist `backend_storage/`).
- Env: `JWT_SECRET` (required, random), `DATABASE_URL`, `FRONTEND_URL`,
  `RETAIN_IMAGES`, `MAX_UPLOAD_MB`. Research metrics on the landing page are
  read live from `outputs/final_prediction/reports/final_metrics.json`.
- GPU: install the CUDA torch build; the backend selects CUDA automatically and
  reports the device/GPU name in `/models/status` and every report.
