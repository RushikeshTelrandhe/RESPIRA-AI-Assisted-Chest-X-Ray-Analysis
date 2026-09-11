# Respira — Setup

## Prerequisites

- Python 3.11+ with the research env (`requirements.txt` / `requirements-app.txt` installed)
- Node 20+, checkpoints present under `outputs/` (see architecture.md)
- Optional: CUDA GPU (used automatically when available), Postgres 16

## 1. Backend

```powershell
copy .env.example .env   # set JWT_SECRET
pip install -r requirements-app.txt -r requirements-backend.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

Health: `GET http://127.0.0.1:8000/api/v1/health`
Model status: `GET http://127.0.0.1:8000/api/v1/models/status`
First startup loads ~1.1 GB of weights (one-time, then cached in process).

## 2. Frontend

```powershell
cd frontend
npm install
npm run dev     # http://localhost:5173 (proxies /api → backend)
```

Production: `npm run build` → `frontend/dist/`.

## 3. Desktop (Electron)

```powershell
cd frontend; npm run build; cd ..
# backend must be running (see step 1), then:
cd electron; npm install; npm run build
npx electron ./dist/main.js
```

`npm run dist` produces the Windows installer via electron-builder.

## 4. Database

Default: SQLite at `backend_storage/respira.db` (created on first startup).
Postgres: `docker compose up -d db`, set
`DATABASE_URL=postgresql+psycopg://respira:respira@localhost:5432/respira`
(requires `psycopg[binary]`), restart the backend — tables auto-create.

## 5. Tests

```powershell
python -m pytest tests/ -q        # backend + REAL model e2e (needs checkpoints + GPU/CPU)
cd frontend; npx vitest run       # frontend unit tests
```

## Troubleshooting

- `Respira AI model is not loaded` → check `outputs/*/checkpoints/best_model.pth` exist.
- 401s → JWT_SECRET changed between restarts invalidates old tokens; sign in again.
- CUDA OOM → backend falls back gracefully per request; restart uvicorn to clear.
- Port clash → set BACKEND_PORT / Vite port accordingly.
