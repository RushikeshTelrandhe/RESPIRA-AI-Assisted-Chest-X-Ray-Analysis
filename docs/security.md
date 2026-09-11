# Respira — Security

- bcrypt hashing (salted), JWT HS256 8 h expiry, `HTTPBearer` guard.
- CORS allowlist (frontend origin), `X-Content-Type-Options`/`X-Frame-Options` headers.
- Upload validation: extension + magic bytes + PIL verify + 15 MB cap.
- Ownership authorization on every clinical row; filesystem paths never exposed
  (relative storage paths only; images served as data URLs or generated PDFs).
- Friendly error handler — no tracebacks reach the UI. Audit log for sensitive actions.
- Electron: `contextIsolation=true`, `nodeIntegration=false`, minimal preload bridge.
- Never trust `doctor_id`/`patient_id`/`analysis_id` from the client:
  all are cross-checked against the JWT identity in every endpoint.
