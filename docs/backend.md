# Respira — Backend API (`/api/v1`)

Auth: `Authorization: Bearer <JWT>` (8 h expiry). All clinical routes verify
ownership server-side; `doctor_id`/`patient_id` from the client are never trusted.

| Method | Path | Description |
|---|---|---|
| POST | /auth/signup | Doctor signup (name, license, email, phone, hospital, specialization, password+confirm) |
| POST | /auth/login | Email+password → JWT |
| POST | /auth/logout | Audit-logged logout |
| GET | /auth/me | Current doctor |
| GET/PUT | /doctors/profile | Read/update profile; POST /doctors/change-password |
| GET/POST | /patients | List (search `q`) / create (own doctor only) |
| GET/PUT/DELETE | /patients/{id} | Read/update/archive (ownership checked, confirm in UI) |
| POST | /xray/upload | Validate PNG/JPG (magic bytes, ≤15 MB, PIL verify) → preview metadata |
| POST | /analyze | `patient_id` + file → REAL pipeline → persists Study+Analysis, returns structured result |
| GET | /analysis/{id} | Full persisted result |
| GET | /history | Filters (patient, disease, uncertainty, q), sort, pagination |
| GET | /patients/{id}/history | Chronological patient history |
| POST | /explainability/gradcam | `target_class` 0–5 + `analysis_id` or file → real Grad-CAM data URLs |
| POST | /explainability/vit-attention | `analysis_id` or file → real 14×14 map + overlays |
| POST | /reports/{analysis_id} | Generate PDF report |
| GET | /reports/{analysis_id}/pdf | Download PDF |
| GET | /reports/{analysis_id} | JSON export |
| GET | /reports | Report list |
| GET | /dashboard | Greeting stats (real DB counts) + recent tests + model flag |
| GET | /health | Liveness + model flag |
| GET | /models/status | Architecture, device/GPU, checkpoint state, verified research metrics |

Errors never leak tracebacks: friendly messages (e.g. “Respira could not
analyze this image…”). See `backend/app/main.py`.
