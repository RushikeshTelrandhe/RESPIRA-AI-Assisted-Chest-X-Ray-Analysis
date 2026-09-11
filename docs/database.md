# Respira — Database (`docs/database.md`)

Engine: SQLAlchemy 2. `DATABASE_URL` unset → SQLite fallback
`backend_storage/respira.db`; set `postgresql+psycopg://…` for Postgres
(tables auto-create via `Base.metadata.create_all`).

Tables:

- `doctors` — account + profile (passwords bcrypt-hashed, never plaintext)
- `patients` — `doctor_id` owner; `archived` soft-delete
- `xray_studies` — `doctor_id`, `patient_id`, file metadata, relative `image_path`
- `analyses` — `doctor_id`, `patient_id`, `study_id`, primary class, confidence,
  uncertainties, probabilities/weights JSON, model version, device, real timings,
  relative explainability image paths
- `reports` — `doctor_id`, `analysis_id`, relative PDF path
- `audit_logs` — login/logout, patient_created/updated/archived, xray_analyzed,
  report_generated (no images, no passwords, no patient data in logs)

Every query filters by the JWT doctor id; cross-doctor access returns 404.
