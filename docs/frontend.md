# Respira — Frontend

`frontend/` — React 19 + TypeScript + Vite + Tailwind v4 + lucide-react +
framer-motion + recharts + react-router-dom.
`VITE_API_URL` empty → same-origin (dev proxy `/api → 127.0.0.1:8000`).
Routes: `/ /login /signup /dashboard /patients /patients/:id /xray-test
/analysis/:id /analysis/:id/explainability /history /reports /profile
/settings /models /help`. Auth state in `AuthContext` (JWT in localStorage);
`ProtectedRoute` guards all clinical pages. All ML values are rendered from
backend responses — the frontend never computes predictions/heatmaps/uncertainty.
Unit tests: `npx vitest run`.
