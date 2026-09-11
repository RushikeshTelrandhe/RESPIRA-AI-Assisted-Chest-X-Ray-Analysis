# Respira — Electron

`electron/main.ts` loads `frontend/dist` (fallback: Vite dev server),
checks backend `/api/v1/health` over TCP, opens external links in the browser,
quits gracefully. `preload.ts` exposes only `{ version, backendUrl }`
(`contextIsolation=true`, `nodeIntegration=false`).
Build: `tsc -p tsconfig.json`, package with electron-builder (`npm run dist`).
The FastAPI backend runs as a separate process (bundled or sidecar) and serves
the real PyTorch pipeline over localhost.
