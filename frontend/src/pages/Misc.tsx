import { useEffect, useState } from "react";
import { api, type ModelStatus } from "../services/api";

export function Models() {
  const [m, setM] = useState<ModelStatus | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api.get<ModelStatus>("/api/v1/models/status").then(setM).catch((e) => setError(e.message));
  }, []);
  if (error) return <div className="card p-6 text-rose-700">{error}</div>;
  if (!m) return <div className="card h-48 animate-pulse" />;
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <h1 className="text-2xl font-bold">Model Status</h1>
      <div className="card grid gap-2 p-5 text-sm sm:grid-cols-2">
        <div>Respira AI: <b className={m.model_loaded ? "text-emerald-600" : "text-rose-600"}>{m.respira_ai}</b></div>
        <div>Backend: <b>{m.backend}</b></div>
        <div>Device: <b>{m.device}{m.gpu_name ? ` (${m.gpu_name})` : ""}</b></div>
        <div>Classes: <b>{m.classes.join(", ")}</b></div>
      </div>
      <div className="card space-y-1 p-5">
        <h2 className="font-semibold">Architecture</h2>
        {Object.entries(m.architecture).map(([k, v]) => (
          <div key={k} className="flex justify-between text-sm border-b border-slate-100 py-1.5 last:border-0">
            <span className="text-slate-600">{k}</span><b>{v}</b>
          </div>
        ))}
      </div>
      {m.error && <div className="card border-rose-200 p-4 text-sm text-rose-700">{m.error}</div>}
    </div>
  );
}

export function Settings() {
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <h1 className="text-2xl font-bold">Settings</h1>
      <div className="card p-5 text-sm text-slate-600">
        Theme and notification preferences can be managed from your Profile. Analysis settings (device,
        storage policy, upload limits) are configured server-side via environment variables — see docs/setup.md.
      </div>
    </div>
  );
}

export function Help() {
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <h1 className="text-2xl font-bold">Help</h1>
      <div className="card space-y-2 p-5 text-sm text-slate-600">
        <p><b>Workflow:</b> Dashboard → New X-Ray → select patient → upload PNG/JPG → Analyze → review prediction, uncertainty, Grad-CAM and ViT attention → generate report.</p>
        <p><b>Troubleshooting:</b> if analysis fails, verify the file is a valid PNG/JPG under 15 MB. If the model shows “not loaded”, check the backend logs and that checkpoint files exist under outputs/.</p>
        <p><b>Support:</b> see docs/ in the repository for setup, security, and deployment guides.</p>
      </div>
    </div>
  );
}
