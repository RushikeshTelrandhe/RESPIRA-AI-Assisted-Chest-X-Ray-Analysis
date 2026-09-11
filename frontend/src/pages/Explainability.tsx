import { useState } from "react";
import { useParams } from "react-router-dom";
import { api, DISEASES } from "../services/api";
import { useAuth } from "../context/AuthContext";

type Gx = { image: string; heatmap: string; overlay: string; target_class?: string; attention_map?: number[][]; explanation?: string };

export function Explainability() {
  const { id } = useParams();
  const { token } = useAuth();
  const [tab, setTab] = useState<"overview" | "gradcam" | "vit">("overview");
  const [disease, setDisease] = useState(0);
  const [opacity, setOpacity] = useState(0.5);
  const [mode, setMode] = useState<"overlay" | "side">("overlay");
  const [grad, setGrad] = useState<Gx | null>(null);
  const [vit, setVit] = useState<Gx | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const loadGrad = async (idx: number) => {
    setBusy("gradcam"); setError("");
    try {
      const form = new FormData();
      form.append("target_class", String(idx));
      if (id) form.append("analysis_id", id);
      setGrad(await api.postForm<Gx>("/api/v1/explainability/gradcam", form, token));
    } catch (e) { setError(e instanceof Error ? e.message : "Grad-CAM failed"); }
    finally { setBusy(""); }
  };
  const loadVit = async () => {
    setBusy("vit"); setError("");
    try {
      const form = new FormData();
      if (id) form.append("analysis_id", id);
      setVit(await api.postForm<Gx>("/api/v1/explainability/vit-attention", form, token));
    } catch (e) { setError(e instanceof Error ? e.message : "ViT attention failed"); }
    finally { setBusy(""); }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Explainability</h1>
      <div className="flex gap-1" role="tablist">
        {(["overview", "gradcam", "vit"] as const).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
            className={`rounded-lg px-4 py-2 text-sm font-medium ${tab === t ? "bg-brand-600 text-white" : "bg-white border border-slate-200 text-slate-600"}`}>
            {t === "overview" ? "Overview" : t === "gradcam" ? "Grad-CAM" : "ViT Attention"}</button>
        ))}
      </div>
      {error && <div role="alert" className="card border-rose-200 p-3 text-sm text-rose-700">{error}</div>}

      {tab === "overview" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="card space-y-2 p-5">
            <h2 className="font-semibold">Grad-CAM vs ViT Attention</h2>
            <p className="text-sm text-slate-600">Grad-CAM highlights spatial regions contributing to the CNN prediction, while ViT attention visualizes attention within the transformer representation.</p>
            <div className="flex gap-2">
              <button className="btn-primary" onClick={() => { setTab("gradcam"); void loadGrad(disease); }}>View Grad-CAM</button>
              <button className="btn-ghost" onClick={() => { setTab("vit"); void loadVit(); }}>View ViT Attention</button>
            </div>
          </div>
          <div className="card p-5 text-sm text-slate-600">Select a disease to explain a specific class. Heatmaps are computed live by the backend from the trained checkpoints — never mocked.</div>
        </div>
      )}

      {tab === "gradcam" && (
        <div className="card space-y-3 p-5">
          <div className="flex flex-wrap items-center gap-3">
            <label className="label" htmlFor="dis">Disease</label>
            <select id="dis" className="input max-w-xs" value={disease} onChange={(e) => { const v = Number(e.target.value); setDisease(v); void loadGrad(v); }}>
              {DISEASES.map((d, i) => <option key={d} value={i}>{d}</option>)}
            </select>
            <label className="label" htmlFor="op">Overlay opacity</label>
            <input id="op" type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Heatmap opacity" />
            <select className="input max-w-40" value={mode} onChange={(e) => setMode(e.target.value as "overlay" | "side")} aria-label="View mode">
              <option value="overlay">Overlay</option><option value="side">Side-by-side</option>
            </select>
            <button className="btn-ghost" onClick={() => void loadGrad(disease)}>{busy === "gradcam" ? "Loading…" : "Reload"}</button>
          </div>
          {!grad ? <p className="text-sm text-slate-500">{busy === "gradcam" ? "Generating Grad-CAM…" : "No heatmap yet."}</p> : mode === "overlay" ? (
            <div className="relative mx-auto max-w-md">
              <img src={grad.image} alt="Original X-ray" className="w-full rounded-lg" />
              <img src={grad.heatmap} alt={`Grad-CAM heatmap for ${DISEASES[disease]}`} className="absolute inset-0 w-full rounded-lg" style={{ opacity }} />
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-3">
              <figure><img src={grad.image} alt="Original X-ray" className="rounded-lg border" /><figcaption className="text-xs text-slate-500">Original</figcaption></figure>
              <figure><img src={grad.heatmap} alt="Grad-CAM heatmap" className="rounded-lg border" /><figcaption className="text-xs text-slate-500">Heatmap</figcaption></figure>
              <figure><img src={grad.overlay} alt="Grad-CAM overlay" className="rounded-lg border" /><figcaption className="text-xs text-slate-500">Overlay</figcaption></figure>
            </div>
          )}
        </div>
      )}

      {tab === "vit" && (
        <div className="card space-y-3 p-5">
          <div className="flex gap-2">
            <button className="btn-primary" onClick={() => void loadVit()}>{busy === "vit" ? "Loading…" : vit ? "Reload" : "Generate ViT attention"}</button>
            <select className="input max-w-40" value={mode} onChange={(e) => setMode(e.target.value as "overlay" | "side")} aria-label="View mode">
              <option value="overlay">Overlay</option><option value="side">Side-by-side</option>
            </select>
          </div>
          {!vit ? <p className="text-sm text-slate-500">{busy === "vit" ? "Generating ViT attention…" : "No attention map yet."}</p> : mode === "overlay" ? (
            <img src={vit.overlay} alt="ViT attention overlay" className="mx-auto max-w-md rounded-lg border" />
          ) : (
            <div className="grid gap-3 sm:grid-cols-3">
              <figure><img src={vit.image} alt="Original X-ray" className="rounded-lg border" /><figcaption className="text-xs text-slate-500">Original</figcaption></figure>
              <figure><img src={vit.heatmap} alt="ViT attention map" className="rounded-lg border" /><figcaption className="text-xs text-slate-500">Attention map</figcaption></figure>
              <figure><img src={vit.overlay} alt="ViT attention overlay" className="rounded-lg border" /><figcaption className="text-xs text-slate-500">Overlay</figcaption></figure>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
