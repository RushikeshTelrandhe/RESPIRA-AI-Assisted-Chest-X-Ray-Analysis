import { Link } from "react-router-dom";
import { Stethoscope, ScanLine, Brain, Eye, ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type ModelStatus } from "../services/api";

export function Landing() {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    api.get<ModelStatus>("/api/v1/models/status").then((s) => {
      if (s.research_evaluation) setMetrics(s.research_evaluation as Record<string, unknown>);
    }).catch(() => undefined);
  }, []);
  const pct = (v: unknown) => typeof v === "number" ? `${(v * 100).toFixed(2)}%` : "—";

  return (
    <div className="min-h-screen bg-white">
      <header className="mx-auto flex max-w-6xl items-center justify-between p-5">
        <div className="flex items-center gap-2">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-brand-600 text-white"><Stethoscope size={20} /></span>
          <span className="font-bold tracking-wide">RESPIRA</span>
        </div>
        <div className="flex gap-2">
          <Link to="/login" className="btn-ghost">Doctor Login</Link>
          <Link to="/signup" className="btn-primary">Create Doctor Account</Link>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-5 pb-14 pt-10 text-center">
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">RESPIRA</h1>
        <p className="mt-2 text-xl text-slate-600">AI-Assisted Chest X-Ray Analysis</p>
        <p className="mt-4 text-slate-500">“Understand. Explain. Review.”</p>
        <div className="mt-6 flex justify-center gap-3">
          <Link to="/login" className="btn-primary inline-flex items-center gap-2">Doctor Login <ArrowRight size={16} /></Link>
          <Link to="/signup" className="btn-ghost">Create Doctor Account</Link>
        </div>
      </section>

      <section className="border-t border-slate-100 bg-slate-50/60">
        <div className="mx-auto grid max-w-6xl gap-4 px-5 py-12 sm:grid-cols-2 lg:grid-cols-5">
          {[["X-Ray", "Upload a chest radiograph"], ["AI Analysis", "DenseNet + ViT fusion pipeline"], ["Disease Prediction", "Six-class probability output"], ["Uncertainty", "Entropy-based confidence signal"], ["Explainability", "Grad-CAM + ViT attention"]].map(([t, d]) => (
            <div key={t} className="card p-4"><div className="font-semibold">{t}</div><div className="text-sm text-slate-500">{d}</div></div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 py-12">
        <h2 className="text-2xl font-bold">AI Architecture</h2>
        <p className="mt-2 text-slate-600">DenseNet121 + ViT + Bidirectional Cross-Attention + Disease-Conditioned Attention + Adaptive Disease Fusion + Disease Relationship Modeling + Multi-Label Classification.</p>
        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <div className="card p-5"><ScanLine className="text-brand-600" /><h3 className="mt-2 font-semibold">Prediction</h3><p className="text-sm text-slate-500">Six disease probabilities with confidence and uncertainty on every study.</p></div>
          <div className="card p-5"><Eye className="text-brand-600" /><h3 className="mt-2 font-semibold">Grad-CAM</h3><p className="text-sm text-slate-500">Spatial regions contributing to the CNN prediction, per disease.</p></div>
          <div className="card p-5"><Brain className="text-brand-600" /><h3 className="mt-2 font-semibold">ViT Attention</h3><p className="text-sm text-slate-500">Attention within the transformer representation, visualized over the X-ray.</p></div>
        </div>
      </section>

      <section className="border-t border-slate-100 bg-slate-50/60">
        <div className="mx-auto max-w-6xl px-5 py-12">
          <h2 className="text-2xl font-bold">Research evaluation results</h2>
          <p className="text-sm text-slate-500">Measured on the held-out research test set. Not a clinical guarantee.</p>
          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-5">
            {[["Accuracy", metrics?.accuracy], ["Macro Precision", metrics?.precision_macro], ["Macro Recall", metrics?.recall_macro], ["Macro F1", metrics?.f1_macro], ["ROC-AUC", metrics?.roc_auc_macro]].map(([t, v]) => (
              <div key={String(t)} className="card p-4 text-center"><div className="text-2xl font-bold text-brand-700">{pct(v)}</div><div className="text-xs text-slate-500">{String(t)}</div></div>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t border-slate-100 py-6 text-center text-sm text-slate-500">Research Project · Respira · AI-Assisted Chest X-Ray Analysis</footer>
    </div>
  );
}
