import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Users, ScanLine, CalendarClock, AlertTriangle, Plus } from "lucide-react";
import { api, type HistoryItem, type ModelStatus } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { UncertaintyBadge } from "../components/widgets";

type Dash = {
  doctor_name: string;
  stats: { total_patients: number; total_tests: number; tests_this_week: number; high_uncertainty: number };
  recent_tests: HistoryItem[];
  model: { loaded: boolean; device: string };
};

export function Dashboard() {
  const { doctor, token } = useAuth();
  const [dash, setDash] = useState<Dash | null>(null);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.get<Dash>("/api/v1/dashboard", token).then(setDash).catch((e) => setError(e.message));
    api.get<ModelStatus>("/api/v1/models/status").then(setModel).catch(() => undefined);
  }, [token]);

  const hour = new Date().getHours();
  const greet = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";

  if (error) return <div className="card p-6 text-rose-700">{error}</div>;
  if (!dash) return <div className="grid gap-3">{[1, 2, 3].map((i) => <div key={i} className="card h-24 animate-pulse" />)}</div>;

  const stats = [
    { icon: Users, label: "Total Patients", value: dash.stats.total_patients },
    { icon: ScanLine, label: "Total X-Ray Tests", value: dash.stats.total_tests },
    { icon: CalendarClock, label: "Tests This Week", value: dash.stats.tests_this_week },
    { icon: AlertTriangle, label: "High-Uncertainty Cases", value: dash.stats.high_uncertainty },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{greet}, Dr. {doctor?.full_name ?? dash.doctor_name}</h1>
        <p className="text-slate-500">Review chest X-rays with AI-assisted analysis.</p>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="card p-4">
            <s.icon size={18} className="text-brand-600" />
            <div className="mt-2 text-2xl font-bold tabular-nums">{s.value}</div>
            <div className="text-xs text-slate-500">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        <Link to="/xray-test" className="btn-primary inline-flex items-center gap-2"><Plus size={16} /> New X-Ray Test</Link>
        <Link to="/patients" className="btn-ghost">Add Patient</Link>
        <Link to="/history" className="btn-ghost">View History</Link>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          <h2 className="font-semibold">Recent Tests</h2>
          {dash.recent_tests.length === 0 ? (
            <p className="mt-3 text-sm text-slate-500">No examinations yet. Start a new X-ray test to see AI-assisted results here.</p>
          ) : (
            <ul className="mt-3 divide-y divide-slate-100">
              {dash.recent_tests.map((t) => (
                <li key={t.analysis_id} className="flex items-center justify-between gap-3 py-2.5">
                  <div className="min-w-0">
                    <Link to={`/analysis/${t.analysis_id}`} className="truncate font-medium text-brand-700 hover:underline">{t.patient_name}</Link>
                    <div className="text-xs text-slate-500">{t.primary_class} · {(t.confidence * 100).toFixed(1)}% · {t.created_at?.slice(0, 10)}</div>
                  </div>
                  <UncertaintyBadge level={t.uncertainty_level} />
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="card space-y-2 p-5">
          <h2 className="font-semibold">Model Status</h2>
          <div className="text-sm">Respira AI: <span className={`font-semibold ${model?.model_loaded ? "text-emerald-600" : "text-rose-600"}`}>{model?.model_loaded ? "Loaded" : "Not loaded"}</span></div>
          <div className="text-sm">Backend: <span className="font-semibold text-emerald-600">ok</span></div>
          <div className="text-sm">Device: <span className="font-semibold">{model?.device ?? dash.model.device}{model?.gpu_name ? ` (${model.gpu_name})` : ""}</span></div>
          <Link to="/models" className="text-sm font-semibold text-brand-700 hover:underline">Details →</Link>
        </div>
      </div>
    </div>
  );
}
