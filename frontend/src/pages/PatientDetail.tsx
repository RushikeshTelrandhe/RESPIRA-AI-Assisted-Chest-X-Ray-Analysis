import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type HistoryItem, type Patient } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { UncertaintyBadge } from "../components/widgets";

export function PatientDetail() {
  const { id } = useParams();
  const { token } = useAuth();
  const navigate = useNavigate();
  const [p, setP] = useState<Patient | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [tab, setTab] = useState("overview");
  const [error, setError] = useState("");
  const [notes, setNotes] = useState("");

  useEffect(() => {
    api.get<Patient>(`/api/v1/patients/${id}`, token).then((v) => { setP(v); setNotes(v.notes); }).catch((e) => setError(e.message));
    api.get<HistoryItem[]>(`/api/v1/patients/${id}/history`, token).then(setHistory).catch(() => undefined);
  }, [id, token]);

  const saveNotes = async () => {
    try { const v = await api.put<Patient>(`/api/v1/patients/${id}`, { notes }, token); setP(v); }
    catch (e) { setError(e instanceof Error ? e.message : "Save failed"); }
  };
  const remove = async () => {
    if (!confirm("Archive this patient? Their test history is preserved.")) return;
    await api.del(`/api/v1/patients/${id}`, token);
    navigate("/patients");
  };

  if (error && !p) return <div className="card p-6 text-rose-700">{error}</div>;
  if (!p) return <div className="card h-40 animate-pulse" />;

  const tabs = ["overview", "xray-tests", "ai-results", "reports", "notes"];
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{p.full_name}</h1>
        <div className="flex gap-2">
          <Link to={`/xray-test?patient=${p.id}`} className="btn-primary">New X-Ray</Link>
          <button className="btn-ghost" onClick={() => void remove()}>Archive</button>
        </div>
      </div>
      <div className="flex gap-1 overflow-x-auto" role="tablist">
        {tabs.map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
            className={`rounded-lg px-3 py-2 text-sm font-medium ${tab === t ? "bg-brand-600 text-white" : "bg-white border border-slate-200 text-slate-600"}`}>{t.replace("-", " ")}</button>
        ))}
      </div>

      {tab === "overview" && (
        <div className="card grid gap-2 p-5 text-sm sm:grid-cols-2">
          <div><span className="text-slate-500">Patient ID:</span> <b>{p.patient_code || p.id.slice(0, 8)}</b></div>
          <div><span className="text-slate-500">Age / Gender:</span> <b>{p.age || "—"} / {p.gender || "—"}</b></div>
          <div><span className="text-slate-500">Phone:</span> <b>{p.phone || "—"}</b></div>
          <div><span className="text-slate-500">Email:</span> <b>{p.email || "—"}</b></div>
          <div className="sm:col-span-2"><span className="text-slate-500">Medical history:</span> <b>{p.medical_history || "—"}</b></div>
        </div>
      )}
      {(tab === "xray-tests" || tab === "ai-results") && (
        <div className="card p-5">
          {history.length === 0 ? <p className="text-sm text-slate-500">No examinations yet.</p> : (
            <ul className="divide-y divide-slate-100">
              {history.map((h) => (
                <li key={h.analysis_id} className="flex items-center justify-between py-2.5">
                  <div>
                    <Link to={`/analysis/${h.analysis_id}`} className="font-medium text-brand-700 hover:underline">X-Ray · {h.created_at?.slice(0, 10)}</Link>
                    <div className="text-xs text-slate-500">{h.primary_class} · {(h.confidence * 100).toFixed(1)}% confidence</div>
                  </div>
                  <UncertaintyBadge level={h.uncertainty_level} />
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {tab === "reports" && (
        <div className="card space-y-2 p-5">
          {history.length === 0 ? <p className="text-sm text-slate-500">No reports yet.</p> :
            history.map((h) => (
              <div key={h.analysis_id} className="flex items-center justify-between text-sm">
                <span>Report · {h.created_at?.slice(0, 10)} · {h.primary_class}</span>
                <Link to={`/analysis/${h.analysis_id}`} className="font-semibold text-brand-700 hover:underline">Open →</Link>
              </div>
            ))}
        </div>
      )}
      {tab === "notes" && (
        <div className="card space-y-3 p-5">
          <label className="label" htmlFor="notes">Clinical notes</label>
          <textarea id="notes" className="input min-h-32" value={notes} onChange={(e) => setNotes(e.target.value)} />
          <div><button className="btn-primary" onClick={() => void saveNotes()}>Save notes</button></div>
          {error && <div className="text-sm text-rose-700">{error}</div>}
        </div>
      )}
    </div>
  );
}
