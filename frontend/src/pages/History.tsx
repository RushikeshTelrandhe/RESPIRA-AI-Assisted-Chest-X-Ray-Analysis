import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, DISEASES, type HistoryItem } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { EmptyState, UncertaintyBadge } from "../components/widgets";

export function History() {
  const { token } = useAuth();
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [q, setQ] = useState("");
  const [disease, setDisease] = useState("");
  const [unc, setUnc] = useState("");
  const [page, setPage] = useState(1);

  const load = async () => {
    const r = await api.get<{ total: number; items: HistoryItem[] }>(
      `/api/v1/history?q=${encodeURIComponent(q)}&disease=${disease}&uncertainty=${unc}&page=${page}`, token);
    setItems(r.items); setTotal(r.total);
  };
  useEffect(() => { void load(); }, [page]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Test History</h1>
      <div className="card flex flex-wrap gap-2 p-4">
        <input className="input max-w-56" placeholder="Search patient…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search history" />
        <select className="input max-w-56" value={disease} onChange={(e) => setDisease(e.target.value)} aria-label="Filter by disease">
          <option value="">All diseases</option>{DISEASES.map((d) => <option key={d}>{d}</option>)}
        </select>
        <select className="input max-w-44" value={unc} onChange={(e) => setUnc(e.target.value)} aria-label="Filter by uncertainty">
          <option value="">All uncertainty</option><option>Low</option><option>Moderate</option><option>High</option>
        </select>
        <button className="btn-primary" onClick={() => { setPage(1); void load(); }}>Apply</button>
      </div>
      {items.length === 0 ? <EmptyState title="No tests found" hint="Run an X-ray test to populate history." /> : (
        <div className="card overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead><tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500">
              <th className="p-3">Patient</th><th className="p-3">Date</th><th className="p-3">Prediction</th>
              <th className="p-3">Confidence</th><th className="p-3">Uncertainty</th><th className="p-3">Actions</th>
            </tr></thead>
            <tbody>
              {items.map((t) => (
                <tr key={t.analysis_id} className="border-b border-slate-100 last:border-0">
                  <td className="p-3 font-medium">{t.patient_name}</td>
                  <td className="p-3 tabular-nums">{t.created_at?.slice(0, 10)}</td>
                  <td className="p-3">{t.primary_class}</td>
                  <td className="p-3 tabular-nums">{(t.confidence * 100).toFixed(1)}%</td>
                  <td className="p-3"><UncertaintyBadge level={t.uncertainty_level} /></td>
                  <td className="p-3 flex gap-2">
                    <Link to={`/analysis/${t.analysis_id}`} className="font-semibold text-brand-700 hover:underline">View</Link>
                    <Link to={`/analysis/${t.analysis_id}/explainability`} className="font-semibold text-brand-700 hover:underline">Explain</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="flex items-center gap-3 text-sm text-slate-500">
        <span>Total: {total}</span>
        <button className="btn-ghost !py-1.5" disabled={page <= 1} onClick={() => setPage(page - 1)}>Prev</button>
        <span>Page {page}</span>
        <button className="btn-ghost !py-1.5" disabled={items.length < 20} onClick={() => setPage(page + 1)}>Next</button>
      </div>
    </div>
  );
}
