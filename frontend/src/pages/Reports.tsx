import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { EmptyState } from "../components/widgets";

export function Reports() {
  const { token } = useAuth();
  const [items, setItems] = useState<{ report_id: string; analysis_id: string; created_at: string | null }[]>([]);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api.get<typeof items>("/api/v1/reports", token).then(setItems).catch(() => undefined);
  }, [token]);

  const download = async (analysisId: string) => {
    setError("");
    setBusyId(analysisId);
    try {
      await api.downloadPdf(`/api/v1/reports/${analysisId}/pdf`, `respira-report-${analysisId.slice(0, 8)}.pdf`, token);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed");
    } finally {
      setBusyId(null);
    }
  };
  if (items.length === 0) return (<div className="space-y-4"><h1 className="text-2xl font-bold">Reports</h1><EmptyState title="No reports yet" hint="Generate a report from any analysis." /></div>);
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Reports</h1>
      {error && <div className="card p-3 text-sm text-rose-700">{error}</div>}
      <ul className="grid gap-3">
        {items.map((r) => (
          <li key={r.report_id} className="card flex items-center justify-between p-4">
            <div className="text-sm"><b>Report {r.report_id.slice(0, 8)}</b><div className="text-slate-500">{r.created_at?.slice(0, 16)}</div></div>
            <div className="flex gap-3">
              <Link to={`/analysis/${r.analysis_id}`} className="font-semibold text-brand-700 hover:underline">Open analysis</Link>
              <button
                className="font-semibold text-brand-700 hover:underline disabled:opacity-50"
                disabled={busyId === r.analysis_id}
                onClick={() => void download(r.analysis_id)}
              >
                {busyId === r.analysis_id ? "Downloading…" : "Download PDF"}
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
