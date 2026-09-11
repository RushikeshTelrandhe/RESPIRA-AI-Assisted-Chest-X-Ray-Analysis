import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { api, type AnalysisDetail } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { DiseaseBars, Disclaimer, UncertaintyBadge, UncertaintyGauge } from "../components/widgets";

export function AnalysisDetailPage() {
  const { id } = useParams();
  const { token } = useAuth();
  const [a, setA] = useState<AnalysisDetail | null>(null);
  const [error, setError] = useState("");
  const [reportMsg, setReportMsg] = useState("");
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    api.get<AnalysisDetail>(`/api/v1/analysis/${id}`, token).then(setA).catch((e) => setError(e.message));
  }, [id, token]);

  const makeReport = async () => {
    setReportMsg("");
    try {
      await api.post(`/api/v1/reports/${id}`, {}, token);
      setReportMsg("Report generated. Use Export PDF below.");
    } catch (e) { setReportMsg(e instanceof Error ? e.message : "Report failed"); }
  };

  const exportPdf = async () => {
    if (!a) return;
    setReportMsg("");
    setDownloading(true);
    try {
      await api.downloadPdf(`/api/v1/reports/${a.analysis_id}/pdf`, `respira-report-${a.analysis_id.slice(0, 8)}.pdf`, token);
    } catch (e) { setReportMsg(e instanceof Error ? e.message : "Download failed"); }
    finally { setDownloading(false); }
  };

  if (error) return <div className="card p-6 text-rose-700">{error}</div>;
  if (!a) return <div className="card h-64 animate-pulse" />;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold">AI Prediction — {a.patient.name}</h1>
        <div className="flex gap-2">
          <Link to={`/analysis/${a.analysis_id}/explainability`} className="btn-primary">Explainability</Link>
          <button className="btn-ghost" onClick={() => void makeReport()}>Generate Report</button>
          <button className="btn-ghost disabled:opacity-50" disabled={downloading} onClick={() => void exportPdf()}>
            {downloading ? "Downloading…" : "Export PDF"}
          </button>
        </div>
      </div>
      {reportMsg && <div className="card p-3 text-sm">{reportMsg}</div>}

      <div className="card space-y-2 p-5">
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Primary AI finding</div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-2xl font-bold">{a.prediction.primary_class}</span>
          <UncertaintyBadge level={a.prediction.uncertainty_level} />
        </div>
        <div className="grid gap-2 text-sm sm:grid-cols-3">
          <div>Model Confidence: <b className="tabular-nums">{(a.prediction.confidence * 100).toFixed(1)}%</b></div>
          <div>Model Uncertainty: <b className="tabular-nums">{a.prediction.uncertainty.toFixed(3)}</b></div>
          <div>Margin: <b className="tabular-nums">{a.prediction.margin_uncertainty.toFixed(3)}</b></div>
        </div>
        <UncertaintyGauge value={a.prediction.uncertainty} />
        <p className="text-xs text-slate-500">Uncertainty reflects the model&apos;s internal confidence and does not measure clinical risk by itself.</p>
      </div>

      <div className="card space-y-4 p-5">
        <h2 className="font-semibold">Disease probabilities</h2>
        <DiseaseBars diseases={a.diseases} />
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={a.diseases.map((d) => ({ name: d.name, p: +(d.probability * 100).toFixed(1) }))} layout="vertical">
              <XAxis type="number" domain={[0, 100]} />
              <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [`${v}%`, "Probability"]} />
              <Bar dataKey="p" fill="#146684" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card p-5 text-sm text-slate-600">
        Model {a.model.version} · Device {a.model.device} · Inference {a.timing.inference_ms.toFixed(0)} ms · Total {a.timing.total_ms.toFixed(0)} ms
      </div>
      <Disclaimer />
    </div>
  );
}
