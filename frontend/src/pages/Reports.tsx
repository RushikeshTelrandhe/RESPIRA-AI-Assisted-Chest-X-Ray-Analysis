import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Download, FileText, ArrowRight, ClipboardPen, Info } from "lucide-react";
import { api } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useResource } from "../context/WorkspaceContext";
import { Alert, Empty, LoadingBlock, PageHeader } from "../components/UI";
import { SearchInput } from "../components/SearchInput";
import { dateLabel, errorText } from "../utils/display";
type Report = { report_id: string; analysis_id: string; created_at: string | null; };
export function Reports() {
  const { token } = useAuth();
  const reports = useResource<Report[]>("reports", () => api.get("/api/v1/reports", token));
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const sorted = useMemo(() => [...(reports.data || [])].sort((a,b) => (Date.parse(b.created_at || '') || 0) - (Date.parse(a.created_at || '') || 0)), [reports.data]);
  const visible = sorted.filter(r => `${r.report_id} ${r.analysis_id} ${dateLabel(r.created_at,true)}`.toLowerCase().includes(search.toLowerCase()));
  async function download(id: string) {
    if (busy) return;
    setBusy(id); setError("");
    try { await api.downloadPdf(`/api/v1/reports/${id}/pdf`, `respira-report-${id.slice(0,8)}.pdf`, token); }
    catch(e) { setError(errorText(e)); } finally { setBusy(""); }
  }
  return <div className="stack"><PageHeader eyebrow="Clinical documentation" title="Reports" description="Review the study, add your assessment, and download the generated PDF." action={<Link className="btn-ghost" to="/history">Browse studies <ArrowRight size={17} /></Link>} />
    <section className="report-summary-strip" aria-label="Report overview"><div><small>Generated reports</small><strong>{reports.loading ? '…' : reports.data ? sorted.length : 'Unavailable'}</strong></div><div><small>Latest document</small><strong>{sorted[0] ? dateLabel(sorted[0].created_at) : 'No documents yet'}</strong></div><div><small>Doctor assessment</small><strong><Link to="/feedback">Notes & feedback <ArrowRight size={15} /></Link></strong></div></section>
    <div className="clinical-section-row"><div><h2>Document archive</h2><p className="small muted">Most recent first</p></div><SearchInput label="Search reports" placeholder="Report ID, analysis ID or date…" value={search} onChange={setSearch} /></div>
    {error && <Alert>{error}</Alert>}
    {reports.loading ? <LoadingBlock label="Loading reports…" /> : reports.error ? <Alert retry={reports.reload}>{reports.error}</Alert> : !visible.length ? <Empty title={search ? 'No matching reports' : 'No generated reports yet'} hint={search ? 'Try a report ID, analysis ID or date.' : 'Open an analysis and choose Generate report.'} action={<Link className="btn-ghost" to="/history">Browse analyses</Link>} /> : <div className="table-wrap report-table"><table className="data-table"><thead><tr><th>Report</th><th>Created</th><th>Study</th><th>Doctor feedback</th><th>PDF</th></tr></thead><tbody>{visible.map(r => <tr key={r.report_id}><td><div className="report-document-title"><span><FileText size={21} /></span><span><strong>Report {r.report_id.slice(0,8)}</strong><small>Chest X-ray analysis</small></span></div></td><td>{dateLabel(r.created_at,true)}</td><td><Link className="text-button" to={`/analysis/${r.analysis_id}`}>Open study <ArrowRight size={15} /></Link></td><td><Link className="text-button" to={`/analysis/${r.analysis_id}#doctor-feedback`}><ClipboardPen size={16} /> Notes & feedback</Link></td><td><button className="btn-primary" disabled={!!busy} onClick={() => void download(r.analysis_id)}><Download size={16} />{busy === r.analysis_id ? 'Downloading…' : 'Download PDF'}</button></td></tr>)}</tbody></table></div>}
    <div className="report-context-note"><Info size={17} /><span>Doctor notes are saved with the patient record. Use <Link to="/feedback">Doctor feedback</Link> to review or export them separately from the model PDF.</span></div>
  </div>;
}
