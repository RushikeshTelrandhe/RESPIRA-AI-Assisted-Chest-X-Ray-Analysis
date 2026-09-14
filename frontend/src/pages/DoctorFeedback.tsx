import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, ClipboardPen, Mail, MessageSquareText } from "lucide-react";
import { api, type HistoryItem } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useResource } from "../context/WorkspaceContext";
import { Alert, Empty, LoadingBlock, PageHeader } from "../components/UI";
import { SearchInput } from "../components/SearchInput";
import { dateLabel } from "../utils/display";
export function DoctorFeedback() {
  const { token } = useAuth();
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(1);
  useEffect(() => { const t = setTimeout(() => { setQuery(search.trim()); setPage(1); }, 250); return () => clearTimeout(t); }, [search]);
  const params = new URLSearchParams({ q: query, page: String(page) });
  const result = useResource<{ total: number; items: HistoryItem[] }>(`history:${params}`, () => api.get(`/api/v1/history?${params}`, token));
  return <div className="stack"><PageHeader eyebrow="Clinical review" title="Doctor feedback" description="Open a study to add notes, document agreement, and review your saved feedback." />
    <div className="review-hub-intro"><div><span className="review-symbol"><ClipboardPen size={24} /></span><h2>Your assessment belongs with the evidence.</h2><p>Clinical observations, follow-up notes and model feedback stay linked to the patient and study.</p></div><a className="review-support-link" href="mailto:respirahelp@gmail.com"><Mail size={20} /><span><strong>Contact the developers</strong><small>respirahelp@gmail.com</small></span><ArrowRight size={17} /></a></div>
    <div className="clinical-section-row"><div><h2>Choose a study</h2><p className="small muted">Search for the patient you want to review.</p></div><SearchInput value={search} onChange={setSearch} /></div>
    {result.loading ? <LoadingBlock label="Loading studies…" /> : result.error ? <Alert retry={result.reload}>{result.error}</Alert> : !result.data?.items.length ? <Empty title="No studies found" hint="Completed analyses will appear here for doctor review." /> : <div className="table-wrap"><table className="data-table"><thead><tr><th>Patient</th><th>Study date</th><th>Model output</th><th>Doctor feedback</th></tr></thead><tbody>{result.data.items.map(item => <tr key={item.analysis_id}><td><Link className="row-name" to={`/patients/${item.patient_id}`}>{item.patient_name}</Link></td><td>{dateLabel(item.created_at,true)}</td><td>{item.primary_class}</td><td><Link className="text-button" to={`/analysis/${item.analysis_id}#doctor-feedback`}><MessageSquareText size={16} /> Notes & feedback <ArrowRight size={14} /></Link></td></tr>)}</tbody></table></div>}
    <nav className="pagination" aria-label="Feedback study pages"><span>{result.data?.total ?? 0} studies · Page {page}</span><button className="btn-ghost" disabled={page <= 1 || result.loading} onClick={() => setPage(p => p - 1)}>Previous</button><button className="btn-ghost" disabled={page * 20 >= (result.data?.total ?? 0) || result.loading} onClick={() => setPage(p => p + 1)}>Next</button></nav>
  </div>;
}
