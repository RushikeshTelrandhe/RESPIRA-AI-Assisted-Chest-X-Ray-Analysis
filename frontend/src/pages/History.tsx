import { MessageSquareText } from "lucide-react";
import { SearchInput } from "../components/SearchInput";
import { useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Search, SlidersHorizontal } from "lucide-react";
import { api, DISEASES, type HistoryItem } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useResource } from "../context/WorkspaceContext";
import { Alert, Empty, LoadingBlock, PageHeader } from "../components/UI";
import { UncertaintyBadge } from "../components/widgets";
import { dateLabel, score } from "../utils/display";
export function History() {
  const { token } = useAuth();
  const [params, setParams] = useSearchParams();
  const activeQ = params.get("q") || "",
    disease = params.get("disease") || "",
    uncertainty = params.get("uncertainty") || "",
    page = Math.max(1, Number(params.get("page") || 1) || 1);
  const [q, setQ] = useState(activeQ);
  const [diseaseDraft, setDisease] = useState(disease);
  const [uncDraft, setUnc] = useState(uncertainty);
  const query = new URLSearchParams({
    q: activeQ,
    disease,
    uncertainty,
    page: String(page),
  });
  const result = useResource<{ total: number; items: HistoryItem[] }>(
    "history:" + query.toString(),
    () => api.get("/api/v1/history?" + query.toString(), token),
  );
  function apply(e: FormEvent) {
    e.preventDefault();
    const next = new URLSearchParams();
    if (q.trim()) next.set("q", q.trim());
    if (diseaseDraft) next.set("disease", diseaseDraft);
    if (uncDraft) next.set("uncertainty", uncDraft);
    next.set("page", "1");
    setParams(next);
  }
  function go(nextPage: number) {
    const next = new URLSearchParams(params);
    next.set("page", String(nextPage));
    setParams(next);
  }
  const items = result.data?.items ?? [],
    total = result.data?.total ?? 0;
  return (
    <div className="stack">
      <PageHeader
        eyebrow="Review archive"
        title="Analysis history"
        description="Find previous RESPIRA outputs while keeping your filters in the page URL."
      />
      <form className="filter-bar clinical-filter-bar" onSubmit={apply}>
        <div className="filter-intro"><strong>Find a study</strong><span>Search and refine results</span></div>
        <SearchInput value={q} onChange={setQ} label="Search patient" placeholder="Search patient…" />
        <select
          className="input"
          aria-label="Filter by output class"
          value={diseaseDraft}
          onChange={(e) => setDisease(e.target.value)}
        >
          <option value="">All output classes</option>
          {DISEASES.map((d) => (
            <option key={d}>{d}</option>
          ))}
        </select>
        <select
          className="input"
          aria-label="Filter by uncertainty"
          value={uncDraft}
          onChange={(e) => setUnc(e.target.value)}
        >
          <option value="">All uncertainty</option>
          <option>Low</option>
          <option>Moderate</option>
          <option>High</option>
        </select>
        <button className="btn-primary">
          <SlidersHorizontal size={17} />
          Apply filters
        </button>
      </form>
      {result.loading ? (
        <LoadingBlock label="Loading analysis history…" />
      ) : result.error ? (
        <Alert retry={result.reload}>{result.error}</Alert>
      ) : items.length === 0 ? (
        <Empty
          title="No analyses found"
          hint={
            activeQ || disease || uncertainty
              ? "Change the filters and try again."
              : "Completed analyses will appear here."
          }
        />
      ) : (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Patient</th>
                <th>Date</th>
                <th>Primary model output</th>
                <th>Score</th>
                <th>Uncertainty</th>
                <th>Actions</th>
                <th>Doctor feedback</th>
              </tr>
            </thead>
            <tbody>
              {items.map((t) => (
                <tr key={t.analysis_id}>
                  <td>
                    <Link className="row-name" to={"/patients/" + t.patient_id}>
                      {t.patient_name}
                    </Link>
                  </td>
                  <td>{dateLabel(t.created_at, true)}</td>
                  <td>{t.primary_class || "Unavailable"}</td>
                  <td>{score(t.confidence)}</td>
                  <td>
                    <UncertaintyBadge level={t.uncertainty_level} />
                  </td>
                  <td>
                    <div className="table-actions">
                      <Link
                        className="text-button"
                        to={"/analysis/" + t.analysis_id}
                      >
                        Review
                      </Link>
                      <Link
                        className="text-button"
                        to={"/analysis/" + t.analysis_id + "/explainability"}
                      >
                        Explain
                      </Link>
                    </div>
                  </td>
                  <td><Link className="text-button" to={`/analysis/${t.analysis_id}#doctor-feedback`}><MessageSquareText size={16} /> Add / view</Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <nav className="pagination" aria-label="History pages">
        <span>
          {total + " result" + (total === 1 ? "" : "s") + " · Page " + page}
        </span>
        <button
          className="btn-ghost"
          disabled={page <= 1 || result.loading}
          onClick={() => go(page - 1)}
        >
          Previous
        </button>
        <button
          className="btn-ghost"
          disabled={page * 20 >= total || result.loading}
          onClick={() => go(page + 1)}
        >
          Next
        </button>
      </nav>
    </div>
  );
}
