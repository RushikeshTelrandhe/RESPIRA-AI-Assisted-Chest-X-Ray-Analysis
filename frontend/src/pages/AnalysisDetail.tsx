import { DoctorReview } from "../components/DoctorReview";
import { useState } from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import { Download, FilePlus2, Maximize2, Minimize2, ScanLine } from "lucide-react";
import { api, type AnalysisDetail } from "../services/api";
import { useAuth } from "../context/AuthContext";
import {
  usePreview,
  useResource,
  useWorkspace,
} from "../context/WorkspaceContext";
import { Alert, LoadingBlock, PageHeader, Tabs } from "../components/UI";
import {
  DiseaseBars,
  Disclaimer,
  UncertaintyBadge,
} from "../components/widgets";
import { XrayViewer } from "../components/XrayViewer";
import { dateLabel, errorText, numeric, score } from "../utils/display";
type Tab = "summary" | "visual" | "technical";
export function AnalysisDetailPage() {
  const { id = "" } = useParams();
  const { token } = useAuth();
  const { images, cache } = useWorkspace();
  const nav = useNavigate();
  const [params, setParams] = useSearchParams();
  const analysis = useResource<AnalysisDetail>(
    id ? "analysis:" + id : null,
    () => api.get("/api/v1/analysis/" + id, token),
  );
  const original = usePreview(images.get(id));
  const [message, setMessage] = useState("");
  const [working, setWorking] = useState("");
  const [focusReview, setFocusReview] = useState(false);
  const tab: Tab = params.get("view") === "technical" ? "technical" : "summary";
  function changeTab(next: Tab) {
    if (next === "visual") {
      nav("/analysis/" + id + "/explainability");
      return;
    }
    if (next === "technical") setParams({ view: "technical" });
    else setParams({});
  }
  async function report() {
    setWorking("report");
    setMessage("");
    try {
      await api.post("/api/v1/reports/" + id, {}, token);
      cache.invalidate("reports");
      setMessage("Report generated. You can download the PDF now.");
    } catch (e) {
      setMessage(errorText(e));
    } finally {
      setWorking("");
    }
  }
  async function pdf() {
    if (!analysis.data) return;
    setWorking("pdf");
    setMessage("");
    try {
      await api.downloadPdf(
        "/api/v1/reports/" + id + "/pdf",
        "respira-report-" + id.slice(0, 8) + ".pdf",
        token,
      );
    } catch (e) {
      setMessage(errorText(e));
    } finally {
      setWorking("");
    }
  }
  if (analysis.loading) return <LoadingBlock label="Loading study review…" />;
  if (analysis.error || !analysis.data)
    return (
      <Alert retry={analysis.reload}>
        {analysis.error || "This analysis is unavailable."}
      </Alert>
    );
  const a = analysis.data;
  const tabOptions = [
    { value: "summary" as const, label: "Summary" },
    { value: "visual" as const, label: "Visual explanation" },
    { value: "technical" as const, label: "Technical details" },
  ];
  return (
    <div className={`stack study-review-page${focusReview ? " focus-review" : ""}`}>
      <PageHeader
        eyebrow="Study review"
        title={a.patient.name}
        description={
          "Analysis " +
          a.analysis_id.slice(0, 8) +
          " · " +
          dateLabel(a.created_at || a.study.created_at, true)
        }
        action={
          <div className="page-actions">
            <button className="btn-ghost" onClick={() => setFocusReview((value) => !value)}>
              {focusReview ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
              {focusReview ? "Exit focus" : "Focus review"}
            </button>
            <button
              className="btn-ghost"
              disabled={Boolean(working)}
              onClick={() => void report()}
            >
              <FilePlus2 size={17} />
              {working === "report" ? "Generating…" : "Generate report"}
            </button>
            <button
              className="btn-primary"
              disabled={Boolean(working)}
              onClick={() => void pdf()}
            >
              <Download size={17} />
              {working === "pdf" ? "Downloading…" : "Export PDF"}
            </button>
          </div>
        }
      />
      {message && (
        <Alert
          kind={message.startsWith("Report generated") ? "success" : "error"}
        >
          {message}
        </Alert>
      )}
      <section className="study-banner">
        <div className="study-person">
          <span className="avatar avatar-lg">{a.patient.name.charAt(0)}</span>
          <div>
            <span>Patient</span>
            <strong>{a.patient.name}</strong>
          </div>
        </div>
        <div className="study-meta">
          <span>
            Study file<strong>{a.study.filename || "Unavailable"}</strong>
          </span>
          <span>
            Model<strong>{a.model.version || "Unavailable"}</strong>
          </span>
          <span>
            Backend device<strong>{a.model.device || "Unavailable"}</strong>
          </span>
        </div>
      </section>
      <Tabs
        id="analysis-tabs"
        options={tabOptions}
        value={tab}
        onChange={changeTab}
      />
      <section
        id="analysis-tabs-panel"
        role="tabpanel"
        aria-labelledby={"analysis-tabs-" + tab}
      >
        {tab === "summary" && (
          <div className={`review-grid${focusReview ? " review-grid-focus" : ""}`}>
            <div>
              {original ? (
                <XrayViewer original={original} />
              ) : (
                <div className="viewer">
                  <div className="viewer-toolbar">
                    <div>
                      <strong>Study viewer</strong>
                      <span>
                        Original image unavailable in this browser session
                      </span>
                    </div>
                  </div>
                  <div className="viewer-empty">
                    <ScanLine size={34} />
                    <strong>
                      Open Visual explanation to request study imagery
                    </strong>
                    <span>
                      Your original upload is no longer in this browser session.
                      Open an explanation to retrieve the study imagery.
                    </span>
                    <Link
                      className="btn-light"
                      to={"/analysis/" + id + "/explainability"}
                    >
                      Open visual explanation
                    </Link>
                  </div>
                </div>
              )}
            </div>
            <aside className="panel stack">
              <div className="finding-heading">
                <div>
                  <p className="eyebrow">AI-predicted finding</p>
                  <h2>{a.prediction.primary_class || "Unavailable"}</h2>
                </div>
                <UncertaintyBadge level={a.prediction.uncertainty_level} />
              </div>
              <div className="finding-numbers">
                <div>
                  <span>Class score</span>
                  <strong>{score(a.prediction.confidence)}</strong>
                </div>
                <div>
                  <span>Model uncertainty</span>
                  <strong>{numeric(a.prediction.uncertainty)}</strong>
                </div>
              </div>
              <p className="small muted">
                Class score is a model output, not diagnostic accuracy.
                Uncertainty describes the model output and is not clinical
                severity.
              </p>
              <div>
                <h3>All class scores</h3>
                <DiseaseBars
                  diseases={a.diseases}
                  primary={a.prediction.primary_class}
                />
              </div>
              <Link
                className="btn-primary"
                to={"/analysis/" + id + "/explainability"}
              >
                What the model highlighted
              </Link>
            </aside>
          </div>
        )}
        {tab === "technical" && (
          <div className="panel stack">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Backend-reported metadata</p>
                <h2>Technical details</h2>
              </div>
            </div>
            <div className="two-col">
              <dl className="detail-list">
                <div>
                  <dt>Analysis ID</dt>
                  <dd className="mono">{a.analysis_id}</dd>
                </div>
                <div>
                  <dt>Study ID</dt>
                  <dd className="mono">{a.study.id}</dd>
                </div>
                <div>
                  <dt>Image dimensions</dt>
                  <dd>
                    {a.study.width && a.study.height
                      ? a.study.width + " × " + a.study.height
                      : "Unavailable"}
                  </dd>
                </div>
                <div>
                  <dt>Model version</dt>
                  <dd>{a.model.version || "Unavailable"}</dd>
                </div>
                <div>
                  <dt>Device</dt>
                  <dd>{a.model.device || "Unavailable"}</dd>
                </div>
              </dl>
              <dl className="detail-list">
                <div>
                  <dt>Preprocessing</dt>
                  <dd>{numeric(a.timing.preprocessing_ms, 0)} ms</dd>
                </div>
                <div>
                  <dt>Inference</dt>
                  <dd>{numeric(a.timing.inference_ms, 0)} ms</dd>
                </div>
                <div>
                  <dt>Explainability</dt>
                  <dd>{numeric(a.timing.explainability_ms, 0)} ms</dd>
                </div>
                <div>
                  <dt>Total request</dt>
                  <dd>{numeric(a.timing.total_ms, 0)} ms</dd>
                </div>
                <div>
                  <dt>Margin uncertainty</dt>
                  <dd>{numeric(a.prediction.margin_uncertainty)}</dd>
                </div>
              </dl>
            </div>
            <details className="technical-details">
              <summary>Additional prediction metadata</summary>
              <div className="two-col">
                <div>
                  <h3>Reported uncertainties</h3>
                  {Object.keys(a.prediction.uncertainties || {}).length ? (
                    <dl className="detail-list">
                      {Object.entries(a.prediction.uncertainties).map(
                        ([k, v]) => (
                          <div key={k}>
                            <dt>{k}</dt>
                            <dd>{numeric(v)}</dd>
                          </div>
                        ),
                      )}
                    </dl>
                  ) : (
                    <p className="muted">Not supplied.</p>
                  )}
                </div>
                <div>
                  <h3>Reported disease weights</h3>
                  {Object.keys(a.prediction.disease_weights || {}).length ? (
                    <dl className="detail-list">
                      {Object.entries(a.prediction.disease_weights).map(
                        ([k, v]) => (
                          <div key={k}>
                            <dt>{k}</dt>
                            <dd>{numeric(v)}</dd>
                          </div>
                        ),
                      )}
                    </dl>
                  ) : (
                    <p className="muted">Not supplied.</p>
                  )}
                </div>
              </div>
            </details>
          </div>
        )}
      </section>
      <DoctorReview key={a.analysis_id} analysis={a} />
      <Disclaimer />
    </div>
  );
}
