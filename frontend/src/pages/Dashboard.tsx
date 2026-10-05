import { Link } from "react-router-dom";
import {
  ArrowRight,
  CalendarDays,
  CheckCircle2,
  CircleGauge,
  Cpu,
  Layers3,
  Plus,
  ScanLine,
  ShieldCheck,
  Users,
} from "lucide-react";
import { api, type HistoryItem, type ModelStatus } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useResource } from "../context/WorkspaceContext";
import { Alert, Empty, LoadingBlock, PageHeader } from "../components/UI";
import { Disclaimer, UncertaintyBadge } from "../components/widgets";
import { dateLabel, score } from "../utils/display";

type Dash = {
  doctor_name: string;
  stats: {
    total_patients: number;
    total_tests: number;
    tests_this_week: number;
    high_uncertainty: number;
  };
  recent_tests: HistoryItem[];
  model: { loaded: boolean; device: string };
};

export function Dashboard() {
  const { doctor, token } = useAuth();
  const dashboard = useResource<Dash>("dashboard", () =>
    api.get("/api/v1/dashboard", token),
  );
  const status = useResource<ModelStatus>("model-status", () =>
    api.get("/api/v1/models/status", token),
  );
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";

  if (dashboard.loading) return <LoadingBlock label="Preparing your clinical workspace…" />;
  if (dashboard.error || !dashboard.data) {
    return (
      <Alert retry={dashboard.reload}>
        {dashboard.error || "Dashboard data is unavailable."}
      </Alert>
    );
  }

  const data = dashboard.data;
  const cards = [
    { icon: Users, label: "Patients", value: data.stats.total_patients, note: "Registered records" },
    { icon: ScanLine, label: "Analyses", value: data.stats.total_tests, note: "Completed studies" },
    { icon: CalendarDays, label: "This week", value: data.stats.tests_this_week, note: "Recent workload" },
    { icon: CircleGauge, label: "High uncertainty", value: data.stats.high_uncertainty, note: "Needs closer review" },
  ];
  const models = status.data?.models ?? [];
  const readyModels = models.filter((model) => model.loaded).length;
  const ready = readyModels > 0;

  return (
    <div className="stack dashboard-page">
      <PageHeader
        eyebrow="Clinical command centre"
        title={`${greeting}, ${doctor?.full_name || data.doctor_name}`}
        description="Your patients, study activity and AI readiness in one focused view."
        action={
          <Link className="btn-primary" to="/xray-test">
            <Plus size={18} /> New analysis
          </Link>
        }
      />

      <section className="dashboard-command">
        <div className="dashboard-command-copy">
          <span className={`command-status${ready ? " ready" : ""}`}>
            <i /> {status.loading ? "Checking inference system" : ready ? `${readyModels} prediction models ready` : "Prediction models unavailable"}
          </span>
          <p className="eyebrow">Start a guided review</p>
          <h2>Move from radiograph to explainable model evidence.</h2>
          <p>
            Select the patient, validate the image and choose a backend-confirmed
            model before sending a study to RESPIRA.
          </p>
          <div className="dashboard-command-actions">
            <Link className="btn-light" to="/xray-test">
              Begin new study <ArrowRight size={17} />
            </Link>
            <Link className="dashboard-secondary-action" to="/history">
              Open study archive
            </Link>
          </div>
          <div className="command-safety">
            <ShieldCheck size={16} /> Outputs support review and do not establish a diagnosis.
          </div>
        </div>
        <div className="dashboard-command-visual" aria-hidden="true">
          <img src="/media/respira-intro-poster.webp" alt="" />
          <div className="command-scan" />
          <span className="command-corner top-left" />
          <span className="command-corner top-right" />
          <span className="command-corner bottom-left" />
          <span className="command-corner bottom-right" />
          <div className="command-visual-meta">
            <span>Study acquisition</span>
            <strong>Explainable review</strong>
          </div>
        </div>
      </section>

      <section className="dashboard-stats" aria-label="Workspace summary">
        {cards.map((card) => (
          <article className="stat-card" key={card.label}>
            <span className="stat-icon"><card.icon size={19} /></span>
            <div className="stat-copy">
              <span className="stat-label">{card.label}</span>
              <strong className="stat-value">{card.value}</strong>
              <small>{card.note}</small>
            </div>
          </article>
        ))}
      </section>

      <div className="workspace-grid dashboard-content-grid">
        <section className="panel activity-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Latest activity</p>
              <h2>Recent analyses</h2>
            </div>
            <Link className="text-button" to="/history">
              View archive <ArrowRight size={16} />
            </Link>
          </div>
          {data.recent_tests.length === 0 ? (
            <Empty
              title="No analyses yet"
              hint="Completed studies will appear here with their output and uncertainty."
              action={<Link className="btn-ghost" to="/xray-test">Start the first analysis</Link>}
            />
          ) : (
            <div className="table-wrap">
              <table className="data-table clinical-table">
                <thead>
                  <tr><th>Patient</th><th>AI-predicted finding</th><th>Score</th><th>Uncertainty</th><th>Reviewed</th></tr>
                </thead>
                <tbody>
                  {data.recent_tests.map((test) => (
                    <tr key={test.analysis_id}>
                      <td>
                        <Link className="row-name patient-sensitive" to={`/analysis/${test.analysis_id}`}>
                          {test.patient_name}
                        </Link>
                        <small className="mono">{test.analysis_id.slice(0, 8)}</small>
                      </td>
                      <td>{test.primary_class || "Unavailable"}</td>
                      <td className="mono">{score(test.confidence)}</td>
                      <td><UncertaintyBadge level={test.uncertainty_level} /></td>
                      <td>{dateLabel(test.created_at, true)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <aside className="dashboard-rail stack-sm">
          <section className="panel system-readiness-panel">
            <div className="section-heading">
              <div><p className="eyebrow">Inference system</p><h2>Model readiness</h2></div>
              <span className="rail-icon"><Cpu size={20} /></span>
            </div>
            {status.loading ? (
              <LoadingBlock label="Checking models…" />
            ) : status.error ? (
              <Alert kind="warning" retry={status.reload}>Status could not be checked.</Alert>
            ) : (
              <>
                <div className={`readiness-score${ready ? " ready" : ""}`}>
                  <strong>{readyModels}</strong>
                  <span>of {models.length} models ready</span>
                </div>
                <div className="status-line compact-status">
                  <span className={`status-dot${ready ? "" : " off"}`} />
                  <div><strong>{ready ? "Analysis available" : "Analysis unavailable"}</strong><span>{status.data?.respira_ai || "Not reported"}</span></div>
                </div>
                <dl className="detail-list system-details">
                  <div><dt>Device</dt><dd>{status.data?.device || "Unavailable"}</dd></div>
                  <div><dt>GPU</dt><dd>{status.data?.gpu_name || "Not reported"}</dd></div>
                </dl>
                <Link className="text-button" to="/models">Inspect model registry <ArrowRight size={15} /></Link>
              </>
            )}
          </section>

          <section className="panel review-path-panel">
            <p className="eyebrow">Review path</p>
            <h2>Three checkpoints</h2>
            <ol>
              <li><span><CheckCircle2 size={15} /></span><div><strong>Validate</strong><small>Patient, image and model</small></div></li>
              <li><span><CircleGauge size={15} /></span><div><strong>Interpret</strong><small>Prediction and uncertainty</small></div></li>
              <li><span><Layers3 size={15} /></span><div><strong>Explain</strong><small>Grad-CAM or ViT attention</small></div></li>
            </ol>
          </section>
        </aside>
      </div>
      <Disclaimer />
    </div>
  );
}
