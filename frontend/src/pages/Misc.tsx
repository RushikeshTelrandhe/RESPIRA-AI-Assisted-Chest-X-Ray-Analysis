import { DeveloperContact } from "../components/DeveloperContact";
import {
  ChevronDown,
  Cpu,
  CircleHelp,
  EyeOff,
  MonitorCog,
  RotateCcw,
  Rows3,
  ShieldCheck,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { api, type ModelStatus } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useResource, useWorkspace } from "../context/WorkspaceContext";
import { Alert, LoadingBlock, PageHeader } from "../components/UI";
import { errorText } from "../utils/display";
import { resetIntroSession } from "../services/introSession";
export function Models() {
  const { token } = useAuth();
  const result = useResource<ModelStatus>("model-status", () =>
    api.get("/api/v1/models/status", token),
  );
  if (result.loading) return <LoadingBlock label="Checking model status…" />;
  if (result.error || !result.data)
    return (
      <Alert retry={result.reload}>
        {result.error ||
          "Model status is unavailable. Models are not assumed ready."}
      </Alert>
    );
  const m = result.data;
  return (
    <div className="stack">
      <PageHeader
        eyebrow="Backend status"
        title="Model status"
        description="Readiness and device values below are reported by the running RESPIRA backend."
        action={
          <button className="btn-ghost" onClick={result.reload}>
            Refresh status
          </button>
        }
      />
      <section className="panel">
        <div className="status-line">
          <span className={"status-dot " + (m.model_loaded ? "" : "off")} />
          <div>
            <strong>
              {m.model_loaded
                ? "RESPIRA models loaded"
                : "RESPIRA models unavailable"}
            </strong>
            <span>{m.respira_ai || "Status unavailable"}</span>
          </div>
        </div>
        <dl className="detail-list">
          <div>
            <dt>Backend</dt>
            <dd>{m.backend || "Unavailable"}</dd>
          </div>
          <div>
            <dt>Device</dt>
            <dd>{m.device || "Unavailable"}</dd>
          </div>
          <div>
            <dt>GPU name</dt>
            <dd>{m.gpu_name || "Not reported"}</dd>
          </div>
        </dl>
        <p className="small muted">
          A loaded checkpoint does not by itself prove GPU execution. Use the
          backend-reported device above.
        </p>
      </section>
      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Selectable registry</p>
            <h2>Prediction models</h2>
          </div>
          <Cpu />
        </div>
        <div className="stack-sm">
          {m.models.map((model) => (
            <article className="model-option" key={model.key}>
              <span className={"status-dot " + (model.loaded ? "" : "off")} />
              <span>
                <strong>{model.name}</strong>
                <small>{model.description}</small>
                <em>
                  {model.loaded
                    ? "Loaded · " + model.version
                    : "Unavailable" +
                      (model.error
                        ? " · " + errorText(new Error(model.error))
                        : "")}
                </em>
              </span>
            </article>
          ))}
        </div>
      </section>
      <details className="panel technical-details">
        <summary>Verified architecture labels reported by the backend</summary>
        <dl className="detail-list">
          {Object.entries(m.architecture).map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
      </details>
      {m.error && <Alert kind="warning">{errorText(new Error(m.error))}</Alert>}
    </div>
  );
}
export function Settings() {
  const {
    reduced,
    setReduced,
    viewMode,
    setViewMode,
    presentationMode,
    setPresentationMode,
  } = useWorkspace();
  const navigate = useNavigate();
  return (
    <div className="stack">
      <PageHeader
        eyebrow="Interface preferences"
        title="Settings"
        description="These options change only how RESPIRA is displayed in this browser. They do not change model inference."
      />
      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Accessibility</p>
            <h2>Display preferences</h2>
          </div>
          <MonitorCog />
        </div>
        <div className="preference-grid" role="group" aria-label="Interface detail level">
          <button
            className={`preference-card${viewMode === "guided" ? " selected" : ""}`}
            type="button"
            onClick={() => setViewMode("guided")}
          >
            <MonitorCog size={20} />
            <span><strong>Guided view</strong><small>Helpful explanations and comfortable spacing for a first-time user.</small></span>
          </button>
          <button
            className={`preference-card${viewMode === "clinical" ? " selected" : ""}`}
            type="button"
            onClick={() => setViewMode("clinical")}
          >
            <Rows3 size={20} />
            <span><strong>Clinical view</strong><small>Denser tables and reduced guidance for faster repeat review.</small></span>
          </button>
        </div>
        <label className="settings-row">
          <span>
            <strong>Reduce motion</strong>
            <small>Use the static X-ray intro and remove non-essential movement.</small>
          </span>
          <input
            type="checkbox"
            checked={reduced}
            onChange={(e) => setReduced(e.target.checked)}
          />
        </label>
        <label className="settings-row">
          <span>
            <strong>Presentation privacy</strong>
            <small>
              Mask patient names and identifiers during classroom or public demonstrations.
            </small>
          </span>
          <input
            type="checkbox"
            checked={presentationMode}
            onChange={(e) => setPresentationMode(e.target.checked)}
          />
        </label>
      </section>
      <DeveloperContact />
      <section className="research-note">
        <ShieldCheck size={18} />
        <span>
          No patient data is stored by these preferences. Medical workflow
          settings remain controlled by the backend.
        </span>
      </section>
    </div>
  );
}
const faqs = [
  {
    q: "How do I start an analysis?",
    a: "Open New analysis, choose an existing patient, upload a PNG or JPG, choose a model marked Available, then select Analyze with RESPIRA.",
  },
  {
    q: "Why is Analyze disabled?",
    a: "RESPIRA needs a patient, a backend-validated image and a model that the backend reports as loaded. Check each item in the study confirmation panel.",
  },
  {
    q: "What does model uncertainty mean?",
    a: "It describes uncertainty in the model output. It is not clinical severity and does not replace review of the image and patient context.",
  },
  {
    q: "Why can I not see a historical original image?",
    a: "The supplied frontend API does not include a separate original-image download endpoint. Generate an available explanation, or analyze a new upload in the same session.",
  },
  {
    q: "What if a model is unavailable?",
    a: "Keep the backend terminal running, open Model status and use Refresh status. The page will show only what the backend reports and will not assume readiness.",
  },
  {
    q: "What do the explanation maps prove?",
    a: "They show model attribution or attention patterns. They do not confirm disease and are not verified lesion outlines.",
  },
];
export function Help() {
  return (
    <div className="stack">
      <PageHeader
        eyebrow="Guidance"
        title="Help centre"
        description="Plain-language answers for the RESPIRA review workflow."
      />
      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Common questions</p>
            <h2>Using RESPIRA</h2>
          </div>
          <CircleHelp />
        </div>
        <div className="faq">
          {faqs.map((item) => (
            <details key={item.q}>
              <summary>
                {item.q}
                <ChevronDown size={18} />
              </summary>
              <p>{item.a}</p>
            </details>
          ))}
        </div>
      </section>
      <Alert kind="info">
        Research prototype. If a request fails, keep your backend terminal open,
        confirm model status, and retry only the failed action.
      </Alert>
      <DeveloperContact />
    </div>
  );
}
