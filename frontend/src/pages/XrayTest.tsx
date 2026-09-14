import { SearchInput } from "../components/SearchInput";
import { flushSync } from "react-dom";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import {
  Check,
  FileImage,
  Search,
  UploadCloud,
  X,
  RefreshCw,
  BrainCircuit,
  CheckCircle2,
  ClipboardCheck,
  UserRound,
} from "lucide-react";
import {
  api,
  type AnalyzeResponse,
  type ModelStatus,
  type Patient,
} from "../services/api";
import { useAuth } from "../context/AuthContext";
import {
  usePreview,
  useResource,
  useUnsavedChanges,
  useWorkspace,
} from "../context/WorkspaceContext";
import { Alert, LoadingBlock, PageHeader } from "../components/UI";
import { Disclaimer } from "../components/widgets";
import { errorText } from "../utils/display";
export function XrayTest() {
  const { token } = useAuth();
  const nav = useNavigate();
  const [params] = useSearchParams();
  const { draft, setDraft, images, cache, setGuard } = useWorkspace();
  const [patientSearch, setPatientSearch] = useState("");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const fileRequest = useRef(0);
  const pending = useRef(false);
  const patients = useResource<Patient[]>("analysis-patients", () =>
    api.get("/api/v1/patients", token),
  );
  const status = useResource<ModelStatus>("model-status", () =>
    api.get("/api/v1/models/status", token),
  );
  const preview = usePreview(draft.file);
  useEffect(() => {
    const fromUrl = params.get("patient");
    if (fromUrl && !draft.patientId)
      setDraft((d) => ({ ...d, patientId: fromUrl }));
  }, [params, draft.patientId, setDraft]);
  useEffect(() => {
    const models = status.data?.models;
    if (!models?.length) return;
    if (!models.some((m) => m.key === draft.model && m.loaded)) {
      const first = models.find((m) => m.loaded);
      if (first) setDraft((d) => ({ ...d, model: first.key }));
    }
  }, [status.data, draft.model, setDraft]);
  useUnsavedChanges(
    Boolean(draft.file) || busy,
    busy
      ? "An analysis request is running. Leaving may not stop work already sent to the backend."
      : "Your selected X-ray has not been analyzed. Leave and discard it?",
  );
  const filtered = (patients.data ?? []).filter((p) =>
    (p.full_name + " " + p.patient_code + " " + p.phone)
      .toLowerCase()
      .includes(patientSearch.toLowerCase()),
  );
  const selectedPatient = patients.data?.find((p) => p.id === draft.patientId);
  const models = status.data?.models ?? [];
  const selectedModel = models.find((m) => m.key === draft.model);
  const workflowIndex = busy
    ? 4
    : draft.meta && selectedModel?.loaded
      ? 3
      : draft.file
        ? 2
        : draft.patientId
          ? 1
          : 0;
  const canAnalyze = Boolean(
    draft.patientId &&
      draft.file &&
      draft.meta &&
      selectedModel?.loaded &&
      !uploading &&
      !busy,
  );
  function removeFile() {
    fileRequest.current++;
    setDraft((d) => ({ ...d, file: null, meta: null }));
    if (input.current) input.current.value = "";
  }
  async function choose(file?: File) {
    setError("");
    if (!file) return;
    const allowed =
      /\.(png|jpe?g)$/i.test(file.name) &&
      (file.type === "" ||
        file.type === "image/png" ||
        file.type === "image/jpeg");
    if (!allowed) {
      setError("Choose a PNG, JPG or JPEG chest X-ray.");
      return;
    }
    const request = ++fileRequest.current;
    setDraft((d) => ({ ...d, file, meta: null }));
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const meta = await api.postForm<{
        width: number;
        height: number;
        file_size: number;
        filename: string;
        preview?: string;
      }>("/api/v1/xray/upload", form, token);
      if (request === fileRequest.current)
        setDraft((d) => (d.file === file ? { ...d, meta } : d));
    } catch (e) {
      if (request === fileRequest.current) {
        setError(errorText(e));
        setDraft((d) => ({ ...d, file: null, meta: null }));
        if (input.current) input.current.value = "";
      }
    } finally {
      if (request === fileRequest.current) setUploading(false);
    }
  }
  async function analyze(e: FormEvent) {
    e.preventDefault();
    if (pending.current || !canAnalyze || !draft.file) return;
    pending.current = true;
    setBusy(true);
    setError("");
    const submittedFile = draft.file;
    try {
      const form = new FormData();
      form.append("patient_id", draft.patientId);
      form.append("file", submittedFile);
      form.append("model", draft.model);
      const result = await api.postForm<AnalyzeResponse>(
        "/api/v1/analyze",
        form,
        token,
      );
      cache.invalidate("dashboard");
      cache.invalidate("history:");
      cache.invalidate("reports");
      cache.invalidate("patient-history:" + draft.patientId);
      flushSync(() => {
        images.set(result.analysis_id, submittedFile);
        setDraft({ patientId: "", model: "fusion", file: null, meta: null });
        setGuard({ active: false, message: "" });
      });
      nav("/analysis/" + result.analysis_id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }
  return (
    <form className="stack" onSubmit={analyze}>
      <PageHeader
        eyebrow="Guided workflow"
        title="New chest X-ray analysis"
        description="Confirm the patient, image and available model before sending the study to RESPIRA."
      />
      <ol className="workflow-progress" aria-label="Analysis preparation progress">
        {[
          { label: "Patient", icon: UserRound },
          { label: "X-ray", icon: FileImage },
          { label: "AI model", icon: BrainCircuit },
          { label: "Review", icon: ClipboardCheck },
        ].map((item, index) => {
          const complete = workflowIndex > index;
          const active = workflowIndex === index;
          return (
            <li
              key={item.label}
              className={complete ? "complete" : active ? "active" : ""}
              aria-current={active ? "step" : undefined}
            >
              <span>{complete ? <Check size={16} /> : <item.icon size={16} />}</span>
              <div><small>Step {index + 1}</small><strong>{item.label}</strong></div>
            </li>
          );
        })}
      </ol>
      {error && (
        <Alert retry={status.error ? status.reload : undefined}>{error}</Alert>
      )}
      <div className="upload-layout">
        <div className="stack">
          <section className="panel clinical-step-card">
            <div className="step-heading">
              <span className="step-circle">1</span>
              <div>
                <h2>Select patient</h2>
                <p>Search by name or patient code.</p>
              </div>
              <SearchInput value={patientSearch} onChange={setPatientSearch} label="Filter patients" placeholder="Name or patient code…" />
            </div>
            {patients.loading ? (
              <LoadingBlock label="Loading patients…" />
            ) : patients.error ? (
              <Alert retry={patients.reload}>{patients.error}</Alert>
            ) : (
              <>
                <select
                  className="input"
                  aria-label="Select patient"
                  required
                  value={draft.patientId}
                  onChange={(e) =>
                    setDraft((d) => ({ ...d, patientId: e.target.value }))
                  }
                >
                  <option value="">Choose a patient…</option>
                  {filtered.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.full_name} ({p.patient_code || p.id.slice(0, 8)})
                    </option>
                  ))}
                </select>
                <p className="field-hint">
                  Cannot find the record?{" "}
                  <Link to="/patients">Add a patient</Link> first.
                </p>
              </>
            )}
          </section>
          <section className="panel clinical-step-card upload-step-card">
            <div className="step-heading">
              <span className="step-circle">2</span>
              <div>
                <h2>Upload chest X-ray</h2>
                <p>
                  PNG, JPG or JPEG. Final validation is performed by the
                  backend.
                </p>
              </div>
            </div>
            <label
              className={"drop-zone " + (dragging ? "dragging" : "")}
              onDragEnter={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragOver={(e) => e.preventDefault()}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                void choose(e.dataTransfer.files?.[0]);
              }}
            >
              <input
                ref={input}
                type="file"
                accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                onChange={(e) => void choose(e.target.files?.[0])}
              />
              <span className="drop-content">
                <span className="upload-icon">
                  <UploadCloud size={28} />
                </span>
                <strong>
                  {dragging
                    ? "Drop the image here"
                    : "Drag and drop, or choose a file"}
                </strong>
                <span>
                  Keyboard users can press Enter when this area is focused.
                </span>
              </span>
            </label>
            {draft.file && (
              <div className="file-preview">
                {preview ? (
                  <img src={preview} alt="Selected chest X-ray preview" />
                ) : (
                  <span className="upload-icon">
                    <FileImage />
                  </span>
                )}
                <div className="file-meta">
                  <strong>{draft.file.name}</strong>
                  <span>
                    {uploading
                      ? "Checking image…"
                      : draft.meta
                        ? draft.meta.width +
                          " × " +
                          draft.meta.height +
                          " pixels"
                        : "Dimensions unavailable"}
                  </span>
                  <span>
                    {(
                      (draft.meta?.file_size ?? draft.file.size) / 1024
                    ).toFixed(0)}{" "}
                    KB
                  </span>
                  <div className="page-actions">
                    <button
                      type="button"
                      className="btn-ghost"
                      onClick={() => input.current?.click()}
                      disabled={uploading}
                    >
                      <RefreshCw size={16} />
                      Replace
                    </button>
                    <button
                      type="button"
                      className="btn-ghost"
                      onClick={removeFile}
                      disabled={uploading}
                    >
                      <X size={16} />
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            )}
          </section>
          <section className="panel clinical-step-card">
            <div className="step-heading">
              <span className="step-circle">3</span>
              <div>
                <h2>Choose an available model</h2>
                <p>
                  Availability and device details come directly from the
                  backend.
                </p>
              </div>
            </div>
            {status.loading ? (
              <LoadingBlock label="Checking model availability…" />
            ) : status.error ? (
              <Alert kind="warning" retry={status.reload}>
                Model status could not be checked. Analysis remains disabled
                until the backend reports an available model.
              </Alert>
            ) : (
              <>
                <div className="model-options">
                  {models.map((m) => (
                    <label
                      key={m.key}
                      className={
                        "model-option " +
                        (draft.model === m.key ? "selected " : "") +
                        (!m.loaded ? "unavailable" : "")
                      }
                    >
                      <input
                        type="radio"
                        name="model"
                        value={m.key}
                        checked={draft.model === m.key}
                        disabled={!m.loaded}
                        onChange={() =>
                          setDraft((d) => ({ ...d, model: m.key }))
                        }
                      />
                      <span>
                        <strong>{m.name}</strong>
                        <small>{m.description}</small>
                        <em>{m.loaded ? "Available" : "Unavailable"}</em>
                        {!m.loaded && m.error && (
                          <details className="model-error-details">
                            <summary>Why this model is unavailable</summary>
                            <span>{m.error}</span>
                          </details>
                        )}
                      </span>
                    </label>
                  ))}
                </div>
                <p className="field-hint">
                  Backend device:{" "}
                  <strong>{status.data?.device || "not reported"}</strong>
                  {status.data?.gpu_name ? " · " + status.data.gpu_name : ""}.
                  This display reports backend state; it does not switch CUDA
                  on.
                </p>
              </>
            )}
          </section>
        </div>
        <aside className="analysis-summary card">
          <p className="eyebrow">Before you analyze</p>
          <h2>Study confirmation</h2>
          <div className="summary-steps">
            <div
              className={
                selectedPatient ? "summary-step ready" : "summary-step"
              }
            >
              <span>{selectedPatient ? <Check /> : "1"}</span>
              <div>
                <strong>Patient</strong>
                <small>
                  {selectedPatient
                    ? selectedPatient.full_name +
                      " · " +
                      (selectedPatient.patient_code ||
                        selectedPatient.id.slice(0, 8))
                    : "Not selected"}
                </small>
              </div>
            </div>
            <div className={draft.meta ? "summary-step ready" : "summary-step"}>
              <span>{draft.meta ? <Check /> : "2"}</span>
              <div>
                <strong>X-ray</strong>
                <small>
                  {draft.meta ? draft.meta.filename : "Not uploaded"}
                </small>
              </div>
            </div>
            <div
              className={
                selectedModel?.loaded ? "summary-step ready" : "summary-step"
              }
            >
              <span>{selectedModel?.loaded ? <Check /> : "3"}</span>
              <div>
                <strong>Model</strong>
                <small>
                  {selectedModel?.loaded
                    ? selectedModel.name
                    : "No available model selected"}
                </small>
              </div>
            </div>
          </div>
          <button
            className="btn-primary full-width"
            type="submit"
            disabled={!canAnalyze}
          >
            {busy ? (
              <>
                <span className="spin" />
                Analyzing X-ray…
              </>
            ) : (
              "Analyze with RESPIRA"
            )}
          </button>
          {busy && (
            <div className="analysis-processing" role="status">
              <div className="processing-orbit"><BrainCircuit size={24} /></div>
              <strong>RESPIRA is reviewing the study</strong>
              <p>
                The backend is processing the image. These are workflow labels,
                not live percentages because the API does not report stages.
              </p>
              <div className="processing-stages" aria-hidden="true">
                <span><CheckCircle2 size={14} /> Request submitted</span>
                <span className="active"><span className="spin" /> Inference in progress</span>
                <span>Preparing results</span>
              </div>
            </div>
          )}
          <Disclaimer />
        </aside>
      </div>
    </form>
  );
}
