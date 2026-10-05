import { decodeNotes, savePatientNotes, agreementLabels } from "../services/clinicalReviews";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Archive, ArrowLeft, ScanLine } from "lucide-react";
import { api, type HistoryItem, type Patient } from "../services/api";
import { useAuth } from "../context/AuthContext";
import {
  useResource,
  useUnsavedChanges,
  useWorkspace,
} from "../context/WorkspaceContext";
import {
  Alert,
  Empty,
  LoadingBlock,
  Modal,
  PageHeader,
  Tabs,
} from "../components/UI";
import { UncertaintyBadge } from "../components/widgets";
import { dateLabel, errorText, score } from "../utils/display";
type Tab = "overview" | "xray-tests" | "ai-results" | "reports" | "notes";
export function PatientDetail() {
  const { id = "" } = useParams();
  const { token } = useAuth();
  const { cache } = useWorkspace();
  const nav = useNavigate();
  const patient = useResource<Patient>(id ? "patient:" + id : null, () =>
    api.get("/api/v1/patients/" + id, token),
  );
  const history = useResource<HistoryItem[]>(
    id ? "patient-history:" + id : null,
    () => api.get("/api/v1/patients/" + id + "/history", token),
  );
  const [tab, setTab] = useState<Tab>("overview");
  const [notes, setNotes] = useState("");
  const [savedNotes, setSavedNotes] = useState("");
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);
  const [archive, setArchive] = useState(false);
  const [archiving, setArchiving] = useState(false);
  const initialized = useRef("");
  useEffect(() => {
    if (patient.data && initialized.current !== patient.data.id) {
      initialized.current = patient.data.id;
      try {
        const text = decodeNotes(patient.data.notes || "").text;
        setNotes(text); setSavedNotes(text);
      } catch (e) { setMessage(errorText(e)); }
    }
  }, [patient.data]);
  useUnsavedChanges(
    notes !== savedNotes,
    "Your patient notes have not been saved. Leave and discard them?",
  );
  async function save() {
    setSaving(true);
    setMessage("");
    try {
      const p = await savePatientNotes(id, notes, savedNotes, token);
      setSavedNotes(decodeNotes(p.notes || "").text);
      cache.invalidate("patient:" + id);
      setMessage("Notes saved.");
    } catch (e) {
      setMessage(errorText(e));
    } finally {
      setSaving(false);
    }
  }
  async function remove() {
    setArchiving(true);
    setMessage("");
    try {
      await api.del("/api/v1/patients/" + id, token);
      cache.invalidate("patients:");
      cache.invalidate("dashboard");
      nav("/patients", { replace: true });
    } catch (e) {
      setMessage(errorText(e));
      setArchiving(false);
      setArchive(false);
    }
  }
  if (patient.loading) return <LoadingBlock label="Loading patient record…" />;
  if (patient.error || !patient.data)
    return (
      <Alert retry={patient.reload}>
        {patient.error || "Patient record is unavailable."}
      </Alert>
    );
  const pData = patient.data;
  let noteContent: ReturnType<typeof decodeNotes> = { text: "Notes could not be read.", reviews: [] };
  try { noteContent = decodeNotes(pData.notes || ""); } catch { /* save is also guarded */ }

  const tabs = [
    { value: "overview" as const, label: "Overview" },
    { value: "xray-tests" as const, label: "X-ray tests" },
    { value: "ai-results" as const, label: "AI results" },
    { value: "reports" as const, label: "Reports" },
    { value: "notes" as const, label: "Notes" },
  ];
  const records = history.data ?? [];
  return (
    <div className="stack">
      <Link className="text-button" to="/patients">
        <ArrowLeft size={16} />
        All patients
      </Link>
      <PageHeader
        eyebrow={"Patient " + (pData.patient_code || pData.id.slice(0, 8))}
        title={pData.full_name}
        description={
          (pData.age || "Age unavailable") +
          " · " +
          (pData.gender || "Gender unavailable") +
          " · " +
          pData.test_count +
          " previous " +
          (pData.test_count === 1 ? "analysis" : "analyses")
        }
        action={
          <div className="page-actions">
            <Link className="btn-primary" to={"/xray-test?patient=" + pData.id}>
              <ScanLine size={18} />
              New analysis
            </Link>
            <button className="btn-ghost" onClick={() => setArchive(true)}>
              <Archive size={17} />
              Archive
            </button>
          </div>
        }
      />
      <Tabs id="patient-tabs" options={tabs} value={tab} onChange={setTab} />
      <section
        id="patient-tabs-panel"
        role="tabpanel"
        aria-labelledby={"patient-tabs-" + tab}
        className="panel"
      >
        {tab === "overview" && (
          <div className="two-col">
            <div>
              <p className="eyebrow">Patient details</p>
              <dl className="detail-list">
                <div>
                  <dt>Patient code</dt>
                  <dd>{pData.patient_code || pData.id.slice(0, 8)}</dd>
                </div>
                <div>
                  <dt>Phone</dt>
                  <dd>{pData.phone || "Not provided"}</dd>
                </div>
                <div>
                  <dt>Email</dt>
                  <dd>{pData.email || "Not provided"}</dd>
                </div>
                <div>
                  <dt>Address</dt>
                  <dd>{pData.address || "Not provided"}</dd>
                </div>
              </dl>
            </div>
            <div>
              <p className="eyebrow">Context</p>
              <h3>Medical history</h3>
              <p>
                {pData.medical_history || "No medical history has been added."}
              </p>
              <h3>Latest notes</h3>
              <p>{noteContent.text || "No general patient notes have been added."}</p>
            </div>
          </div>
        )}
        {(tab === "xray-tests" || tab === "ai-results") &&
          (history.loading ? (
            <LoadingBlock label="Loading analyses…" />
          ) : history.error ? (
            <Alert retry={history.reload}>{history.error}</Alert>
          ) : records.length === 0 ? (
            <Empty
              title="No analyses for this patient"
              hint="Start a new analysis to add the first study."
            />
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Primary output</th>
                    <th>Score</th>
                    <th>Uncertainty</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {records.map((h) => (
                    <tr key={h.analysis_id}>
                      <td>{dateLabel(h.created_at, true)}</td>
                      <td>{h.primary_class || "Unavailable"}</td>
                      <td>{score(h.confidence)}</td>
                      <td>
                        <UncertaintyBadge level={h.uncertainty_level} />
                      </td>
                      <td>
                        <Link
                          className="text-button"
                          to={"/analysis/" + h.analysis_id}
                        >
                          Review
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        {tab === "reports" &&
          (history.loading ? (
            <LoadingBlock label="Loading report records…" />
          ) : records.length === 0 ? (
            <Empty title="No report records yet" />
          ) : (
            <div className="stack-sm">
              {records.map((h) => (
                <div className="report-row" key={h.analysis_id}>
                  <div>
                    <strong>{dateLabel(h.created_at)}</strong>
                    <span>{h.primary_class || "Output unavailable"}</span>
                  </div>
                  <Link className="btn-ghost" to={"/analysis/" + h.analysis_id}>
                    Open study
                  </Link>
                </div>
              ))}
            </div>
          ))}
        {tab === "notes" && (
          <div className="stack">
            {noteContent.reviews.length > 0 && <div className="patient-saved-reviews"><h3>Saved study reviews</h3>{[...noteContent.reviews].reverse().map(review => <Link className="saved-review-row" key={review.analysisId + review.doctorId} to={`/analysis/${review.analysisId}#doctor-feedback`}><span><strong>{review.doctorName}</strong><small>Study {review.analysisId.slice(0,8)} · {dateLabel(review.updatedAt,true)}</small></span><span>{agreementLabels[review.agreement]}</span></Link>)}</div>}

            <div>
              <label className="label" htmlFor="patient-clinical-notes">
                General patient notes
              </label>
              <textarea
                id="patient-clinical-notes"
                className="input"
                rows={8}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
              />
            </div>
            {message && (
              <Alert kind={message === "Notes saved." ? "success" : "error"}>
                {message}
              </Alert>
            )}
            <div>
              <button
                className="btn-primary"
                disabled={saving || notes === savedNotes}
                onClick={() => void save()}
              >
                {saving ? "Saving…" : "Save notes"}
              </button>
            </div>
          </div>
        )}
      </section>
      <Modal
        open={archive}
        title="Archive this patient?"
        onDismiss={() => !archiving && setArchive(false)}
      >
        <div className="modal-body">
          <p>
            The patient will be removed from the active directory. Existing
            analysis history is preserved by the backend.
          </p>
          <div className="form-actions">
            <button
              className="btn-ghost"
              disabled={archiving}
              onClick={() => setArchive(false)}
            >
              Cancel
            </button>
            <button
              className="btn-danger"
              disabled={archiving}
              onClick={() => void remove()}
            >
              {archiving ? "Archiving…" : "Archive patient"}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
