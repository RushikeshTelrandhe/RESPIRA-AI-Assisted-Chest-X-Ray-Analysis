import { SearchInput } from "../components/SearchInput";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { Plus, Search, ArrowRight } from "lucide-react";
import { api, type Patient } from "../services/api";
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
  SubmitButton,
} from "../components/UI";
import { errorText } from "../utils/display";
const blank = {
  full_name: "",
  patient_code: "",
  age: "",
  gender: "",
  phone: "",
  medical_history: "",
  notes: "",
};
export function Patients() {
  const { token } = useAuth();
  const { cache } = useWorkspace();
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState(blank);
  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const pending = useRef(false);
  useEffect(() => {
    const t = window.setTimeout(() => setQuery(q.trim()), 250);
    return () => window.clearTimeout(t);
  }, [q]);
  const list = useResource<Patient[]>("patients:" + query, () =>
    api.get("/api/v1/patients?q=" + encodeURIComponent(query), token),
  );
  const dirty = Object.values(form).some(Boolean);
  useUnsavedChanges(
    open && dirty,
    "The new patient form has unsaved information. Leave and discard it?",
  );
  const set =
    (key: keyof typeof form) =>
    (
      e: React.ChangeEvent<
        HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
      >,
    ) =>
      setForm({ ...form, [key]: e.target.value });
  async function create(e: FormEvent) {
    e.preventDefault();
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    setSubmitError("");
    try {
      await api.post(
        "/api/v1/patients",
        { ...form, age: form.age ? Number(form.age) : 0 },
        token,
      );
      cache.invalidate("patients:");
      cache.invalidate("dashboard");
      setForm(blank);
      setOpen(false);
      list.reload();
    } catch (e) {
      setSubmitError(errorText(e));
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }
  return (
    <div className="stack">
      <PageHeader
        eyebrow="Patient directory"
        title="Patients"
        description="Search existing records or add a patient before starting an analysis."
        action={
          <button className="btn-primary" onClick={() => setOpen(true)}>
            <Plus size={18} />
            Add patient
          </button>
        }
      />
      <div className="clinical-section-row">
        <div><h2>Patient directory</h2><p className="small muted">{list.data ? `${list.data.length} records` : 'Checking records…'}</p></div>
        <SearchInput value={q} onChange={setQ} label="Search patients" placeholder="Name, patient code or phone…" />
      </div>
      {list.loading ? (
        <LoadingBlock label="Loading patients…" />
      ) : list.error ? (
        <Alert retry={list.reload}>{list.error}</Alert>
      ) : !list.data?.length ? (
        <Empty
          title={query ? "No matching patients" : "No patients yet"}
          hint={
            query
              ? "Try a different name, code or phone."
              : "Add the first patient to begin."
          }
          action={
            !query ? (
              <button className="btn-primary" onClick={() => setOpen(true)}>
                Add patient
              </button>
            ) : undefined
          }
        />
      ) : (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Patient</th>
                <th>Patient code</th>
                <th>Age / gender</th>
                <th>Contact</th>
                <th>Analyses</th>
                <th>
                  <span className="sr-only">Open</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {list.data.map((p) => (
                <tr key={p.id}>
                  <td>
                    <div className="table-person">
                      <span className="avatar">{p.full_name.charAt(0)}</span>
                      <Link className="row-name" to={"/patients/" + p.id}>
                        {p.full_name}
                      </Link>
                    </div>
                  </td>
                  <td className="mono">{p.patient_code || p.id.slice(0, 8)}</td>
                  <td>
                    {p.age || "—"} / {p.gender || "—"}
                  </td>
                  <td>{p.phone || "Not provided"}</td>
                  <td>{p.test_count}</td>
                  <td>
                    <Link
                      className="icon-button"
                      aria-label={"Open " + p.full_name}
                      to={"/patients/" + p.id}
                    >
                      <ArrowRight size={17} />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Modal
        open={open}
        title="Add a patient"
        onDismiss={() => !busy && setOpen(false)}
      >
        <form className="modal-body stack" onSubmit={create}>
          {submitError && <Alert>{submitError}</Alert>}
          <div className="form-grid">
            <div>
              <label className="label" htmlFor="patient-name">
                Full name *
              </label>
              <input
                id="patient-name"
                className="input"
                required
                value={form.full_name}
                onChange={set("full_name")}
              />
            </div>
            <div>
              <label className="label" htmlFor="patient-code">
                Patient code
              </label>
              <input
                id="patient-code"
                className="input"
                value={form.patient_code}
                onChange={set("patient_code")}
              />
            </div>
            <div>
              <label className="label" htmlFor="patient-age">
                Age
              </label>
              <input
                id="patient-age"
                className="input"
                type="number"
                min="0"
                max="150"
                value={form.age}
                onChange={set("age")}
              />
            </div>
            <div>
              <label className="label" htmlFor="patient-gender">
                Gender
              </label>
              <select
                id="patient-gender"
                className="input"
                value={form.gender}
                onChange={set("gender")}
              >
                <option value="">Not provided</option>
                <option>Female</option>
                <option>Male</option>
                <option>Other</option>
              </select>
            </div>
            <div>
              <label className="label" htmlFor="patient-phone">
                Phone
              </label>
              <input
                id="patient-phone"
                className="input"
                type="tel"
                value={form.phone}
                onChange={set("phone")}
              />
            </div>
            <div className="full-width">
              <label className="label" htmlFor="patient-history">
                Medical history
              </label>
              <textarea
                id="patient-history"
                className="input"
                rows={3}
                value={form.medical_history}
                onChange={set("medical_history")}
              />
            </div>
            <div className="full-width">
              <label className="label" htmlFor="patient-notes">
                Notes
              </label>
              <textarea
                id="patient-notes"
                className="input"
                rows={3}
                value={form.notes}
                onChange={set("notes")}
              />
            </div>
          </div>
          <div className="form-actions">
            <button
              type="button"
              className="btn-ghost"
              disabled={busy}
              onClick={() => setOpen(false)}
            >
              Cancel
            </button>
            <SubmitButton busy={busy} busyText="Saving patient…">
              Save patient
            </SubmitButton>
          </div>
        </form>
      </Modal>
    </div>
  );
}
