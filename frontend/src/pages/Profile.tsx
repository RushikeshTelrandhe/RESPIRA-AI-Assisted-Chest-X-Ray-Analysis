import { useState, type FormEvent } from "react";
import { api, type Doctor } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useUnsavedChanges } from "../context/WorkspaceContext";
import {
  Alert,
  PageHeader,
  PasswordInput,
  SubmitButton,
  Tabs,
} from "../components/UI";
import { errorText, initials } from "../utils/display";
type Tab = "personal" | "professional" | "security";
export function Profile() {
  const { doctor, token, refresh } = useAuth();
  const [tab, setTab] = useState<Tab>("personal");
  const base = {
    full_name: doctor?.full_name ?? "",
    phone: doctor?.phone ?? "",
    hospital: doctor?.hospital ?? "",
    specialization: doctor?.specialization ?? "",
    department: doctor?.department ?? "",
    experience_years: String(doctor?.experience_years ?? 0),
  };
  const [form, setForm] = useState(base);
  const [saved, setSaved] = useState(JSON.stringify(base));
  const [pw, setPw] = useState({ current_password: "", new_password: "" });
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{
    kind: "error" | "success";
    text: string;
  } | null>(null);
  const dirty =
    JSON.stringify(form) !== saved ||
    Boolean(pw.current_password || pw.new_password);
  useUnsavedChanges(
    dirty,
    "Your profile changes have not been saved. Leave and discard them?",
  );
  if (!doctor) return null;
  const set =
    (key: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
      setForm({ ...form, [key]: e.target.value });
  async function save(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      await api.put<Doctor>(
        "/api/v1/doctors/profile",
        { ...form, experience_years: Number(form.experience_years) || 0 },
        token,
      );
      setSaved(JSON.stringify(form));
      await refresh();
      setMessage({ kind: "success", text: "Profile updated." });
    } catch (e) {
      setMessage({ kind: "error", text: errorText(e) });
    } finally {
      setBusy(false);
    }
  }
  async function password(e: FormEvent) {
    e.preventDefault();
    if (pw.new_password.length < 8) {
      setMessage({
        kind: "error",
        text: "The new password must have at least 8 characters.",
      });
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      await api.post("/api/v1/doctors/change-password", pw, token);
      setPw({ current_password: "", new_password: "" });
      setMessage({ kind: "success", text: "Password changed." });
    } catch (e) {
      setMessage({ kind: "error", text: errorText(e) });
    } finally {
      setBusy(false);
    }
  }
  const options = [
    { value: "personal" as const, label: "Personal" },
    { value: "professional" as const, label: "Professional" },
    { value: "security" as const, label: "Security" },
  ];
  return (
    <div className="stack">
      <PageHeader
        eyebrow="Doctor account"
        title="Profile"
        description="Manage the details linked to your RESPIRA account."
      />
      <section className="patient-header">
        <span className="avatar avatar-lg">{initials(doctor.full_name)}</span>
        <div>
          <h2>{doctor.full_name}</h2>
          <p>{doctor.email}</p>
        </div>
        <span className="badge neutral">
          {doctor.specialization || "Doctor"}
        </span>
      </section>
      <Tabs
        id="profile-tabs"
        options={options}
        value={tab}
        onChange={(v) => {
          setTab(v);
          setMessage(null);
        }}
      />
      {message && <Alert kind={message.kind}>{message.text}</Alert>}
      <section
        className="panel"
        id="profile-tabs-panel"
        role="tabpanel"
        aria-labelledby={"profile-tabs-" + tab}
      >
        {tab === "personal" && (
          <form className="form-grid" onSubmit={save}>
            <div>
              <label className="label" htmlFor="profile-name">
                Full name
              </label>
              <input
                id="profile-name"
                className="input"
                required
                value={form.full_name}
                onChange={set("full_name")}
              />
            </div>
            <div>
              <label className="label" htmlFor="profile-email">
                Email
              </label>
              <input
                id="profile-email"
                className="input"
                value={doctor.email}
                disabled
              />
              <p className="field-hint">Email is managed by your account.</p>
            </div>
            <div>
              <label className="label" htmlFor="profile-phone">
                Phone
              </label>
              <input
                id="profile-phone"
                className="input"
                type="tel"
                value={form.phone}
                onChange={set("phone")}
              />
            </div>
            <div>
              <label className="label" htmlFor="profile-specialization">
                Specialization
              </label>
              <input
                id="profile-specialization"
                className="input"
                value={form.specialization}
                onChange={set("specialization")}
              />
            </div>
            <div className="form-actions full-width">
              <SubmitButton busy={busy} disabled={!dirty}>
                Save changes
              </SubmitButton>
            </div>
          </form>
        )}
        {tab === "professional" && (
          <form className="form-grid" onSubmit={save}>
            <div>
              <label className="label" htmlFor="profile-license">
                Medical registration / license
              </label>
              <input
                id="profile-license"
                className="input"
                value={doctor.license_no}
                disabled
              />
            </div>
            <div>
              <label className="label" htmlFor="profile-hospital">
                Hospital / clinic
              </label>
              <input
                id="profile-hospital"
                className="input"
                value={form.hospital}
                onChange={set("hospital")}
              />
            </div>
            <div>
              <label className="label" htmlFor="profile-department">
                Department
              </label>
              <input
                id="profile-department"
                className="input"
                value={form.department}
                onChange={set("department")}
              />
            </div>
            <div>
              <label className="label" htmlFor="profile-experience">
                Years of experience
              </label>
              <input
                id="profile-experience"
                className="input"
                type="number"
                min="0"
                max="80"
                value={form.experience_years}
                onChange={set("experience_years")}
              />
            </div>
            <div className="form-actions full-width">
              <SubmitButton busy={busy} disabled={!dirty}>
                Save changes
              </SubmitButton>
            </div>
          </form>
        )}
        {tab === "security" && (
          <form className="stack" onSubmit={password}>
            <PasswordInput
              id="current-password"
              label="Current password"
              required
              autoComplete="current-password"
              value={pw.current_password}
              onChange={(e) =>
                setPw({ ...pw, current_password: e.target.value })
              }
            />
            <PasswordInput
              id="profile-new-password"
              label="New password"
              required
              minLength={8}
              autoComplete="new-password"
              value={pw.new_password}
              onChange={(e) => setPw({ ...pw, new_password: e.target.value })}
            />
            <p className="field-hint">Use at least 8 characters.</p>
            <div>
              <SubmitButton
                busy={busy}
                disabled={!pw.current_password || pw.new_password.length < 8}
                busyText="Changing password…"
              >
                Change password
              </SubmitButton>
            </div>
          </form>
        )}
      </section>
    </div>
  );
}
