import { useRef, useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { AuthLayout } from "../components/AuthLayout";
import { Alert, PasswordInput, SubmitButton } from "../components/UI";
import { useAuth } from "../context/AuthContext";
import { useUnsavedChanges } from "../context/WorkspaceContext";
import { errorText } from "../utils/display";
const steps = ["Personal", "Professional", "Security", "Review"];
export function Signup() {
  const { signup, doctor } = useAuth();
  const nav = useNavigate();
  const pending = useRef(false);
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [f, setF] = useState({
    full_name: "",
    license_no: "",
    email: "",
    phone: "",
    hospital: "",
    specialization: "",
    password: "",
    confirm_password: "",
  });
  useUnsavedChanges(
    !doctor && Object.values(f).some(Boolean),
    "Your account form has not been submitted. Leave and discard it?",
  );
  const set =
    (key: keyof typeof f) => (e: React.ChangeEvent<HTMLInputElement>) =>
      setF({ ...f, [key]: e.target.value });
  if (doctor) return <Navigate to="/dashboard" replace />;
  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    if (
      step === 2 &&
      (f.password.length < 8 || f.password !== f.confirm_password)
    ) {
      setError("Use at least 8 characters and enter the same password twice.");
      return;
    }
    if (step < 3) {
      setStep(step + 1);
      return;
    }
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    try {
      await signup(f);
      nav("/dashboard", { replace: true });
    } catch (e) {
      setError(errorText(e));
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }
  const field = (
    key: keyof typeof f,
    label: string,
    required = false,
    type = "text",
    autoComplete?: string,
  ) => (
    <div>
      <label className="label" htmlFor={key}>
        {label}
        {required ? " *" : ""}
      </label>
      <input
        className="input"
        id={key}
        type={type}
        required={required}
        autoComplete={autoComplete}
        value={f[key]}
        onChange={set(key)}
      />
    </div>
  );
  return (
    <AuthLayout>
      <div className="auth-title">
        <p className="eyebrow">Join the workspace</p>
        <h1>Create your account.</h1>
        <p>A doctor account for research and clinician review.</p>
      </div>
      <ol className="signup-steps" aria-label="Registration progress">
        {steps.map((s, i) => (
          <li
            key={s}
            className={i === step ? "active" : i < step ? "done" : ""}
            aria-current={i === step ? "step" : undefined}
          >
            <span>{i + 1}</span> {s}
          </li>
        ))}
      </ol>
      <form className="auth-form stack" onSubmit={submit}>
        {error && <Alert>{error}</Alert>}
        {step === 0 && (
          <>
            {field("full_name", "Full name", true, "text", "name")}
            {field("email", "Email address", true, "email", "email")}
            {field("phone", "Phone", false, "tel", "tel")}
          </>
        )}
        {step === 1 && (
          <>
            {field("license_no", "Medical registration / license number", true)}
            {field(
              "hospital",
              "Hospital / clinic",
              false,
              "text",
              "organization",
            )}
            {field("specialization", "Specialization")}
          </>
        )}
        {step === 2 && (
          <>
            <PasswordInput
              id="new-password"
              label="Password *"
              required
              minLength={8}
              autoComplete="new-password"
              value={f.password}
              onChange={set("password")}
            />
            <p className="field-hint">Use at least 8 characters.</p>
            <PasswordInput
              id="confirm-password"
              label="Confirm password *"
              required
              minLength={8}
              autoComplete="new-password"
              value={f.confirm_password}
              onChange={set("confirm_password")}
            />
          </>
        )}
        {step === 3 && (
          <>
            <h2>Check your details</h2>
            <dl className="detail-list">
              {(
                [
                  "full_name",
                  "email",
                  "license_no",
                  "phone",
                  "hospital",
                  "specialization",
                ] as const
              ).map((k) => (
                <div key={k}>
                  <dt>
                    {
                      {
                        full_name: "Name",
                        email: "Email",
                        license_no: "License",
                        phone: "Phone",
                        hospital: "Hospital",
                        specialization: "Specialization",
                      }[k]
                    }
                  </dt>
                  <dd>{f[k] || "Not provided"}</dd>
                </div>
              ))}
            </dl>
          </>
        )}
        <div className="form-actions">
          {step > 0 && (
            <button
              type="button"
              className="btn-ghost"
              disabled={busy}
              onClick={() => {
                setError("");
                setStep(step - 1);
              }}
            >
              Back
            </button>
          )}
          <SubmitButton busy={busy} busyText="Creating account…">
            {step === 3 ? "Create account" : "Continue"}
          </SubmitButton>
        </div>
      </form>
      <p className="auth-bottom">
        Already registered? <Link to="/login">Sign in</Link>
      </p>
    </AuthLayout>
  );
}
