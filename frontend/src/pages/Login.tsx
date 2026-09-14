import { useRef, useState, type FormEvent } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { Mail, ShieldCheck } from "lucide-react";
import { AuthLayout } from "../components/AuthLayout";
import { Alert, PasswordInput, SubmitButton } from "../components/UI";
import { useAuth } from "../context/AuthContext";
import { errorText, safeReturnPath } from "../utils/display";
export function Login() {
  const { login, doctor } = useAuth();
  const nav = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pending = useRef(false);
  const destination = safeReturnPath(
    (location.state as { from?: string } | null)?.from,
  );
  if (doctor) return <Navigate to={destination} replace />;
  async function submit(e: FormEvent) {
    e.preventDefault();
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    setError("");
    try {
      await login(email.trim(), password);
      nav(destination, { replace: true });
    } catch (e) {
      setError(errorText(e));
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }
  return (
    <AuthLayout>
      <div className="auth-title">
        <p className="eyebrow">Clinical intelligence workspace</p>
        <h1>Welcome back to RESPIRA.</h1>
        <p>Sign in with your registered doctor account to continue the review.</p>
      </div>
      <form className="auth-form stack" onSubmit={submit}>
        {error && <Alert>{error}</Alert>}
        <div>
          <label className="label" htmlFor="email">
            Email address
          </label>
          <input
            className="input"
            id="email"
            type="email"
            autoComplete="username"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@hospital.org"
          />
        </div>
        <PasswordInput
          id="password"
          label="Password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <SubmitButton busy={busy} busyText="Signing in…">
          Sign in
        </SubmitButton>
      </form>
      <p className="auth-bottom">
        New to RESPIRA? <Link to="/signup">Create a doctor account</Link>
      </p>
      <div className="auth-support-row">
        <a href="mailto:respirahelp@gmail.com"><Mail size={14} /> Contact developers</a>
        <span><ShieldCheck size={14} /> Clinician review required</span>
      </div>
    </AuthLayout>
  );
}
