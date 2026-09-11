import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const STEPS = ["Personal information", "Professional information", "Security", "Confirmation"];

export function Signup() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [f, setF] = useState({ full_name: "", license_no: "", email: "", phone: "", hospital: "", specialization: "", password: "", confirm_password: "" });
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) => setF({ ...f, [k]: e.target.value });

  const strength = (p: string) => (p.length >= 12 ? "Strong" : p.length >= 8 ? "Medium" : "Weak");

  const next = () => {
    setError("");
    if (step === 0 && (!f.full_name.trim() || !/.+@.+\..+/.test(f.email))) return setError("Enter your full name and a valid email.");
    if (step === 1 && (!f.license_no.trim())) return setError("Medical registration / license number is required.");
    if (step === 2 && (f.password.length < 8 || f.password !== f.confirm_password)) return setError("Password must be 8+ characters and match the confirmation.");
    setStep(step + 1);
  };

  const submit = async () => {
    setError(""); setBusy(true);
    try { await signup(f); navigate("/dashboard"); }
    catch (err) { setError(err instanceof Error ? err.message : "Signup failed"); }
    finally { setBusy(false); }
  };

  return (
    <div className="mx-auto max-w-2xl p-6">
      <h1 className="text-2xl font-bold">Create Doctor Account</h1>
      <ol className="mt-4 flex gap-2" aria-label="Signup progress">
        {STEPS.map((s, i) => (
          <li key={s} className={`flex-1 rounded-lg border p-2 text-center text-xs font-semibold ${i === step ? "border-brand-600 bg-brand-50 text-brand-700" : i < step ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-slate-200 text-slate-400"}`}>{s}</li>
        ))}
      </ol>
      {error && <div role="alert" className="mt-4 rounded-lg bg-rose-50 border border-rose-200 p-3 text-sm text-rose-700">{error}</div>}
      <div className="card mt-4 space-y-4 p-6">
        {step === 0 && (<>
          <div><label className="label" htmlFor="fn">Full name</label><input id="fn" className="input" value={f.full_name} onChange={set("full_name")} /></div>
          <div><label className="label" htmlFor="em">Email</label><input id="em" className="input" type="email" value={f.email} onChange={set("email")} /></div>
          <div><label className="label" htmlFor="ph">Phone</label><input id="ph" className="input" value={f.phone} onChange={set("phone")} /></div>
        </>)}
        {step === 1 && (<>
          <div><label className="label" htmlFor="lic">Medical registration / license number</label><input id="lic" className="input" value={f.license_no} onChange={set("license_no")} /></div>
          <div><label className="label" htmlFor="hosp">Hospital / clinic</label><input id="hosp" className="input" value={f.hospital} onChange={set("hospital")} /></div>
          <div><label className="label" htmlFor="spec">Specialization</label><input id="spec" className="input" value={f.specialization} onChange={set("specialization")} placeholder="e.g. Radiology" /></div>
        </>)}
        {step === 2 && (<>
          <div><label className="label" htmlFor="pw1">Password</label><input id="pw1" className="input" type="password" value={f.password} onChange={set("password")} autoComplete="new-password" /><div className="mt-1 text-xs text-slate-500">Strength: {strength(f.password)}</div></div>
          <div><label className="label" htmlFor="pw2">Confirm password</label><input id="pw2" className="input" type="password" value={f.confirm_password} onChange={set("confirm_password")} autoComplete="new-password" /></div>
        </>)}
        {step === 3 && (
          <dl className="grid gap-2 text-sm">
            <div className="flex justify-between"><dt className="text-slate-500">Name</dt><dd className="font-medium">{f.full_name}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Email</dt><dd className="font-medium">{f.email}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">License</dt><dd className="font-medium">{f.license_no}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Hospital</dt><dd className="font-medium">{f.hospital || "—"}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Specialization</dt><dd className="font-medium">{f.specialization || "—"}</dd></div>
          </dl>
        )}
        <div className="flex justify-between pt-2">
          <div>{step > 0 && <button className="btn-ghost" onClick={() => setStep(step - 1)}>Back</button>}</div>
          {step < 3 ? <button className="btn-primary" onClick={next}>Continue</button>
            : <button className="btn-primary" onClick={submit} disabled={busy}>{busy ? "Creating…" : "Create account"}</button>}
        </div>
      </div>
      <p className="mt-4 text-sm text-slate-500">Already registered? <Link to="/login" className="font-semibold text-brand-700">Sign in</Link></p>
    </div>
  );
}
