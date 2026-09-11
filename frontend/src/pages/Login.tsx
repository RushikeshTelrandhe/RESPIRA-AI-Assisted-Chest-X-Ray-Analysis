import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Stethoscope } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(""); setBusy(true);
    try { await login(email.trim(), password); navigate("/dashboard"); }
    catch (err) { setError(err instanceof Error ? err.message : "Sign in failed"); }
    finally { setBusy(false); }
  };

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <div className="hidden flex-col justify-center bg-brand-700 p-12 text-white lg:flex">
        <div className="flex items-center gap-2"><Stethoscope size={28} /><span className="text-2xl font-bold tracking-wide">RESPIRA</span></div>
        <p className="mt-2 text-brand-100">AI-Assisted Chest X-Ray Analysis</p>
        <div className="mt-8 space-y-3 text-sm text-brand-50">
          <div>EfficientNet-B0 + ViT-B/16 fusion</div>
          <div>Bidirectional cross-attention · disease-conditioned attention</div>
          <div>Grad-CAM + ViT attention explainability</div>
          <div>Entropy-based uncertainty on every prediction</div>
        </div>
      </div>
      <div className="flex items-center justify-center p-6">
        <form onSubmit={submit} className="card w-full max-w-md space-y-4 p-8">
          <h1 className="text-2xl font-bold">Doctor Sign In</h1>
          {error && <div role="alert" className="rounded-lg bg-rose-50 border border-rose-200 p-3 text-sm text-rose-700">{error}</div>}
          <div><label className="label" htmlFor="email">Email</label><input id="email" className="input" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" /></div>
          <div><label className="label" htmlFor="pw">Password</label><input id="pw" className="input" type="password" required value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" /></div>
          <button className="btn-primary w-full" disabled={busy}>{busy ? "Signing in…" : "Sign In"}</button>
          <p className="text-sm text-slate-500">No account? <Link to="/signup" className="font-semibold text-brand-700">Create a doctor account</Link></p>
        </form>
      </div>
    </div>
  );
}
