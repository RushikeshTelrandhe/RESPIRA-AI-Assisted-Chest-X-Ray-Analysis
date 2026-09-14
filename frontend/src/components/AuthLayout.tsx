import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Mail, BrainCircuit, Layers3, ShieldCheck } from "lucide-react";
import { Brand } from "./Brand";
export function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="auth-page">
      <aside className="auth-art-panel">
        <img
          src="/media/respira-intro-anatomy.webp"
          alt=""
          width="1672"
          height="941"
        />
        <div className="auth-brand-row">
          <Brand light to="/login" />
          <span className="auth-secure-chip">
            <ShieldCheck size={14} /> Clinical workspace
          </span>
        </div>
        <div className="auth-art-copy">
          <p className="eyebrow">Explainable review begins here</p>
          <h2>See beyond the prediction.</h2>
          <p>
            Bring patient context, model output, uncertainty and visual
            evidence together in one focused chest X-ray workspace.
          </p>
          <div className="auth-feature-row">
            <span><BrainCircuit size={17} /> Four model choices</span>
            <span><Layers3 size={17} /> Visual explainability</span>
          </div>
          <small>
            Research prototype · AI-assisted output requires qualified
            clinical review.
          </small>
        </div>
        <div className="auth-scan-corners" aria-hidden="true"><i /><i /><i /><i /></div>
      </aside>
      <main className="auth-content" id="main-content">
        <a className="auth-back" href="mailto:respirahelp@gmail.com">
          <Mail size={16} /> Need help signing in?
        </a>
        {children}
      </main>
    </div>
  );
}
