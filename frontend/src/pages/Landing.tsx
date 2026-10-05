import { hasSeenIntro } from "../services/introSession";
import { Link, Navigate } from "react-router-dom";
import {
  ArrowRight,
  BrainCircuit,
  FileCheck2,
  Layers3,
  ScanLine,
  ShieldCheck,
  Sparkles,
  UserRound,
} from "lucide-react";
import { Brand } from "../components/Brand";
import { useAuth } from "../context/AuthContext";
import { DISEASES } from "../services/api";

const steps = [
  {
    icon: UserRound,
    title: "Connect the patient",
    copy: "Keep every radiograph linked to the correct patient and study context.",
  },
  {
    icon: ScanLine,
    title: "Acquire the study",
    copy: "Validate a PNG or JPG chest X-ray before it enters the analysis workflow.",
  },
  {
    icon: BrainCircuit,
    title: "Run a ready model",
    copy: "Choose only from checkpoints that the backend reports as available.",
  },
  {
    icon: Layers3,
    title: "Review the evidence",
    copy: "Compare model scores, uncertainty and available visual explanations.",
  },
];

export function Landing() {
  const { token } = useAuth();
  const workspace = token ? "/dashboard" : "/login";

  if (hasSeenIntro()) return <Navigate to={workspace} replace />;

  return (
    <div className="landing landing-v2">
      <nav className="public-nav public-nav-v2" aria-label="Main">
        <Brand light />
        <div className="public-nav-links">
          <a href="#workflow">Workflow</a>
          <a href="#explainability">Explainability</a>
          <Link to={workspace} className="btn-light">
            {token ? "Open workspace" : "Clinical sign in"}
            <ArrowRight size={17} />
          </Link>
        </div>
      </nav>

      <main id="main-content">
        <section className="hero-section hero-section-v2">
          <img
            className="hero-art hero-art-v2"
            src="/media/respira-intro-anatomy.webp"
            alt=""
            width="1672"
            height="941"
            fetchPriority="high"
          />
          <div className="hero-shade hero-shade-v2" />
          <div className="hero-atmosphere" aria-hidden="true" />
          <div className="hero-body hero-body-v2">
            <div className="hero-copy hero-copy-v2">
              <div className="hero-kicker">
                <span className="live-dot" />
                Explainable chest X-ray intelligence
              </div>
              <h1>
                Clinical clarity,
                <span>from image to insight.</span>
              </h1>
              <p>
                RESPIRA brings model predictions, uncertainty and visual
                explanations into one focused chest X-ray review workspace.
              </p>
              <div className="hero-buttons">
                <Link className="btn-light btn-hero" to={workspace}>
                  {token ? "Enter clinical workspace" : "Sign in to RESPIRA"}
                  <ArrowRight size={18} />
                </Link>
                {!token && (
                  <Link className="hero-secondary btn-hero" to="/signup">
                    Create doctor account
                  </Link>
                )}
              </div>
              <div className="hero-trust-row">
                <span><ShieldCheck size={16} /> Clinician-centred review</span>
                <span><Sparkles size={16} /> Transparent AI evidence</span>
              </div>
            </div>

            <aside className="hero-diagnostic-card" aria-label="Product preview">
              <div className="diagnostic-card-head">
                <span><ScanLine size={15} /> Study preview</span>
                <span className="diagnostic-ready"><i /> Ready</span>
              </div>
              <div className="diagnostic-frame">
                <img
                  src="/media/respira-intro-poster.webp"
                  alt="Synthetic chest X-ray interface preview"
                />
                <div className="diagnostic-scanline" aria-hidden="true" />
                <span className="image-marker marker-left">L</span>
                <span className="image-marker marker-right">R</span>
              </div>
              <div className="diagnostic-card-foot">
                <span>Prediction</span><strong>Model output</strong>
                <span>Evidence</span><strong>Explainability</strong>
              </div>
            </aside>
          </div>
          <div className="hero-bottom hero-bottom-v2">
            <span>Designed for thoughtful radiograph review.</span>
            <span>Research prototype · Qualified clinical review required</span>
          </div>
        </section>

        <section className="public-section workflow-section" id="workflow">
          <div className="public-section-heading">
            <div>
              <p className="eyebrow">A guided clinical path</p>
              <h2>One study. One continuous review.</h2>
            </div>
            <p>
              Clear steps keep patient context, model readiness and
              explainability visible throughout the workflow.
            </p>
          </div>
          <div className="how-grid how-grid-v2">
            {steps.map((step, index) => (
              <article className="how-step how-step-v2" key={step.title}>
                <div className="how-step-top">
                  <span className="step-number">0{index + 1}</span>
                  <span className="step-icon"><step.icon size={21} /></span>
                </div>
                <h3>{step.title}</h3>
                <p>{step.copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="explainability-showcase" id="explainability">
          <div className="public-section explainability-showcase-inner">
            <div className="explainability-copy">
              <p className="eyebrow">Built for transparent review</p>
              <h2>See the output—and what influenced it.</h2>
              <p>
                Inspect class scores and model uncertainty, then request
                Grad-CAM or ViT attention only when the analysis supports it.
                Visual explanations are model evidence, never confirmed lesion
                boundaries.
              </p>
              <ul>
                <li><ShieldCheck size={17} /> Honest model availability</li>
                <li><FileCheck2 size={17} /> Traceable study history</li>
                <li><Layers3 size={17} /> Aligned comparison tools</li>
              </ul>
            </div>
            <div className="explainability-visual" aria-hidden="true">
              <img src="/media/respira-intro-poster.webp" alt="" />
              <div className="attention-orbit orbit-one" />
              <div className="attention-orbit orbit-two" />
              <span className="attention-label">AI influence view</span>
            </div>
          </div>
        </section>

        <section className="category-strip category-strip-v2">
          <div className="public-section">
            <p className="eyebrow">Six configured output classes</p>
            <div className="category-tags">
              {DISEASES.map((d) => <span key={d}>{d}</span>)}
            </div>
            <p className="small muted">
              These are model categories, not a complete assessment of a chest
              radiograph or a definitive diagnosis.
            </p>
          </div>
        </section>
      </main>

      <footer className="public-footer public-footer-v2">
        <Brand />
        <p>
          RESPIRA is a research prototype. Every AI output requires review by a
          qualified clinician and appropriate clinical correlation.
        </p>
      </footer>
    </div>
  );
}
