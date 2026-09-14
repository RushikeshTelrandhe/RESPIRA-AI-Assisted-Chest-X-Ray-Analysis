import { MessageSquareText, Mail } from "lucide-react";
import { useState } from "react";
import ThemeToggle from "../theme/ThemeToggle";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  Activity,
  CircleHelp,
  Cpu,
  EyeOff,
  FileText,
  History,
  LogOut,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  ScanLine,
  Settings,
  Sparkles,
  Stethoscope,
  UserRound,
  Users,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { useResource, useWorkspace } from "../context/WorkspaceContext";
import { api, type ModelStatus } from "../services/api";
import { Brand } from "./Brand";
import { Modal } from "./UI";
import { initials } from "../utils/display";

const primaryLinks = [
  { to: "/dashboard", label: "Overview", icon: Activity },
  { to: "/xray-test", label: "New analysis", icon: ScanLine },
  { to: "/patients", label: "Patients", icon: Users },
  { to: "/history", label: "Analysis history", icon: History },
  { to: "/reports", label: "Reports", icon: FileText },
  { to: "/feedback", label: "Doctor feedback", icon: MessageSquareText },
];
const utilityLinks = [
  { to: "/models", label: "Model status", icon: Cpu },
  { to: "/profile", label: "Profile", icon: UserRound },
  { to: "/settings", label: "Preferences", icon: Settings },
  { to: "/help", label: "Help centre", icon: CircleHelp },
];
const titles: Record<string, { eyebrow: string; title: string }> = {
  "/dashboard": { eyebrow: "Clinical workspace", title: "Overview" },
  "/patients": { eyebrow: "Patient management", title: "Patients" },
  "/xray-test": { eyebrow: "Acquisition workflow", title: "New analysis" },
  "/history": { eyebrow: "Study archive", title: "Analysis history" },
  "/reports": { eyebrow: "Documentation", title: "Reports" },
  "/feedback": { eyebrow: "Clinical review", title: "Doctor feedback" },
  "/profile": { eyebrow: "Account", title: "Doctor profile" },
  "/models": { eyebrow: "Inference system", title: "Model status" },
  "/settings": { eyebrow: "Workspace", title: "Preferences" },
  "/help": { eyebrow: "Guidance", title: "Help centre" },
};

function savedCollapsed() {
  try {
    return localStorage.getItem("respira_sidebar_collapsed") === "true";
  } catch {
    return false;
  }
}

export function Shell() {
  const { doctor, token, logout } = useAuth();
  const {
    guard,
    draft,
    viewMode,
    setViewMode,
    presentationMode,
    setPresentationMode,
  } = useWorkspace();
  const status = useResource<ModelStatus>("model-status", () =>
    api.get("/api/v1/models/status", token),
  );
  const navigate = useNavigate();
  const location = useLocation();
  const [mobile, setMobile] = useState(false);
  const [collapsed, setCollapsed] = useState(savedCollapsed);
  const [confirmLogout, setConfirmLogout] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  const page = location.pathname.startsWith("/analysis/")
    ? { eyebrow: "AI study", title: "Study review" }
    : location.pathname.startsWith("/patients/")
      ? { eyebrow: "Patient management", title: "Patient record" }
      : (titles[location.pathname] ?? {
          eyebrow: "RESPIRA workspace",
          title: "Clinical intelligence",
        });
  const readyCount = status.data?.models.filter((model) => model.loaded).length ?? 0;
  const backendReady = readyCount > 0;
  const device = status.data?.device || "Checking";

  function toggleCollapsed() {
    setCollapsed((current) => {
      const next = !current;
      try {
        localStorage.setItem("respira_sidebar_collapsed", String(next));
      } catch {
        // The layout still works if preferences cannot be stored.
      }
      return next;
    });
  }

  const navList = (items: typeof primaryLinks) => (
    <>
      {items.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          title={collapsed ? item.label : undefined}
          onClick={() => setMobile(false)}
          className={({ isActive }) => `side-link${isActive ? " active" : ""}`}
        >
          <item.icon size={18} aria-hidden="true" />
          <span>{item.label}</span>
        </NavLink>
      ))}
    </>
  );

  async function finishLogout() {
    setLoggingOut(true);
    await logout();
    navigate("/login", { replace: true });
  }

  function requestLogout() {
    if (guard.active || draft.file) setConfirmLogout(true);
    else void finishLogout();
  }

  return (
    <div className={`app-shell${collapsed ? " sidebar-collapsed" : ""}`}>
      <aside className="app-sidebar">
        <div className="sidebar-brand-row">
          <Brand light compact={collapsed} to="/dashboard" />
          <button
            className="sidebar-collapse"
            type="button"
            onClick={toggleCollapsed}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </button>
        </div>

        <div className="sidebar-workspace-chip">
          <span><Stethoscope size={16} /></span>
          <div><strong>Clinical review</strong><small>Research workspace</small></div>
        </div>

        <p className="sidebar-label">Workspace</p>
        <nav className="side-links" aria-label="Workspace navigation">
          {navList(primaryLinks)}
        </nav>

        <div className="sidebar-bottom">
          <p className="sidebar-label">System & support</p>
          <nav className="side-links" aria-label="System and support">
            {navList(utilityLinks)}
          </nav>
          <button className="side-link" type="button" onClick={requestLogout}>
            <LogOut size={18} />
            <span>Log out</span>
          </button>
          <div className="sidebar-safety-note">
            <ShieldMini />
            <span>AI output requires qualified clinical review.</span>
          </div>
        </div>
      </aside>

      <div className="app-body">
        <header className="app-topbar">
          <ThemeToggle />
          <button
            className="mobile-toggle icon-button"
            type="button"
            onClick={() => setMobile(true)}
            aria-label="Open navigation"
          >
            <Menu size={20} />
          </button>
          <div className="topbar-title">
            <span>{page.eyebrow}</span>
            <strong>{page.title}</strong>
          </div>
          <div className="topbar-right">
            <NavLink
              to="/models"
              className={`system-pill${backendReady ? " ready" : ""}`}
              title="Open model status"
            >
              <i />
              <span>{status.loading ? "Checking models" : backendReady ? `${readyCount} models ready` : "Models unavailable"}</span>
              <small>{device}</small>
            </NavLink>

            <div className="view-switch" aria-label="Interface detail level">
              <button
                type="button"
                className={viewMode === "guided" ? "active" : ""}
                onClick={() => setViewMode("guided")}
              >Guided</button>
              <button
                type="button"
                className={viewMode === "clinical" ? "active" : ""}
                onClick={() => setViewMode("clinical")}
              >Clinical</button>
            </div>

            <button
              className={`presentation-toggle${presentationMode ? " active" : ""}`}
              type="button"
              aria-pressed={presentationMode}
              title="Mask patient identifiers during a presentation"
              onClick={() => setPresentationMode(!presentationMode)}
            >
              <EyeOff size={16} />
              <span>{presentationMode ? "Privacy on" : "Present"}</span>
            </button>

            <NavLink className="topbar-new-study" to="/xray-test">
              <Sparkles size={16} /> New study
            </NavLink>

            <NavLink className="user-link" to="/profile">
              <span className="avatar">{initials(doctor?.full_name)}</span>
              <span className="user-name">
                <b>{doctor?.full_name}</b>
                <small>{doctor?.specialization || "Doctor account"}</small>
              </span>
            </NavLink>
          </div>
        </header>

        {presentationMode && (
          <div className="presentation-banner" role="status">
            <EyeOff size={15} /> Presentation mode is masking patient identifiers.
            <button type="button" onClick={() => setPresentationMode(false)}>Turn off</button>
          </div>
        )}

        <main className="main-content" id="main-content">
          <div className="route-page"><Outlet /></div>
        </main>
        <footer className="workspace-footer">
          <span>RESPIRA · Research prototype</span>
          <a href="mailto:respirahelp@gmail.com" className="text-button"><Mail size={14} /> Contact developers</a>
        </footer>
      </div>

      <Modal
        open={mobile}
        title="RESPIRA navigation"
        onDismiss={() => setMobile(false)}
        className="mobile-drawer"
      >
        <nav className="side-links">
          {navList(primaryLinks)}
          <hr />
          {navList(utilityLinks)}
        </nav>
      </Modal>

      <Modal
        open={confirmLogout}
        title="Log out and discard this draft?"
        onDismiss={() => setConfirmLogout(false)}
      >
        <div className="modal-body">
          <p>Your unsaved form or selected X-ray will be cleared when you log out.</p>
          <div className="form-actions">
            <button className="btn-ghost" onClick={() => setConfirmLogout(false)}>
              Stay signed in
            </button>
            <button
              className="btn-primary"
              disabled={loggingOut}
              onClick={() => void finishLogout()}
            >
              {loggingOut ? "Logging out…" : "Log out"}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

function ShieldMini() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 3 20 6v5c0 5-3.2 8.2-8 10-4.8-1.8-8-5-8-10V6l8-3Z" stroke="currentColor" strokeWidth="1.7" />
      <path d="m8.7 12 2.1 2.1 4.7-5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
