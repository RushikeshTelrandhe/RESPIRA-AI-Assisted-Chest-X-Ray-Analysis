import { NavLink, useNavigate } from "react-router-dom";
import { Activity, Users, ScanLine, History, FileText, User, Settings, LifeBuoy, LogOut, Stethoscope, Menu, X } from "lucide-react";
import { useState } from "react";
import { useAuth } from "../context/AuthContext";

const LINKS = [
  { to: "/dashboard", label: "Overview", icon: Activity },
  { to: "/patients", label: "Patients", icon: Users },
  { to: "/xray-test", label: "X-Ray Test", icon: ScanLine },
  { to: "/history", label: "Test History", icon: History },
  { to: "/reports", label: "Reports", icon: FileText },
  { to: "/profile", label: "Profile", icon: User },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const { doctor, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const nav = (
    <nav className="flex flex-col gap-1 p-4">
      {LINKS.map((l) => (
        <NavLink
          key={l.to}
          to={l.to}
          onClick={() => setOpen(false)}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
              isActive ? "bg-brand-600 text-white" : "text-slate-600 hover:bg-slate-100"
            }`
          }
        >
          <l.icon size={18} /> {l.label}
        </NavLink>
      ))}
      <div className="mt-6 border-t border-slate-200 pt-4 flex flex-col gap-1">
        <NavLink to="/settings" className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-slate-600 hover:bg-slate-100">
          <Settings size={18} /> Settings
        </NavLink>
        <NavLink to="/help" className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-slate-600 hover:bg-slate-100">
          <LifeBuoy size={18} /> Help
        </NavLink>
        <button
          onClick={() => { void logout().then(() => navigate("/login")); }}
          className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-slate-600 hover:bg-slate-100 text-left"
        >
          <LogOut size={18} /> Logout
        </button>
      </div>
    </nav>
  );

  return (
    <div className="min-h-screen lg:flex">
      <aside className="hidden lg:flex w-64 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="flex items-center gap-2 px-5 pt-5 pb-2">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-brand-600 text-white"><Stethoscope size={20} /></span>
          <div>
            <div className="font-bold tracking-wide">RESPIRA</div>
            <div className="text-[11px] text-slate-500">Chest X-Ray AI</div>
          </div>
        </div>
        {nav}
      </aside>

      <div className="flex-1 min-w-0">
        <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-slate-200 bg-white/90 px-4 py-3 backdrop-blur">
          <button className="lg:hidden btn-ghost !px-2 !py-2" onClick={() => setOpen(!open)} aria-label="Menu">
            {open ? <X size={18} /> : <Menu size={18} />}
          </button>
          <div className="ml-auto flex items-center gap-3">
            <div className="text-right hidden sm:block">
              <div className="text-sm font-semibold">{doctor?.full_name}</div>
              <div className="text-xs text-slate-500">{doctor?.specialization || "Doctor"}</div>
            </div>
            <span className="grid h-9 w-9 place-items-center rounded-full bg-brand-100 font-bold text-brand-700">
              {(doctor?.full_name ?? "D").charAt(0)}
            </span>
          </div>
        </header>
        {open && <div className="lg:hidden border-b border-slate-200 bg-white">{nav}</div>}
        <main className="mx-auto max-w-6xl p-4 sm:p-6">{children}</main>
      </div>
    </div>
  );
}
