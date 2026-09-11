import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, type Doctor } from "../services/api";

type AuthCtx = {
  doctor: Doctor | null;
  token: string | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (payload: Record<string, string>) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
};

const Ctx = createContext<AuthCtx | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [doctor, setDoctor] = useState<Doctor | null>(null);
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("respira_token"));
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    const t = localStorage.getItem("respira_token");
    if (!t) { setDoctor(null); setToken(null); setLoading(false); return; }
    try {
      const me = await api.get<Doctor>("/api/v1/auth/me", t);
      setDoctor(me); setToken(t);
    } catch {
      localStorage.removeItem("respira_token");
      setDoctor(null); setToken(null);
    } finally { setLoading(false); }
  };

  useEffect(() => { void refresh(); }, []);

  const login = async (email: string, password: string) => {
    const r = await api.post<{ access_token: string; doctor: Doctor }>("/api/v1/auth/login", { email, password });
    localStorage.setItem("respira_token", r.access_token);
    setToken(r.access_token); setDoctor(r.doctor);
  };

  const signup = async (payload: Record<string, string>) => {
    const r = await api.post<{ access_token: string; doctor: Doctor }>("/api/v1/auth/signup", payload);
    localStorage.setItem("respira_token", r.access_token);
    setToken(r.access_token); setDoctor(r.doctor);
  };

  const logout = async () => {
    try { if (token) await api.post("/api/v1/auth/logout", {}, token); } catch { /* noop */ }
    localStorage.removeItem("respira_token");
    setDoctor(null); setToken(null);
  };

  return <Ctx.Provider value={{ doctor, token, loading, login, signup, logout, refresh }}>{children}</Ctx.Provider>;
}

export function useAuth(): AuthCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAuth outside provider");
  return v;
}
