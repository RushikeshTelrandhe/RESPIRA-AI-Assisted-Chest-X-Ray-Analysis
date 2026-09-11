import { Navigate } from "react-router-dom";
import type { ReactNode } from "react";
import { useAuth } from "../context/AuthContext";

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { doctor, token, loading } = useAuth();
  if (loading) return <div className="min-h-screen grid place-items-center text-slate-500">Loading Respira…</div>;
  if (!token || !doctor) return <Navigate to="/login" replace />;
  return <>{children}</>;
}
