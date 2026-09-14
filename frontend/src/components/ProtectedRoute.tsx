import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { LoadingBlock } from "./UI";
export function ProtectedRoute() {
  const { doctor, token, loading } = useAuth();
  const location = useLocation();
  if (loading)
    return (
      <main className="main-content">
        <LoadingBlock label="Opening RESPIRA…" />
      </main>
    );
  if (!token || !doctor)
    return (
      <Navigate
        to="/login"
        replace
        state={{ from: location.pathname + location.search }}
      />
    );
  return <Outlet />;
}
