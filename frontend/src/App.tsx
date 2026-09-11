import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { Shell } from "./components/Shell";
import { Landing } from "./pages/Landing";
import { Login } from "./pages/Login";
import { Signup } from "./pages/Signup";
import { Dashboard } from "./pages/Dashboard";
import { Patients } from "./pages/Patients";
import { PatientDetail } from "./pages/PatientDetail";
import { XrayTest } from "./pages/XrayTest";
import { AnalysisDetailPage } from "./pages/AnalysisDetail";
import { Explainability } from "./pages/Explainability";
import { History } from "./pages/History";
import { Reports } from "./pages/Reports";
import { Profile } from "./pages/Profile";
import { Models, Settings, Help } from "./pages/Misc";

function Guarded({ children }: { children: React.ReactNode }) {
  return (
    <ProtectedRoute>
      <Shell>{children}</Shell>
    </ProtectedRoute>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/dashboard" element={<Guarded><Dashboard /></Guarded>} />
          <Route path="/patients" element={<Guarded><Patients /></Guarded>} />
          <Route path="/patients/:id" element={<Guarded><PatientDetail /></Guarded>} />
          <Route path="/xray-test" element={<Guarded><XrayTest /></Guarded>} />
          <Route path="/analysis/:id" element={<Guarded><AnalysisDetailPage /></Guarded>} />
          <Route path="/analysis/:id/explainability" element={<Guarded><Explainability /></Guarded>} />
          <Route path="/history" element={<Guarded><History /></Guarded>} />
          <Route path="/reports" element={<Guarded><Reports /></Guarded>} />
          <Route path="/profile" element={<Guarded><Profile /></Guarded>} />
          <Route path="/settings" element={<Guarded><Settings /></Guarded>} />
          <Route path="/models" element={<Guarded><Models /></Guarded>} />
          <Route path="/help" element={<Guarded><Help /></Guarded>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
