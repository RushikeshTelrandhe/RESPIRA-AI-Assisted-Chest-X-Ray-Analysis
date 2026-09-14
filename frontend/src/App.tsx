import { lazy, Suspense } from "react";
import {
  Navigate,
  Outlet,
  RouterProvider,
  createBrowserRouter,
  useLocation,
  useNavigate,
} from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { WorkspaceProvider } from "./context/WorkspaceContext";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { Shell } from "./components/Shell";
import { LoadingBlock } from "./components/UI";
import { WelcomeIntro } from "./components/WelcomeIntro";
const Landing = lazy(() =>
  import("./pages/Landing").then((m) => ({ default: m.Landing })),
);
const Login = lazy(() =>
  import("./pages/Login").then((m) => ({ default: m.Login })),
);
const Signup = lazy(() =>
  import("./pages/Signup").then((m) => ({ default: m.Signup })),
);
const Dashboard = lazy(() =>
  import("./pages/Dashboard").then((m) => ({ default: m.Dashboard })),
);
const Patients = lazy(() =>
  import("./pages/Patients").then((m) => ({ default: m.Patients })),
);
const PatientDetail = lazy(() =>
  import("./pages/PatientDetail").then((m) => ({ default: m.PatientDetail })),
);
const XrayTest = lazy(() =>
  import("./pages/XrayTest").then((m) => ({ default: m.XrayTest })),
);
const AnalysisDetailPage = lazy(() =>
  import("./pages/AnalysisDetail").then((m) => ({
    default: m.AnalysisDetailPage,
  })),
);
const Explainability = lazy(() =>
  import("./pages/Explainability").then((m) => ({ default: m.Explainability })),
);
const History = lazy(() =>
  import("./pages/History").then((m) => ({ default: m.History })),
);
const Reports = lazy(() =>
  import("./pages/Reports").then((m) => ({ default: m.Reports })),
);
const Profile = lazy(() =>
  import("./pages/Profile").then((m) => ({ default: m.Profile })),
);
const Models = lazy(() =>
  import("./pages/Misc").then((m) => ({ default: m.Models })),
);
const Settings = lazy(() =>
  import("./pages/Misc").then((m) => ({ default: m.Settings })),
);
const Help = lazy(() =>
  import("./pages/Misc").then((m) => ({ default: m.Help })),
);
const DoctorFeedback = lazy(() => import("./pages/DoctorFeedback").then(m => ({ default: m.DoctorFeedback })));
function WorkspaceBoundary() {
  const { token } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const finishIntro = () => {
    if (location.pathname === "/") navigate(token ? "/dashboard" : "/login", { replace: true });
  };
  return (
    <>
      <WelcomeIntro onComplete={finishIntro} />
      <WorkspaceProvider key={token ?? "public"}>
      <Suspense fallback={<LoadingBlock label="Opening page…" />}>
        <Outlet />
      </Suspense>
      </WorkspaceProvider>
    </>
  );
}
const router = createBrowserRouter([
  {
    element: <WorkspaceBoundary />,
    children: [
      { path: "/", element: <Landing /> },
      { path: "/login", element: <Login /> },
      { path: "/signup", element: <Signup /> },
      {
        element: <ProtectedRoute />,
        children: [
          {
            element: <Shell />,
            children: [
              { path: "/dashboard", element: <Dashboard /> },
              { path: "/patients", element: <Patients /> },
              { path: "/patients/:id", element: <PatientDetail /> },
              { path: "/xray-test", element: <XrayTest /> },
              { path: "/analysis/:id", element: <AnalysisDetailPage /> },
              {
                path: "/analysis/:id/explainability",
                element: <Explainability />,
              },
              { path: "/history", element: <History /> },
              { path: "/reports", element: <Reports /> },
              { path: "/feedback", element: <DoctorFeedback /> },
              { path: "/profile", element: <Profile /> },
              { path: "/settings", element: <Settings /> },
              { path: "/models", element: <Models /> },
              { path: "/help", element: <Help /> },
            ],
          },
        ],
      },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
]);
export default function App() {
  return (
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  );
}
