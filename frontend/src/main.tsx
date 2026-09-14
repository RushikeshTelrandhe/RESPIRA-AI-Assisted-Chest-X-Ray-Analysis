import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import "./clinical-upgrade.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <a className="skip-link" href="#main-content">
      Skip to content
    </a>
    <App />
  </React.StrictMode>,
);
