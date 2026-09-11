import { contextBridge } from "electron";

// Minimal, audited bridge. No unrestricted Node.js APIs are exposed.
contextBridge.exposeInMainWorld("respira", {
  version: "1.0.0",
  backendUrl: process.env.RESPIRA_BACKEND_URL ?? "http://127.0.0.1:8000",
});
