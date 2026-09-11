import { app, BrowserWindow, shell } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";
import net from "node:net";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BACKEND_URL = process.env.RESPIRA_BACKEND_URL ?? "http://127.0.0.1:8000";
const FRONTEND_DIR = process.env.RESPIRA_FRONTEND_DIR ?? path.join(__dirname, "..", "frontend", "dist");

function backendReachable(): Promise<boolean> {
  return new Promise((resolve) => {
    try {
      const url = new URL(`${BACKEND_URL}/api/v1/health`);
      const socket = net.connect({ host: url.hostname, port: Number(url.port) || 80 }, () => {
        socket.end();
        resolve(true);
      });
      socket.on("error", () => resolve(false));
      socket.setTimeout(2000, () => { socket.destroy(); resolve(false); });
    } catch {
      resolve(false);
    }
  });
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    title: "Respira — AI-Assisted Chest X-Ray Analysis",
    webPreferences: {
      preload: path.join(__dirname, "preload.mjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const ok = await backendReachable();
  const indexHtml = path.join(FRONTEND_DIR, "index.html");
  try {
    const fs = await import("node:fs");
    if (fs.existsSync(indexHtml)) await win.loadFile(indexHtml);
    else await win.loadURL("http://localhost:5173");
  } catch {
    await win.loadURL("http://localhost:5173");
  }
  if (!ok) {
    win.webContents.once("did-finish-load", () => {
      win.webContents.executeJavaScript(
        `console.warn("Respira backend not reachable at ${BACKEND_URL}. Start it with: python -m uvicorn backend.app.main:app")`
      ).catch(() => undefined);
    });
  }
  win.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });
}

void app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit(); // graceful shutdown
});
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) void createWindow();
});
