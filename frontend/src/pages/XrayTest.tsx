import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { UploadCloud, X } from "lucide-react";
import { api, type AnalyzeResponse, type Patient } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { Disclaimer } from "../components/widgets";

const STAGES = ["Preparing image", "Running AI model", "Calculating predictions", "Generating explanation"];

export function XrayTest() {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientId, setPatientId] = useState(params.get("patient") ?? "");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ width: number; height: number; file_size: number; filename: string; preview?: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    api.get<Patient[]>("/api/v1/patients", token).then(setPatients).catch(() => undefined);
  }, [token]);
  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onFile = async (f: File | undefined) => {
    setError(""); setMeta(null);
    if (!f) return;
    setFile(f);
    const form = new FormData();
    form.append("file", f);
    try {
      const r = await api.postForm<{ width: number; height: number; file_size: number; filename: string; preview: string }>("/api/v1/xray/upload", form, token);
      setMeta(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Invalid file");
      setFile(null);
    }
  };

  const analyze = async () => {
    if (!patientId) return setError("Select a patient first.");
    if (!file) return setError("Upload a chest X-ray first.");
    setError(""); setBusy(true); setStage(0);
    const tick = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 1200);
    try {
      const form = new FormData();
      form.append("patient_id", patientId);
      form.append("file", file);
      const r = await api.postForm<AnalyzeResponse>("/api/v1/analyze", form, token);
      clearInterval(tick);
      navigate(`/analysis/${r.analysis_id}`);
    } catch (e) {
      clearInterval(tick);
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally { setBusy(false); }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <h1 className="text-2xl font-bold">New X-Ray Test</h1>
      {error && <div role="alert" className="card border-rose-200 p-3 text-sm text-rose-700">{error}</div>}

      <div className="card space-y-2 p-5">
        <h2 className="font-semibold">Step 1 — Select patient</h2>
        <select className="input" value={patientId} onChange={(e) => setPatientId(e.target.value)} aria-label="Select patient">
          <option value="">Choose a patient…</option>
          {patients.map((p) => <option key={p.id} value={p.id}>{p.full_name} ({p.patient_code || p.id.slice(0, 8)})</option>)}
        </select>
      </div>

      <div className="card space-y-3 p-5">
        <h2 className="font-semibold">Step 2 — Upload X-ray (PNG / JPG)</h2>
        <label
          className="grid cursor-pointer place-items-center gap-2 rounded-xl border-2 border-dashed border-slate-300 p-8 text-center hover:border-brand-500"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); void onFile(e.dataTransfer.files?.[0]); }}
        >
          <UploadCloud className="text-brand-600" size={28} />
          <span className="text-sm text-slate-600">Drag & drop or click to browse</span>
          <input type="file" className="hidden" accept=".png,.jpg,.jpeg" onChange={(e) => void onFile(e.target.files?.[0])} />
        </label>
        {preview && (
          <div className="flex gap-4">
            <img src={meta?.preview ?? preview} alt="X-ray preview" className="h-48 w-48 rounded-lg border object-contain bg-black" />
            <div className="text-sm text-slate-600">
              <div><b>File:</b> {meta?.filename ?? file?.name}</div>
              <div><b>Dimensions:</b> {meta ? `${meta.width} × ${meta.height}` : "—"}</div>
              <div><b>Size:</b> {meta ? `${(meta.file_size / 1024).toFixed(0)} KB` : file ? `${(file.size / 1024).toFixed(0)} KB` : "—"}</div>
              <button className="btn-ghost mt-2 inline-flex items-center gap-1 !py-1.5 text-sm" onClick={() => { setFile(null); setMeta(null); }}><X size={14} /> Remove</button>
            </div>
          </div>
        )}
      </div>

      <div className="card space-y-3 p-5">
        <h2 className="font-semibold">Step 3 — Start analysis</h2>
        {!busy ? (
          <button className="btn-primary" onClick={() => void analyze()} disabled={!file || !patientId}>Analyze with Respira AI</button>
        ) : (
          <div className="space-y-2" role="status" aria-live="polite">
            <div className="font-medium">Analyzing X-ray…</div>
            <ul className="space-y-1 text-sm text-slate-600">
              {STAGES.map((s, i) => (
                <li key={s} className={i <= stage ? "text-brand-700 font-medium" : ""}>{i < stage ? "✓" : i === stage ? "…" : "○"} {s}</li>
              ))}
            </ul>
          </div>
        )}
        <Disclaimer />
      </div>
    </div>
  );
}
