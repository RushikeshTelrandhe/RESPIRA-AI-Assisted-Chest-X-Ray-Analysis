const API_BASE: string =
  (import.meta as unknown as { env: Record<string, string> }).env.VITE_API_URL ?? "";

async function req<T>(path: string, options: RequestInit = {}, token?: string | null): Promise<T> {
  const headers: Record<string, string> = {};
  if (!(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers: { ...headers, ...(options.headers as Record<string, string> ?? {}) } });
  if (res.status === 401) {
    localStorage.removeItem("respira_token");
    if (!location.pathname.startsWith("/login")) location.href = "/login";
    throw new Error("Session expired. Please sign in again.");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      if (typeof j?.detail === "string") detail = j.detail;
    } catch { /* keep default */ }
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) return (await res.json()) as T;
  return (await res.text()) as unknown as T;
}

export const api = {
  get: <T,>(p: string, t?: string | null) => req<T>(p, {}, t),
  post: <T,>(p: string, body: unknown, t?: string | null) =>
    req<T>(p, { method: "POST", body: JSON.stringify(body) }, t),
  put: <T,>(p: string, body: unknown, t?: string | null) =>
    req<T>(p, { method: "PUT", body: JSON.stringify(body) }, t),
  del: <T,>(p: string, t?: string | null) => req<T>(p, { method: "DELETE" }, t),
  postForm: <T,>(p: string, form: FormData, t?: string | null) =>
    req<T>(p, { method: "POST", body: form }, t),
  downloadPdf: async (p: string, filename: string, t?: string | null) => {
    const headers: Record<string, string> = {};
    if (t) headers["Authorization"] = `Bearer ${t}`;
    const res = await fetch(`${API_BASE}${p}`, { headers });
    if (res.status === 401) {
      localStorage.removeItem("respira_token");
      if (!location.pathname.startsWith("/login")) location.href = "/login";
      throw new Error("Session expired. Please sign in again.");
    }
    if (!res.ok) {
      let detail = `Download failed (${res.status})`;
      try {
        const j = await res.json();
        if (typeof j?.detail === "string") detail = j.detail;
      } catch { /* body is not JSON (e.g. proxy HTML) - keep default */ }
      throw new Error(detail);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  },
};

export type Doctor = {
  id: string; full_name: string; license_no: string; email: string;
  phone: string; hospital: string; specialization: string;
  department: string; experience_years: number; theme: string;
};

export type Patient = {
  id: string; patient_code: string; full_name: string; date_of_birth: string | null;
  age: number; gender: string; phone: string; email: string; address: string;
  medical_history: string; notes: string; test_count: number; created_at: string | null;
};

export type AnalyzeResponse = {
  analysis_id: string;
  patient: { id: string; name: string };
  study: { id: string; created_at: string | null };
  prediction: { primary_class: string; confidence: number; uncertainty: number; uncertainty_level: string; margin_uncertainty: number };
  diseases: { name: string; probability: number }[];
  explainability: { gradcam: boolean; vit_attention: boolean };
  timing: { preprocessing_ms: number; inference_ms: number; explainability_ms: number; total_ms: number };
  disclaimer: string;
};

export type AnalysisDetail = {
  analysis_id: string;
  patient: { id: string; name: string };
  study: { id: string; created_at: string | null; filename: string; width: number; height: number };
  prediction: { primary_class: string; confidence: number; uncertainty: number; uncertainty_level: string; margin_uncertainty: number; uncertainties: Record<string, number>; disease_weights: Record<string, number> };
  diseases: { name: string; probability: number }[];
  timing: { preprocessing_ms: number; inference_ms: number; explainability_ms: number; total_ms: number };
  model: { version: string; device: string };
  explainability: { gradcam: boolean; vit_attention: boolean };
  created_at: string | null;
  disclaimer: string;
};

export type HistoryItem = {
  analysis_id: string; study_id: string; patient_id: string; patient_name: string;
  created_at: string | null; primary_class: string; confidence: number;
  sample_uncertainty: number; uncertainty_level: string; status: string;
};

export type ModelStatus = {
  respira_ai: string; backend: string; device: string; gpu_name: string | null;
  model_loaded: boolean; error: string | null;
  architecture: Record<string, string>; classes: string[];
  research_evaluation: Record<string, unknown>;
};

export const DISEASES = ["Atelectasis", "Bacterial Pneumonia", "Normal", "Pulmonary Edema", "Tuberculosis", "Viral Pneumonia"];
