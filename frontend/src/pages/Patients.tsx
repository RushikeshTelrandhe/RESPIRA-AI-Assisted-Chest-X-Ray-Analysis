import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Patient } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { EmptyState } from "../components/widgets";

export function Patients() {
  const { token } = useAuth();
  const [items, setItems] = useState<Patient[]>([]);
  const [q, setQ] = useState("");
  const [error, setError] = useState("");
  const [show, setShow] = useState(false);
  const [form, setForm] = useState({ full_name: "", patient_code: "", age: "", gender: "", phone: "", medical_history: "", notes: "" });

  const load = async (query = "") => {
    try { setItems(await api.get<Patient[]>(`/api/v1/patients?q=${encodeURIComponent(query)}`, token)); }
    catch (e) { setError(e instanceof Error ? e.message : "Failed to load"); }
  };
  useEffect(() => { void load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const create = async (e: React.FormEvent) => {
    e.preventDefault(); setError("");
    try {
      await api.post("/api/v1/patients", { ...form, age: form.age ? Number(form.age) : 0 }, token);
      setShow(false); setForm({ full_name: "", patient_code: "", age: "", gender: "", phone: "", medical_history: "", notes: "" });
      await load(q);
    } catch (e2) { setError(e2 instanceof Error ? e2.message : "Create failed"); }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold">Patients</h1>
        <button className="btn-primary" onClick={() => setShow(!show)}>{show ? "Close" : "Add Patient"}</button>
      </div>
      <div className="flex gap-2">
        <input className="input max-w-sm" placeholder="Search name, ID, or phone…" value={q} onChange={(e) => { setQ(e.target.value); void load(e.target.value); }} aria-label="Search patients" />
      </div>
      {error && <div role="alert" className="card border-rose-200 p-3 text-sm text-rose-700">{error}</div>}
      {show && (
        <form onSubmit={create} className="card grid gap-3 p-5 sm:grid-cols-2">
          <div><label className="label">Full name *</label><input className="input" required value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} /></div>
          <div><label className="label">Patient ID</label><input className="input" value={form.patient_code} onChange={(e) => setForm({ ...form, patient_code: e.target.value })} /></div>
          <div><label className="label">Age</label><input className="input" type="number" min={0} max={150} value={form.age} onChange={(e) => setForm({ ...form, age: e.target.value })} /></div>
          <div><label className="label">Gender</label>
            <select className="input" value={form.gender} onChange={(e) => setForm({ ...form, gender: e.target.value })}><option value="">—</option><option>Male</option><option>Female</option><option>Other</option></select></div>
          <div><label className="label">Phone</label><input className="input" value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} /></div>
          <div><label className="label">Medical history</label><input className="input" value={form.medical_history} onChange={(e) => setForm({ ...form, medical_history: e.target.value })} /></div>
          <div className="sm:col-span-2"><label className="label">Notes</label><input className="input" value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} /></div>
          <div className="sm:col-span-2"><button className="btn-primary">Save patient</button></div>
        </form>
      )}
      {items.length === 0 ? <EmptyState title="No patients yet" hint="Add your first patient to begin." /> : (
        <ul className="grid gap-3 sm:grid-cols-2">
          {items.map((p) => (
            <li key={p.id} className="card p-4">
              <Link to={`/patients/${p.id}`} className="font-semibold text-brand-700 hover:underline">{p.full_name}</Link>
              <div className="mt-1 text-sm text-slate-500">{p.patient_code || p.id.slice(0, 8)} · {p.age ? `${p.age} y` : "—"} · {p.gender || "—"} · {p.test_count} test{p.test_count === 1 ? "" : "s"}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
