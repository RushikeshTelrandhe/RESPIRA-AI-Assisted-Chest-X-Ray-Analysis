import { useState } from "react";
import { api, type Doctor } from "../services/api";
import { useAuth } from "../context/AuthContext";

export function Profile() {
  const { doctor, token, refresh } = useAuth();
  const [tab, setTab] = useState("personal");
  const [msg, setMsg] = useState("");
  const [form, setForm] = useState({ full_name: doctor?.full_name ?? "", phone: doctor?.phone ?? "", hospital: doctor?.hospital ?? "", specialization: doctor?.specialization ?? "", department: doctor?.department ?? "", experience_years: String(doctor?.experience_years ?? 0) });
  const [pw, setPw] = useState({ current_password: "", new_password: "" });

  if (!doctor) return null;
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value });

  const save = async () => {
    setMsg("");
    try {
      await api.put<Doctor>("/api/v1/doctors/profile", { ...form, experience_years: Number(form.experience_years) || 0 }, token);
      await refresh(); setMsg("Profile updated.");
    } catch (e) { setMsg(e instanceof Error ? e.message : "Update failed"); }
  };
  const changePw = async () => {
    setMsg("");
    try {
      await api.post("/api/v1/doctors/change-password", pw, token);
      setPw({ current_password: "", new_password: "" }); setMsg("Password changed.");
    } catch (e) { setMsg(e instanceof Error ? e.message : "Password change failed"); }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <h1 className="text-2xl font-bold">Profile</h1>
      <div className="flex gap-1" role="tablist">
        {(["personal", "professional", "security"] as const).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
            className={`rounded-lg px-4 py-2 text-sm font-medium capitalize ${tab === t ? "bg-brand-600 text-white" : "bg-white border border-slate-200 text-slate-600"}`}>{t}</button>
        ))}
      </div>
      {msg && <div className="card p-3 text-sm">{msg}</div>}
      {tab === "personal" && (
        <div className="card grid gap-3 p-5 sm:grid-cols-2">
          <div><label className="label">Name</label><input className="input" value={form.full_name} onChange={set("full_name")} /></div>
          <div><label className="label">Email</label><input className="input" value={doctor.email} disabled /></div>
          <div><label className="label">Phone</label><input className="input" value={form.phone} onChange={set("phone")} /></div>
          <div><label className="label">Specialization</label><input className="input" value={form.specialization} onChange={set("specialization")} /></div>
          <div className="sm:col-span-2"><button className="btn-primary" onClick={() => void save()}>Save</button></div>
        </div>
      )}
      {tab === "professional" && (
        <div className="card grid gap-3 p-5 sm:grid-cols-2">
          <div><label className="label">Medical registration number</label><input className="input" value={doctor.license_no} disabled /></div>
          <div><label className="label">Hospital / clinic</label><input className="input" value={form.hospital} onChange={set("hospital")} /></div>
          <div><label className="label">Department</label><input className="input" value={form.department} onChange={set("department")} /></div>
          <div><label className="label">Years of experience</label><input className="input" type="number" min={0} max={80} value={form.experience_years} onChange={set("experience_years")} /></div>
          <div className="sm:col-span-2"><button className="btn-primary" onClick={() => void save()}>Save</button></div>
        </div>
      )}
      {tab === "security" && (
        <div className="card grid gap-3 p-5">
          <div><label className="label" htmlFor="cpw">Current password</label><input id="cpw" type="password" className="input" value={pw.current_password} onChange={(e) => setPw({ ...pw, current_password: e.target.value })} /></div>
          <div><label className="label" htmlFor="npw">New password (8+ characters)</label><input id="npw" type="password" className="input" value={pw.new_password} onChange={(e) => setPw({ ...pw, new_password: e.target.value })} /></div>
          <div><button className="btn-primary" onClick={() => void changePw()}>Change password</button></div>
        </div>
      )}
    </div>
  );
}
