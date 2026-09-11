export function DiseaseBars({ diseases }: { diseases: { name: string; probability: number }[] }) {
  const sorted = [...diseases].sort((a, b) => b.probability - a.probability);
  return (
    <div className="flex flex-col gap-3" role="list" aria-label="Disease probabilities">
      {sorted.map((d) => (
        <div key={d.name} role="listitem">
          <div className="mb-1 flex items-center justify-between text-sm">
            <span className="font-medium">{d.name}</span>
            <span className="tabular-nums text-slate-600">{(d.probability * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-slate-100" role="progressbar" aria-valuenow={Math.round(d.probability * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={d.name}>
            <div className="h-full rounded-full bg-brand-600 transition-all" style={{ width: `${(d.probability * 100).toFixed(1)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export function UncertaintyBadge({ level }: { level: string }) {
  const l = level.toLowerCase();
  const cls = l === "low" ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : l === "moderate" ? "bg-amber-50 text-amber-700 border-amber-200"
    : "bg-rose-50 text-rose-700 border-rose-200";
  return <span className={`inline-block rounded-full border px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide ${cls}`}>{level}</span>;
}

export function UncertaintyGauge({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const color = value < 0.2 ? "#059669" : value < 0.4 ? "#d97706" : "#e11d48";
  return (
    <div className="flex items-center gap-3">
      <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-100" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label="Model uncertainty">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-sm font-semibold tabular-nums">{value.toFixed(3)}</span>
    </div>
  );
}

export function Disclaimer() {
  return (
    <p className="rounded-lg bg-slate-50 border border-slate-200 p-3 text-xs leading-relaxed text-slate-600">
      Respira provides AI-assisted analysis for research and clinical decision support. Results should be
      reviewed by a qualified medical professional and should not be considered a definitive diagnosis.
    </p>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="card grid place-items-center gap-1 p-10 text-center">
      <div className="font-semibold">{title}</div>
      {hint && <div className="text-sm text-slate-500">{hint}</div>}
    </div>
  );
}
