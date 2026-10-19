import { useId } from "react";
import { Search, X } from "lucide-react";

export function SearchInput({ value, onChange, label = "Search patients", placeholder = "Name or patient code…" }: {
  value: string; onChange: (value: string) => void; label?: string; placeholder?: string;
}) {
  const id = useId();
  return <div className="clinical-search">
    <label className="sr-only" htmlFor={id}>{label}</label>
    <input id={id} type="search" autoComplete="off" value={value} onChange={event => onChange(event.target.value)} placeholder={placeholder} />
    {value ? <button type="button" onClick={() => onChange("")} aria-label={`Clear ${label.toLowerCase()}`}><X size={17} /></button>
      : <span className="clinical-search-icon" aria-hidden="true"><Search size={18} /></span>}
  </div>;
}
