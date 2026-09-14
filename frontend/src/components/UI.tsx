import {
  useEffect,
  useRef,
  useState,
  type ReactNode,
  type InputHTMLAttributes,
} from "react";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Eye,
  EyeOff,
  Info,
  LoaderCircle,
  X,
} from "lucide-react";

export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1 tabIndex={-1}>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {action && <div className="page-actions">{action}</div>}
    </header>
  );
}
export function Alert({
  children,
  kind = "error",
  retry,
}: {
  children: ReactNode;
  kind?: "error" | "info" | "success" | "warning";
  retry?: () => void;
}) {
  const Icon =
    kind === "success" ? CheckCircle2 : kind === "error" ? AlertCircle : Info;
  return (
    <div
      className={"notice notice-" + kind}
      role={kind === "error" ? "alert" : "status"}
    >
      <Icon size={18} aria-hidden="true" />
      <div>{children}</div>
      {retry && (
        <button className="text-button" onClick={retry}>
          Try again
        </button>
      )}
    </div>
  );
}
export function LoadingBlock({
  label = "Loading your workspace…",
}: {
  label?: string;
}) {
  return (
    <div className="loading-block" role="status">
      <LoaderCircle className="spin" size={22} aria-hidden="true" />
      <span>{label}</span>
      <div className="skeleton-lines" aria-hidden="true">
        <i />
        <i />
        <i />
      </div>
    </div>
  );
}
export function Empty({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <span className="empty-icon">
        <Info size={22} aria-hidden="true" />
      </span>
      <h3>{title}</h3>
      {hint && <p>{hint}</p>}
      {action}
    </div>
  );
}
export function Modal({
  open,
  title,
  children,
  onDismiss,
  className = "",
}: {
  open: boolean;
  title: string;
  children: ReactNode;
  onDismiss: () => void;
  className?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const previous = document.activeElement as HTMLElement | null;
    if (open && !el.open) el.showModal();
    if (!open && el.open) el.close();
    return () => {
      if (el.open) el.close();
      if (previous?.isConnected) previous.focus();
    };
  }, [open]);
  return (
    <dialog
      ref={ref}
      className={"modal " + className}
      aria-label={title}
      onCancel={(e) => {
        e.preventDefault();
        onDismiss();
      }}
    >
      <div className="modal-header">
        <h2>{title}</h2>
        <button
          className="icon-button"
          type="button"
          aria-label={"Close " + title}
          onClick={onDismiss}
        >
          <X size={20} />
        </button>
      </div>
      {open && children}
    </dialog>
  );
}
export function Tabs<T extends string>({
  id,
  options,
  value,
  onChange,
}: {
  id: string;
  options: { value: T; label: string; disabled?: boolean }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="tabs" role="tablist" aria-label={id.replace(/-/g, " ")}>
      {options.map((o, i) => (
        <button
          key={o.value}
          id={id + "-" + o.value}
          role="tab"
          aria-selected={value === o.value}
          aria-controls={id + "-panel"}
          tabIndex={value === o.value ? 0 : -1}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
          onKeyDown={(e) => {
            if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(e.key))
              return;
            e.preventDefault();
            const enabled = options
              .map((v, index) => ({ ...v, index }))
              .filter((v) => !v.disabled);
            const current = enabled.findIndex((v) => v.index === i);
            const next =
              e.key === "Home"
                ? enabled[0]
                : e.key === "End"
                  ? enabled[enabled.length - 1]
                  : enabled[
                      (current +
                        (e.key === "ArrowRight" ? 1 : -1) +
                        enabled.length) %
                        enabled.length
                    ];
            if (next) {
              onChange(next.value);
              document.getElementById(id + "-" + next.value)?.focus();
            }
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
export function PasswordInput({
  id,
  label,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & { id: string; label: string }) {
  const [visible, setVisible] = useState(false);
  return (
    <div>
      <label className="label" htmlFor={id}>
        {label}
      </label>
      <div className="password-field">
        <input
          {...props}
          id={id}
          className="input"
          type={visible ? "text" : "password"}
        />
        <button
          type="button"
          className="icon-button"
          aria-label={visible ? "Hide password" : "Show password"}
          aria-pressed={visible}
          onClick={() => setVisible((v) => !v)}
        >
          {visible ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>
      </div>
    </div>
  );
}
export function SubmitButton({
  busy,
  children,
  busyText = "Saving…",
  disabled = false,
}: {
  busy: boolean;
  children: ReactNode;
  busyText?: string;
  disabled?: boolean;
}) {
  return (
    <button className="btn-primary" type="submit" disabled={busy || disabled}>
      {busy ? (
        <>
          <LoaderCircle size={17} className="spin" />
          {busyText}
        </>
      ) : (
        <>
          {children}
          <ArrowRight size={17} aria-hidden="true" />
        </>
      )}
    </button>
  );
}
