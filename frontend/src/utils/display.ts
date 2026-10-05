export function score(value: unknown, digits = 1): string {
  return typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 1
    ? (value * 100).toFixed(digits) + "%"
    : "Unavailable";
}
export function numeric(value: unknown, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "Unavailable";
}
export function dateLabel(value?: string | null, withTime = false): string {
  if (!value) return "Date unavailable";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "Date unavailable";
  return new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    ...(withTime ? ({ hour: "2-digit", minute: "2-digit" } as const) : {}),
  }).format(d);
}
export function initials(name?: string): string {
  return (
    (name ?? "")
      .replace(/^dr\.?\s+/i, "")
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((s) => s[0])
      .join("")
      .toUpperCase() || "DR"
  );
}
export function errorText(error: unknown): string {
  const s =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : "";
  if (/failed to fetch|networkerror|network request|502|503|504|ECONN/i.test(s))
    return "We couldn't reach the service. Keep the backend running, then try again.";
  if (
    /traceback|FileNotFoundError|site-packages|[A-Z]:\\|\/home\/|\/workspace\/|checkpoint missing/i.test(
      s,
    )
  )
    return "The service couldn't complete this request. Check model status or contact your project administrator.";
  return s && s.length <= 260 ? s : "Something went wrong. Please try again.";
}
export function safeReturnPath(value: unknown): string {
  return typeof value === "string" &&
    value.startsWith("/") &&
    !value.startsWith("//") &&
    !value.startsWith("/login") &&
    !value.startsWith("/signup")
    ? value
    : "/dashboard";
}
