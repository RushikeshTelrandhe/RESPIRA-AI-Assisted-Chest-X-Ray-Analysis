import { Link } from "react-router-dom";
export function Brand({
  light = false,
  to = "/",
  compact = false,
}: {
  light?: boolean;
  to?: string;
  compact?: boolean;
}) {
  return (
    <Link
      to={to}
      className={"brand" + (light ? " brand-light" : "")}
      aria-label="RESPIRA home"
    >
      <span className="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 40 40" fill="none">
          <path
            d="M20 7v12m0-6-5 5m5-5 5 5M15 12c-3 1-9 8-9 15 0 5 4 7 9 5 3-1 3-5 3-9V11m7 1c3 1 9 8 9 15 0 5-4 7-9 5-3-1-3-5-3-9V11"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
      {!compact && (
        <span>
          <strong>RESPIRA</strong>
          <small>CLINICAL INTELLIGENCE</small>
        </span>
      )}
    </Link>
  );
}
