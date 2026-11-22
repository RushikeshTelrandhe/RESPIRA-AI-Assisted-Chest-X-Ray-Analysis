import { Info } from "lucide-react";
import { score, numeric } from "../utils/display";
import { Empty } from "./UI";
export function DiseaseBars({
  diseases,
  primary,
}: {
  diseases: { name: string; probability: number }[];
  primary?: string;
}) {
  // Sort descending by probability to establish ranked differential diagnosis
  const sorted = [...diseases].sort(
    (a, b) => (b.probability || 0) - (a.probability || 0),
  );

  return (
    <div className="score-list" role="list" aria-label="Ranked model differential diagnosis">
      {sorted.map((d, index) => {
        const rank = index + 1;
        const valid =
          Number.isFinite(d.probability) &&
          d.probability >= 0 &&
          d.probability <= 1;
        const isPrimary = rank === 1 || primary === d.name;
        const isDifferential = rank === 2;

        return (
          <div role="listitem" key={d.name} className="score-item">
            <div className="score-label">
              <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
                <span
                  style={{
                    display: "inline-block",
                    minWidth: "22px",
                    textAlign: "center",
                    fontWeight: 700,
                    fontSize: "0.74rem",
                    padding: "1px 5px",
                    borderRadius: "4px",
                    background: isPrimary ? "#0f766e" : isDifferential ? "#e0f2fe" : "#f1f5f9",
                    color: isPrimary ? "#ffffff" : isDifferential ? "#0369a1" : "#64748b",
                  }}
                >
                  #{rank}
                </span>
                <span style={{ fontWeight: isPrimary ? 600 : isDifferential ? 500 : 400 }}>
                  {d.name}
                </span>
                {isPrimary ? (
                  <span style={{ fontSize: "0.72rem", color: "#0f766e", fontWeight: 600 }}>
                    · Primary
                  </span>
                ) : isDifferential ? (
                  <span style={{ fontSize: "0.72rem", color: "#0284c7", fontWeight: 600 }}>
                    · Differential
                  </span>
                ) : null}
              </span>
              <strong>{score(d.probability)}</strong>
            </div>
            <div className="score-track" aria-hidden="true">
              <div
                className={
                  "score-fill " + (isPrimary ? "primary" : isDifferential ? "differential" : "")
                }
                style={{
                  width: valid ? d.probability * 100 + "%" : "0%",
                  background: isPrimary
                    ? "#0f766e"
                    : isDifferential
                    ? "#0284c7"
                    : "#94a3b8",
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
export function UncertaintyBadge({ level }: { level?: string }) {
  const normalized = (level ?? "").toLowerCase();
  const known = ["low", "moderate", "high"].includes(normalized);
  return (
    <span className={"badge " + (known ? normalized : "")}>
      {known
        ? normalized.charAt(0).toUpperCase() +
          normalized.slice(1) +
          " uncertainty"
        : "Uncertainty unavailable"}
    </span>
  );
}
export function UncertaintyGauge({ value }: { value: number }) {
  return <span className="mono">{numeric(value)}</span>;
}
export function Disclaimer() {
  return (
    <p className="research-note">
      <Info size={16} aria-hidden="true" />
      <span>
        Research prototype. AI outputs require review by a qualified clinician
        and are not a definitive diagnosis.
      </span>
    </p>
  );
}
export const EmptyState = Empty;
