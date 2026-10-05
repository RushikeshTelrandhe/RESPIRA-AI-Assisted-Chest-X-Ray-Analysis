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
  return (
    <div className="score-list" role="list" aria-label="Model class scores">
      {diseases.map((d) => {
        const valid =
          Number.isFinite(d.probability) &&
          d.probability >= 0 &&
          d.probability <= 1;
        return (
          <div role="listitem" key={d.name}>
            <div className="score-label">
              <span>
                {d.name}
                {primary === d.name ? " · primary" : ""}
              </span>
              <strong>{score(d.probability)}</strong>
            </div>
            <div className="score-track" aria-hidden="true">
              <div
                className={
                  "score-fill " + (primary === d.name ? "primary" : "")
                }
                style={{ width: valid ? d.probability * 100 + "%" : "0%" }}
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
