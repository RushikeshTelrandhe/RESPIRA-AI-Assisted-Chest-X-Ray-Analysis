export function pct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`;
}

export function uncertaintyLabel(v: number): "Low" | "Moderate" | "High" {
  if (v < 0.2) return "Low";
  if (v < 0.4) return "Moderate";
  return "High";
}

export function greeting(hour: number): string {
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}
