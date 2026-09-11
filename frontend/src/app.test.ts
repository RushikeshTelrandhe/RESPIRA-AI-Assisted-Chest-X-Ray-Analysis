import { describe, expect, it } from "vitest";
import { pct, uncertaintyLabel, greeting } from "./utils/format";
import { DISEASES } from "./services/api";

describe("format helpers", () => {
  it("formats probabilities as percent", () => {
    expect(pct(0.924)).toBe("92.4%");
  });
  it("maps uncertainty thresholds like the backend", () => {
    expect(uncertaintyLabel(0.05)).toBe("Low");
    expect(uncertaintyLabel(0.3)).toBe("Moderate");
    expect(uncertaintyLabel(0.9)).toBe("High");
  });
  it("greets by hour", () => {
    expect(greeting(9)).toBe("Good morning");
    expect(greeting(15)).toBe("Good afternoon");
    expect(greeting(21)).toBe("Good evening");
  });
  it("keeps canonical disease order", () => {
    expect(DISEASES).toEqual(["Atelectasis", "Bacterial Pneumonia", "Normal", "Pulmonary Edema", "Tuberculosis", "Viral Pneumonia"]);
  });
});
