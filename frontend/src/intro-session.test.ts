import { describe, expect, it } from "vitest";
import {
  hasSeenIntro,
  markIntroSeen,
  resetIntroSession,
  type SessionStore,
} from "./services/introSession";

function memoryStore(): SessionStore {
  const values = new Map<string, string>();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

describe("cinematic intro session gate", () => {
  it("marks the introduction as seen for the current session", () => {
    const store = memoryStore();
    expect(hasSeenIntro(store)).toBe(false);
    markIntroSeen(store);
    expect(hasSeenIntro(store)).toBe(true);
  });

  it("allows the introduction to be replayed", () => {
    const store = memoryStore();
    markIntroSeen(store);
    resetIntroSession(store);
    expect(hasSeenIntro(store)).toBe(false);
  });

  it("fails safely when session storage is blocked", () => {
    const blocked: SessionStore = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
      removeItem: () => {
        throw new Error("blocked");
      },
    };
    expect(hasSeenIntro(blocked)).toBe(false);
    expect(() => markIntroSeen(blocked)).not.toThrow();
    expect(() => resetIntroSession(blocked)).not.toThrow();
  });
});
