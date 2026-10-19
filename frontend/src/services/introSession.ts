export const INTRO_SESSION_KEY = "respira_intro_seen_clinical_v4";

export type SessionStore = Pick<Storage, "getItem" | "setItem" | "removeItem">;

function browserStore(): SessionStore | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function hasSeenIntro(store: SessionStore | null = browserStore()) {
  try {
    return store?.getItem(INTRO_SESSION_KEY) === "1";
  } catch {
    return false;
  }
}

export function markIntroSeen(store: SessionStore | null = browserStore()) {
  try {
    store?.setItem(INTRO_SESSION_KEY, "1");
  } catch {
    // The intro is optional when browser storage is unavailable.
  }
}

export function resetIntroSession(store: SessionStore | null = browserStore()) {
  try {
    store?.removeItem(INTRO_SESSION_KEY);
  } catch {
    // The replay control remains safe when browser storage is unavailable.
  }
}
