import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
  type ReactNode,
} from "react";
import { useBeforeUnload, useBlocker } from "react-router-dom";
import { SessionCache } from "../services/SessionCache";
import { Modal } from "../components/UI";
import { errorText } from "../utils/display";

export type UploadMeta = {
  width: number;
  height: number;
  file_size: number;
  filename: string;
  preview?: string;
};
export type Draft = {
  patientId: string;
  model: string;
  file: File | null;
  meta: UploadMeta | null;
};
const initialDraft: Draft = {
  patientId: "",
  model: "fusion",
  file: null,
  meta: null,
};
type Workspace = {
  cache: SessionCache;
  draft: Draft;
  setDraft: Dispatch<SetStateAction<Draft>>;
  images: Map<string, File>;
  guard: { active: boolean; message: string };
  setGuard: Dispatch<SetStateAction<{ active: boolean; message: string }>>;
  reduced: boolean;
  setReduced: (value: boolean) => void;
  compact: boolean;
  setCompact: (value: boolean) => void;
  viewMode: "guided" | "clinical";
  setViewMode: (value: "guided" | "clinical") => void;
  presentationMode: boolean;
  setPresentationMode: (value: boolean) => void;
};
const Ctx = createContext<Workspace | null>(null);
function preference(key: string) {
  try {
    return localStorage.getItem(key) === "true";
  } catch {
    return false;
  }
}
function textPreference(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}
export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [cache] = useState(() => new SessionCache());
  const [images] = useState(() => new Map<string, File>());
  const [draft, setDraft] = useState<Draft>(initialDraft);
  const [guard, setGuard] = useState({ active: false, message: "" });
  const [reduced, setReducedState] = useState(() =>
    preference("respira_ui_reduce_motion"),
  );
  const [compact, setCompactState] = useState(() =>
    preference("respira_ui_compact"),
  );
  const [viewMode, setViewModeState] = useState<"guided" | "clinical">(() =>
    textPreference("respira_ui_view", "guided") === "clinical"
      ? "clinical"
      : "guided",
  );
  const [presentationMode, setPresentationModeState] = useState(() =>
    preference("respira_ui_presentation"),
  );
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      guard.active &&
      currentLocation.pathname + currentLocation.search !==
        nextLocation.pathname + nextLocation.search,
  );
  useBeforeUnload((event) => {
    if (guard.active || draft.file) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  useEffect(
    () => () => {
      cache.clear();
      images.clear();
    },
    [cache, images],
  );
  useEffect(() => {
    document.documentElement.dataset.reducedMotion = String(reduced);
  }, [reduced]);
  useEffect(() => {
    document.documentElement.dataset.compact = String(compact);
  }, [compact]);
  useEffect(() => {
    document.documentElement.dataset.viewMode = viewMode;
  }, [viewMode]);
  useEffect(() => {
    document.documentElement.dataset.presentation = String(presentationMode);
  }, [presentationMode]);
  const setReduced = (v: boolean) => {
    setReducedState(v);
    try {
      localStorage.setItem("respira_ui_reduce_motion", String(v));
    } catch {
      /* Optional preference storage. */
    }
  };
  const setCompact = (v: boolean) => {
    setCompactState(v);
    try {
      localStorage.setItem("respira_ui_compact", String(v));
    } catch {
      /* Optional preference storage. */
    }
  };
  const setViewMode = (v: "guided" | "clinical") => {
    setViewModeState(v);
    setCompactState(v === "clinical");
    try {
      localStorage.setItem("respira_ui_view", v);
      localStorage.setItem("respira_ui_compact", String(v === "clinical"));
    } catch {
      /* Optional preference storage. */
    }
  };
  const setPresentationMode = (v: boolean) => {
    setPresentationModeState(v);
    try {
      localStorage.setItem("respira_ui_presentation", String(v));
    } catch {
      /* Optional preference storage. */
    }
  };
  return (
    <Ctx.Provider
      value={{
        cache,
        draft,
        setDraft,
        images,
        guard,
        setGuard,
        reduced,
        setReduced,
        compact,
        setCompact,
        viewMode,
        setViewMode,
        presentationMode,
        setPresentationMode,
      }}
    >
      {children}
      <Modal
        open={blocker.state === "blocked"}
        title="Leave this page?"
        onDismiss={() => blocker.state === "blocked" && blocker.reset()}
      >
        <div className="modal-body">
          <p>{guard.message || "Your unsaved changes will be lost."}</p>
          <div className="form-actions">
            <button
              className="btn-ghost"
              onClick={() => blocker.state === "blocked" && blocker.reset()}
            >
              Stay here
            </button>
            <button
              className="btn-primary"
              onClick={() => blocker.state === "blocked" && blocker.proceed()}
            >
              Leave page
            </button>
          </div>
        </div>
      </Modal>
    </Ctx.Provider>
  );
}
export function useWorkspace() {
  const c = useContext(Ctx);
  if (!c) throw new Error("Workspace provider is missing");
  return c;
}
export function useUnsavedChanges(
  active: boolean,
  message = "Your unsaved changes will be lost.",
) {
  const { setGuard } = useWorkspace();
  useEffect(() => {
    setGuard({ active, message });
    return () => setGuard({ active: false, message: "" });
  }, [active, message, setGuard]);
}
export function useResource<T>(key: string | null, loader: () => Promise<T>) {
  const { cache } = useWorkspace();
  const loadRef = useRef(loader);
  loadRef.current = loader;
  const [revision, setRevision] = useState(0);
  const [state, setState] = useState<{
    key: string | null;
    data?: T;
    error: string;
    loading: boolean;
  }>({ key: null, error: "", loading: false });
  useEffect(() => {
    if (!key) return;
    let active = true;
    const cached = cache.peek<T>(key);
    setState({ key, data: cached, error: "", loading: cached === undefined });
    void cache
      .get<T>(key, () => loadRef.current())
      .then((data) => {
        if (active) setState({ key, data, error: "", loading: false });
      })
      .catch((error) => {
        if (active) setState({ key, error: errorText(error), loading: false });
      });
    return () => {
      active = false;
    };
  }, [key, revision, cache]);
  const current =
    key === null
      ? { data: undefined, loading: false, error: "" }
      : state.key === key
        ? state
        : {
            data: cache.peek<T>(key),
            loading: cache.peek<T>(key) === undefined,
            error: "",
          };
  return {
    ...current,
    reload: () => {
      if (key) {
        cache.invalidate(key);
        setRevision((v) => v + 1);
      }
    },
  };
}
export function usePreview(file: File | null | undefined) {
  const [value, setValue] = useState<{ file: File; url: string } | null>(null);
  useEffect(() => {
    if (!file) {
      setValue(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setValue({ file, url });
    return () => URL.revokeObjectURL(url);
  }, [file]);
  return value && value.file === file ? value.url : null;
}
