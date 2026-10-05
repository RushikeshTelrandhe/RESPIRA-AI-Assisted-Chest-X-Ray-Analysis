import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Eye, Info, Layers3 } from "lucide-react";
import { api, DISEASES, type AnalysisDetail } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useResource, useWorkspace } from "../context/WorkspaceContext";
import { Alert, LoadingBlock, PageHeader } from "../components/UI";
import { Disclaimer } from "../components/widgets";
import { XrayViewer, type ViewerMode } from "../components/XrayViewer";
import { errorText } from "../utils/display";
import { explanationImages, explanationKey, type ExplanationResult as Gx } from "../services/explanationImages";
type Method = "gradcam" | "vit";
type ResultState = { key: string; data?: Gx; loading: boolean; error: string };
export function Explainability() {
  const { id = "" } = useParams();
  const { token } = useAuth();
  const { cache } = useWorkspace();
  const analysis = useResource<AnalysisDetail>(
    id ? "analysis:" + id : null,
    () => api.get("/api/v1/analysis/" + id, token),
  );
  const [method, setMethod] = useState<Method>("gradcam");
  const [disease, setDisease] = useState(0);
  const [mode, setMode] = useState<ViewerMode>("overlay");
  const [opacity, setOpacity] = useState(1);
  const requestKey = explanationKey(id, analysis.data?.model.version || "", method, disease);
  const current = useRef(requestKey);
  const [state, setState] = useState<ResultState>({
    key: requestKey,
    loading: false,
    error: "",
  });
  useEffect(() => {
    if (analysis.data) {
      const found = DISEASES.indexOf(analysis.data.prediction.primary_class);
      if (found >= 0) setDisease(found);
    }
  }, [analysis.data]);
  useEffect(() => {
    setOpacity(1);
    setMode("overlay");
    current.current = requestKey;
    setState({
      key: requestKey,
      data: cache.peek<Gx>(requestKey),
      loading: false,
      error: "",
    });
  }, [requestKey, cache]);
  const supported =
    method === "gradcam"
      ? analysis.data?.explainability.gradcam
      : analysis.data?.explainability.vit_attention;
  async function load(force = false) {
    if (force) cache.invalidate(requestKey);
    const key = requestKey;
    current.current = key;
    setState({ key, data: cache.peek<Gx>(key), loading: true, error: "" });
    try {
      const data = await cache.get<Gx>(key, async () => {
        const form = new FormData();
        form.append("analysis_id", id);
        if (method === "gradcam") form.append("target_class", String(disease));
        return api.postForm<Gx>(
          method === "gradcam"
            ? "/api/v1/explainability/gradcam"
            : "/api/v1/explainability/vit-attention",
          form,
          token,
        );
      });
      if (current.current === key)
        setState({ key, data, loading: false, error: "" });
    } catch (e) {
      cache.invalidate(key);
      if (current.current === key)
        setState({ key, loading: false, error: errorText(e) });
    }
  }
  const shown =
    state.key === requestKey
      ? state
      : {
          key: requestKey,
          data: cache.peek<Gx>(requestKey),
          loading: false,
          error: "",
        };
  if (analysis.loading)
    return <LoadingBlock label="Loading explanation workspace…" />;
  if (analysis.error || !analysis.data)
    return (
      <Alert retry={analysis.reload}>
        {analysis.error || "Analysis details are unavailable."}
      </Alert>
    );
  const a = analysis.data;
  const media = shown.data ? explanationImages(shown.data) : null;
  return (
    <div className="stack explainability-page">
      <Link className="text-button" to={"/analysis/" + id}>
        <ArrowLeft size={16} />
        Back to summary
      </Link>
      <PageHeader
        eyebrow="Visual explanation"
        title="What the model highlighted"
        description={
          a.patient.name +
          " · " +
          a.prediction.primary_class +
          " primary model output"
        }
      />
      <div className="explanation-controls">
        <div className="explanation-control-intro">
          <span className="explanation-control-icon"><Layers3 size={19} /></span>
          <span><strong>Choose the evidence view</strong><small>Requests run only when you select Generate.</small></span>
        </div>
        <div>
          <span className="label">Explanation method</span>
          <div className="page-actions">
            <button
              className={method === "gradcam" ? "btn-primary" : "btn-ghost"}
              onClick={() => setMethod("gradcam")}
            >
              <Layers3 size={17} />
              Grad-CAM
            </button>
            <button
              className={method === "vit" ? "btn-primary" : "btn-ghost"}
              onClick={() => setMethod("vit")}
            >
              <Eye size={17} />
              ViT attention
            </button>
          </div>
        </div>
        {method === "gradcam" && (
          <div>
            <label className="label" htmlFor="explain-class">
              Target class
            </label>
            <select
              id="explain-class"
              className="input"
              value={disease}
              onChange={(e) => setDisease(Number(e.target.value))}
            >
              {DISEASES.map((d, i) => (
                <option value={i} key={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>
        )}
        <div>
          <span className="label">Request</span>
          <button
            className="btn-primary"
            disabled={!supported || shown.loading}
            onClick={() => void load(Boolean(shown.data))}
          >
            {shown.loading ? (
              <>
                <span className="spin" />
                Generating explanation…
              </>
            ) : shown.data ? (
              "Regenerate explanation"
            ) : (
              "Generate explanation"
            )}
          </button>
        </div>
      </div>
      {!supported && (
        <Alert kind="warning">
          This explanation method is not reported as available for this
          analysis.
        </Alert>
      )}
      {shown.error && <Alert retry={() => void load()}>{shown.error}</Alert>}
      {!shown.data && !shown.loading && supported && (
        <div className="viewer">
          <div className="viewer-empty">
            <Layers3 size={34} />
            <strong>No explanation requested yet</strong>
            <span>
              Generate only the method and target class you want to review.
            </span>
            <button className="btn-light" onClick={() => void load(Boolean(shown.data))}>
              Generate {method === "gradcam" ? "Grad-CAM" : "ViT attention"}
            </button>
          </div>
        </div>
      )}
      {shown.loading && !shown.data && (
        <LoadingBlock
          label={
            method === "gradcam"
              ? "Generating Grad-CAM…"
              : "Generating ViT attention…"
          }
        />
      )}
      {media?.original && (media.overlay || media.heatmap) && (
        <XrayViewer key={requestKey} original={media.original} overlay={media.overlay} map={media.heatmap}
          mapLabel={method === "gradcam" ? "Grad-CAM · " + DISEASES[disease] : "ViT attention"}
          mode={mode} setMode={setMode} opacity={opacity} setOpacity={setOpacity} />
      )}
      {shown.data && (!media?.original || !(media.overlay || media.heatmap)) && (
        <Alert kind="warning">This response is missing a usable study image or explanation. Please regenerate this method.</Alert>
      )}
      <div className="two-col">
        <section className="explanation-card">
          <Info size={21} />
          <div>
            <p className="eyebrow">
              {method === "gradcam"
                ? "CNN attribution"
                : "Transformer attention"}
            </p>
            <h2>
              {method === "gradcam"
                ? "About this Grad-CAM map"
                : "About this ViT attention map"}
            </h2>
            <p>
              {method === "gradcam"
                ? "This map highlights image regions associated with the selected CNN class output. It does not prove that a lesion is present or explain every part of the fusion model."
                : "This map visualizes attention within the ViT representation. Attention is not a verified lesion outline and does not by itself establish the reason for the final fusion output."}
            </p>
          </div>
        </section>
        <section className="panel">
          <p className="eyebrow">Shown labels</p>
          <dl className="detail-list">
            <div>
              <dt>Analysis</dt>
              <dd className="mono">{id.slice(0, 8)}</dd>
            </div>
            <div>
              <dt>Method</dt>
              <dd>{method === "gradcam" ? "Grad-CAM" : "ViT attention"}</dd>
            </div>
            <div>
              <dt>Target class</dt>
              <dd>
                {method === "gradcam"
                  ? DISEASES[disease]
                  : "Not class-selectable"}
              </dd>
            </div>
            <div>
              <dt>Model</dt>
              <dd>{a.model.version || "Unavailable"}</dd>
            </div>
          </dl>
        </section>
      </div>
      <Disclaimer />
    </div>
  );
}
