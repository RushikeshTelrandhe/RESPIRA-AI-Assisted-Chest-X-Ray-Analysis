import { useEffect, useId, useRef, useState, type PointerEvent } from "react";
import { Columns2, Expand, Image as ImageIcon, Layers3, Maximize2, Minus, Move, Plus, RotateCcw, SlidersHorizontal } from "lucide-react";
import { Alert } from "./UI";
import { fitImage } from "../services/explanationImages";
import "./xray-viewer.css";

export type ViewerMode = "original" | "overlay" | "side" | "wipe";
type Size = { width: number; height: number };
function useImageSize(src?: string) {
  const [state, setState] = useState<{ src?: string; size?: Size; error?: string }>({});
  useEffect(() => {
    if (!src) return;
    let active = true;
    const img = new Image();
    img.onload = () => { if (active) setState({ src, size: { width: img.naturalWidth, height: img.naturalHeight } }); };
    img.onerror = () => { if (active) setState({ src, error: "This image could not be loaded. Generate the explanation again." }); };
    img.src = src;
    return () => { active = false; img.onload = null; img.onerror = null; };
  }, [src]);
  return state.src === src ? state : {};
}

function ImageFrame({ src, label, size, upper, upperLabel, opacity = 1, wipe, zoom }: {
  src: string; label: string; size?: Size; upper?: string; upperLabel?: string; opacity?: number; wipe?: number; zoom: number;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; left: number; top: number } | null>(null);
  const [bounds, setBounds] = useState<Size>({ width: 0, height: 0 });
  useEffect(() => {
    if (!viewport.current) return;
    const observer = new ResizeObserver(([entry]) => setBounds({ width: entry.contentRect.width, height: entry.contentRect.height }));
    observer.observe(viewport.current);
    return () => observer.disconnect();
  }, []);
  const fit = size ? fitImage(size.width, size.height, Math.max(0, bounds.width - 32), Math.max(0, bounds.height - 32)) : { width: 0, height: 0 };
  const width = fit.width * zoom / 100;
  const height = fit.height * zoom / 100;
  useEffect(() => {
    if (viewport.current) {
      viewport.current.scrollLeft = Math.max(0, (width + 32 - bounds.width) / 2);
      viewport.current.scrollTop = Math.max(0, (height + 32 - bounds.height) / 2);
    }
  }, [src, zoom, width, height, bounds.width, bounds.height]);
  function start(event: PointerEvent<HTMLDivElement>) {
    if (zoom <= 100 || !viewport.current) return;
    drag.current = { x: event.clientX, y: event.clientY, left: viewport.current.scrollLeft, top: viewport.current.scrollTop };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  return <div className={`xr-viewport${zoom > 100 ? " xr-can-pan" : ""}`} ref={viewport}
    onPointerDown={start} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}
    onPointerMove={event => {
      if (!drag.current || !viewport.current) return;
      viewport.current.scrollLeft = drag.current.left - event.clientX + drag.current.x;
      viewport.current.scrollTop = drag.current.top - event.clientY + drag.current.y;
    }}>
    <div className="xr-surface" style={{ width: Math.max(bounds.width, width + 32), height: Math.max(bounds.height, height + 32) }}>
      {!size ? <span className="xr-image-loading">Loading image…</span> : <div className="xr-image-frame" style={{ width, height }}>
        <img src={src} alt={label} draggable={false} />
        {upper && <img className="xr-upper" src={upper} alt={upperLabel || "Explanation overlay"} draggable={false}
          style={{ opacity, clipPath: wipe === undefined ? undefined : `inset(0 ${100 - wipe}% 0 0)` }} />}
        {upper && wipe !== undefined && <span className="xr-divider" style={{ left: `${wipe}%` }} aria-hidden="true" />}
      </div>}
    </div>
  </div>;
}

export function XrayViewer({ original, map, overlay, mapLabel = "Explanation", mode, setMode, opacity = 1, setOpacity }: {
  original: string; map?: string; overlay?: string; mapLabel?: string;
  mode?: ViewerMode; setMode?: (value: ViewerMode) => void; opacity?: number; setOpacity?: (value: number) => void;
}) {
  const id = useId();
  const root = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(100);
  const [wipe, setWipe] = useState(50);
  const [localMode, setLocalMode] = useState<ViewerMode>("overlay");
  const [full, setFull] = useState(false);
  const explanation = overlay || map;
  const source = useImageSize(original);
  const evidence = useImageSize(explanation);
  const aligned = !!(source.size && evidence.size && source.size.width === evidence.size.width && source.size.height === evidence.size.height);
  const selected = explanation ? mode ?? localMode : "original";
  const shown = selected === "wipe" && source.size && evidence.size && !aligned ? "side" : selected;
  useEffect(() => { setZoom(100); setWipe(50); }, [original, explanation]);
  useEffect(() => {
    const change = () => setFull(document.fullscreenElement === root.current);
    document.addEventListener("fullscreenchange", change);
    return () => document.removeEventListener("fullscreenchange", change);
  }, []);
  const choose = (v: ViewerMode) => setMode ? setMode(v) : setLocalMode(v);
  const reset = () => { setZoom(100); setWipe(50); };
  async function fullscreen() {
    try { if (document.fullscreenElement === root.current) await document.exitFullscreen(); else await root.current?.requestFullscreen(); }
    catch { /* The fit and zoom controls still work when fullscreen is unsupported. */ }
  }
  // The finished backend overlay is displayed directly at full opacity by default.
  // A raw map is not passed off as the report's finished overlay.
  const finishedLabel = overlay ? mapLabel : `${mapLabel} · raw map`;
  const modes: [ViewerMode, string, typeof ImageIcon][] = [["original", "Original", ImageIcon], ["overlay", "Explanation", Layers3], ["side", "Compare", Columns2], ["wipe", "Wipe", SlidersHorizontal]];
  return <div ref={root} className="xr-viewer" onKeyDown={event => {
    if (event.target instanceof HTMLInputElement) return;
    if (event.key === "+" || event.key === "=") setZoom(v => Math.min(400, v + 25));
    if (event.key === "-") setZoom(v => Math.max(100, v - 25));
    if (event.key === "0") reset();
  }}>
    <div className="xr-toolbar">
      <div className="xr-title"><span className="xr-live-dot" /><div><strong>{shown === "original" ? "Original chest X-ray" : mapLabel}</strong><small>Full image at fit · zoom when needed</small></div></div>
      <div className="xr-modes" aria-label="Image view">{explanation && modes.map(([value, label, Icon]) => <button type="button" key={value} aria-pressed={selected === value} onClick={() => choose(value)}><Icon size={15} />{label}</button>)}</div>
    </div>
    {(source.error || evidence.error) ? <Alert>{source.error || evidence.error}</Alert> : shown === "side" && explanation ?
      <div className="xr-side"><figure><ImageFrame src={original} label="Original chest X-ray" size={source.size} zoom={zoom} /><figcaption>Original X-ray</figcaption></figure>
        <figure><ImageFrame src={explanation} label={finishedLabel} size={evidence.size} zoom={zoom} /><figcaption>{finishedLabel}</figcaption></figure></div>
      : shown === "original" || !explanation ? <ImageFrame src={original} label="Original chest X-ray" size={source.size} zoom={zoom} />
      : shown === "wipe" && aligned ? <ImageFrame src={original} label="Original chest X-ray" size={source.size} upper={explanation} upperLabel={finishedLabel} opacity={opacity} wipe={wipe} zoom={zoom} />
      : overlay && aligned && opacity < 1 ? <ImageFrame src={original} label="Original chest X-ray" size={source.size} upper={overlay} upperLabel={mapLabel} opacity={opacity} zoom={zoom} />
      : <ImageFrame src={explanation} label={finishedLabel} size={evidence.size} zoom={zoom} />}
    {selected === "wipe" && source.size && evidence.size && !aligned && <Alert kind="warning">Different image dimensions. Showing a separate comparison to preserve alignment.</Alert>}
    {shown === "wipe" && aligned && <label className="xr-slider" htmlFor={`${id}-wipe`}>Compare position <input id={`${id}-wipe`} type="range" min="0" max="100" value={wipe} onChange={e => setWipe(Number(e.target.value))} /><output>{wipe}%</output></label>}
    {overlay && aligned && setOpacity && (shown === "overlay" || shown === "wipe") && <label className="xr-slider" htmlFor={`${id}-opacity`}>Overlay visibility <input id={`${id}-opacity`} type="range" min="0" max="1" step="0.05" value={opacity} onChange={e => setOpacity(Number(e.target.value))} /><output>{Math.round(opacity * 100)}%</output></label>}
    <div className="xr-footer"><span><Move size={14} /> {zoom > 100 ? "Drag to pan" : "Whole image visible"}</span><div className="xr-zoom">
      <button type="button" onClick={() => setZoom(v => Math.max(100, v - 25))} disabled={zoom === 100} aria-label="Zoom out"><Minus size={16} /></button>
      <output aria-label="Zoom level">{zoom === 100 ? "Fit" : `${zoom}%`}</output>
      <button type="button" onClick={() => setZoom(v => Math.min(400, v + 25))} disabled={zoom === 400} aria-label="Zoom in"><Plus size={16} /></button>
      <button type="button" onClick={reset} aria-label="Fit full image"><Expand size={16} /><span>Fit</span></button>
      <button type="button" onClick={reset} aria-label="Reset image view"><RotateCcw size={16} /></button>
      <button type="button" onClick={() => void fullscreen()} aria-label={full ? "Exit fullscreen" : "Open fullscreen"}><Maximize2 size={16} /></button>
    </div></div>
    {explanation && <p className="xr-note">{overlay ? "Backend explanation overlay" : "Raw explanation map · finished overlay unavailable"} · Visual influence does not indicate clinical severity.</p>}
  </div>;
}
