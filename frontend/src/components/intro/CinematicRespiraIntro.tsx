import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { useReducedMotion } from "framer-motion";
import { ArrowRight, ScanLine, SkipForward, Volume2, VolumeX } from "lucide-react";
import { INTRO_MEDIA } from "../../services/introMedia";
import { markIntroSeen } from "../../services/introSession";
import { IntroAudio } from "../../services/introAudio";
import { INTRO_STAGES, scheduleIntro, type IntroScene } from "../../services/introTimeline";
import "./respira-motion.css";

const PATIENT_PLATE = "/media/respira-cough-keyframes.webp";
const XRAY_PLATE = "/media/respira-intro-poster.webp";
const ANATOMY_PLATE = "/media/respira-intro-anatomy.webp";

/** Brand mark, not an anatomical/diagnostic model. */
function BreathMark() {
  return <svg viewBox="0 0 64 64" fill="none" aria-hidden="true">
    <path d="M29 9v21l-8 8M35 9v21l8 8" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    <path d="M25 20C15 22 8 35 8 47c0 6 8 8 16 4 3-1 4-4 4-8V23c0-3-1-4-3-3ZM39 20c10 2 17 15 17 27 0 6-8 8-16 4-3-1-4-4-4-8V23c0-3 1-4 3-3Z" stroke="currentColor" strokeWidth="2" />
  </svg>;
}

export function CinematicRespiraIntro({ onComplete }: { onComplete?: () => void }) {
  const systemReduced = useReducedMotion();
  const [reducedPreference] = useState(() => {
    try { return localStorage.getItem("respira_ui_reduce_motion") === "true"; }
    catch { return false; }
  });
  const shouldReduce = Boolean(reducedPreference || systemReduced);
  // Mounted above the routes: a new browser document replays the intro, while
  // client-side navigation retains this completed component and cannot replay it.
  const [visible, setVisible] = useState(true);
  const [phase, setPhase] = useState<"preparing" | "playing" | "leaving">("preparing");
  const [scene, setScene] = useState<IntroScene>("patient");
  const [muted, setMuted] = useState(true);
  const audioAvailable = Boolean(INTRO_MEDIA.voice || INTRO_MEDIA.music);
  const video = useRef<HTMLVideoElement>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  const [mediaFailed, setMediaFailed] = useState(false);
  const [mediaReady, setMediaReady] = useState(false);
  const [chosenReduced, setChosenReduced] = useState(false);
  const dialog = useRef<HTMLElement>(null);
  const firstButton = useRef<HTMLButtonElement>(null);
  const finished = useRef(false);
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const audio = useRef<IntroAudio | null>(null);
  const cancelTimeline = useRef<(() => void) | null>(null);
  const completeCallback = useRef(onComplete);
  completeCallback.current = onComplete;

  const finish = useCallback(() => {
    if (finished.current) return;
    finished.current = true;
    cancelTimeline.current?.();
    audio.current?.stop();
    markIntroSeen();
    setPhase("leaving");
    leaveTimer.current = setTimeout(() => {
      setVisible(false);
      completeCallback.current?.();
    }, shouldReduce ? 80 : 260);
  }, [shouldReduce]);

  useEffect(() => {
    if (!visible) return;
    let active = true;
    const slow = setTimeout(() => { if (active) setMediaFailed(true); }, 2500);
    const images = [PATIENT_PLATE, ANATOMY_PLATE, XRAY_PLATE].map((src) => {
      const image = new Image();
      const promise = new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error("Intro illustration unavailable"));
      });
      image.src = src;
      return { image, promise };
    });
    void Promise.all(images.map(({ promise }) => promise)).then(() => {
      clearTimeout(slow);
      if (active) { setMediaReady(true); setMediaFailed(false); }
    }).catch(() => { if (active) setMediaFailed(true); });
    return () => {
      active = false;
      clearTimeout(slow);
      images.forEach(({ image }) => { image.onload = null; image.onerror = null; });
    };
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    const root = document.getElementById("root");
    const wasInert = root?.hasAttribute("inert") ?? false;
    root?.setAttribute("inert", "");
    document.body.style.overflow = "hidden";
    firstButton.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); finish(); return; }
      if (event.key !== "Tab") return;
      const buttons = Array.from(dialog.current?.querySelectorAll<HTMLButtonElement>("button:not([disabled])") ?? [])
        .filter((button) => button.getClientRects().length > 0);
      const first = buttons[0];
      const last = buttons[buttons.length - 1];
      if (!first) return;
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
      if (!wasInert) root?.removeAttribute("inert");
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, [visible, finish]);

  useEffect(() => {
    if (phase !== "playing" || finished.current) return;
    const introAudio = new IntroAudio(INTRO_MEDIA);
    audio.current = introAudio;
    void introAudio.start().then(playing => setMuted(!playing));
    if (video.current) void video.current.play().catch(() => setVideoFailed(true));
    cancelTimeline.current = scheduleIntro(chosenReduced, setScene, () => introAudio.announce(), finish);
    const onVisibility = () => { if (document.hidden) finish(); };
    document.addEventListener("visibilitychange", onVisibility);
    return () => { introAudio.stop(); cancelTimeline.current?.(); document.removeEventListener("visibilitychange", onVisibility); };
  }, [phase, chosenReduced, finish]);

  useEffect(() => {
    if (phase !== "preparing" || (!mediaReady && !mediaFailed)) return;
    setChosenReduced(shouldReduce || mediaFailed);
    setScene(shouldReduce || mediaFailed ? "brand" : "patient");
    setPhase("playing");
    dialog.current?.focus();
  }, [phase, mediaReady, mediaFailed, shouldReduce]);

  useEffect(() => {
    if (phase === "playing" && shouldReduce && !chosenReduced) {
      setChosenReduced(true);
      setScene("brand");
    }
  }, [phase, shouldReduce, chosenReduced]);

  useEffect(() => () => {
    audio.current?.stop();
    cancelTimeline.current?.();
    if (leaveTimer.current) clearTimeout(leaveTimer.current);
  }, []);

  async function toggleSound() {
    if (!audio.current) return;
    const next = !muted;
    const playing = await audio.current.setMuted(next);
    setMuted(!playing);
  }

  if (!visible || typeof document === "undefined") return null;
  const stageIndex = INTRO_STAGES.findIndex((item) => item.scene === scene);
  return createPortal(
    <section ref={dialog} className={`respira-film film-${phase}${chosenReduced ? " film-static" : ""}${INTRO_MEDIA.video && !videoFailed ? " film-has-video" : ""}`}
      role="dialog" aria-modal="true" aria-label="RESPIRA cinematic introduction" tabIndex={-1} data-scene={scene}>
      <div className="film-environment" aria-hidden="true" />
      <header className="film-topbar">
        <div className="film-mark"><BreathMark /><span>CHEST X-RAY INTELLIGENCE</span></div>
        <div className="film-top-actions">
          {phase === "playing" && audio.current && audioAvailable && <button type="button" className="film-icon-button"
            aria-label={muted ? "Unmute introduction" : "Mute introduction"} aria-pressed={!muted} onClick={() => void toggleSound()}>
            {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
          </button>}
          <button ref={firstButton} type="button" className="film-skip" onClick={finish}>Skip intro <SkipForward size={15} /></button>
        </div>
      </header>

      {phase === "preparing" ? <div className="film-preparing" aria-label="Preparing introduction"><div className="film-patient-still" /><span>Every breath. A clearer picture.</span></div> : <>
        <div className="film-stage" aria-hidden="true">
          {INTRO_MEDIA.video && !videoFailed && !chosenReduced && <video ref={video} className="film-real-footage" src={INTRO_MEDIA.video} autoPlay muted playsInline preload="auto" onError={() => setVideoFailed(true)} />}
          <div className="film-patient-camera"><div className="film-patient-still" /><div className="film-patient-cough" /></div>
          <div className="film-thorax-camera">
            <img src={ANATOMY_PLATE} className="film-anatomy-plate" alt="" />
            <div className="film-breath-light" /><div className="film-thorax-scan" />
          </div>
          <div className="film-radiograph-camera">
            <img src={XRAY_PLATE} alt="" />
            <div className="film-region"><span /><i /></div>
            <div className="film-region-caption"><i /> Illustrative highlight</div>
            <div className="film-radiograph-scan" />
            <span className="film-image-corner film-corner-tl" /><span className="film-image-corner film-corner-br" />
          </div>
          <div className="film-entry-light" /><div className="film-stage-shade" />
        </div>
        {scene !== "brand" && <div className="film-scene-copy" role="status"><span className="film-scene-number">0{stageIndex + 1}</span>
          <span><small>A CLOSER LOOK</small><strong>{INTRO_STAGES[stageIndex].label}</strong></span>
        </div>}
        {scene === "brand" && <div className="film-reveal">
          <div className="film-reveal-symbol"><BreathMark /></div>
          <p className="film-overline">SEE BEYOND THE IMAGE</p>
          <h1 aria-label="Respira">{"RESPIRA".split("").map((letter, index) => <span key={index} style={{ "--letter": index } as CSSProperties} aria-hidden="true">{letter}</span>)}</h1>
          <h2>AI-Assisted Chest X-Ray Analysis</h2>
          <p className="film-reveal-caption">Every breath. A clearer picture.</p>
          <p className="film-opening">Opening your clinical workspace <ArrowRight size={16} /></p>
          <small>Supports clinical review. Never replaces professional judgement.</small>
        </div>}
      </>}
      <footer className="film-footer">
        <span><ScanLine size={14} /> Illustrative sequence · not a patient result</span>
        <span className="film-footer-right">PATIENT → ANATOMY → INSIGHT</span>
      </footer>
      {phase !== "preparing" && <div className="film-progress" aria-hidden="true"><span /></div>}
    </section>, document.body,
  );
}
