export type IntroScene = "patient" | "enter" | "anatomy" | "xray" | "highlight" | "brand";
export const INTRO_DURATION = 8000;
export const INTRO_REDUCED_DURATION = 2200;
export const INTRO_STAGES: ReadonlyArray<{ at: number; scene: IntroScene; label: string }> = [
  { at: 0, scene: "patient", label: "A breath in focus" },
  { at: 1550, scene: "enter", label: "Beneath the surface" },
  { at: 2750, scene: "anatomy", label: "Inside the thorax" },
  { at: 4200, scene: "xray", label: "From anatomy to radiograph" },
  { at: 4850, scene: "highlight", label: "Illustrating visual explanation" },
  { at: 5900, scene: "brand", label: "From image to insight" },
];

export function sceneAt(elapsed: number): IntroScene {
  return [...INTRO_STAGES].reverse().find((stage) => elapsed >= stage.at)?.scene ?? "patient";
}

/** One clock for scene labels, sound cue and completion. Disposing cancels everything. */
export function scheduleIntro(reduced: boolean, onScene: (scene: IntroScene) => void, onReveal: () => void, onFinish: () => void) {
  let cancelled = false;
  const timers: ReturnType<typeof setTimeout>[] = [];
  const later = (at: number, fn: () => void) => {
    timers.push(setTimeout(() => { if (!cancelled) fn(); }, at));
  };
  if (reduced) {
    onScene("brand");
    later(80, onReveal);
    later(INTRO_REDUCED_DURATION, onFinish);
  } else {
    onScene("patient");
    for (const stage of INTRO_STAGES.slice(1)) later(stage.at, () => {
      onScene(stage.scene);
      if (stage.scene === "brand") onReveal();
    });
    later(INTRO_DURATION, onFinish);
  }
  return () => { cancelled = true; timers.forEach(clearTimeout); };
}
