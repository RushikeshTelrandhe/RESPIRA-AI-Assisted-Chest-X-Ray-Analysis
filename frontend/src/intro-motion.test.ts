import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { IntroAudio } from "./services/introAudio";
import { INTRO_DURATION, INTRO_REDUCED_DURATION, INTRO_STAGES, scheduleIntro, sceneAt } from "./services/introTimeline";

describe("intro director", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("shows the radiograph and illustrative highlight before the name", () => {
    expect(sceneAt(0)).toBe("patient");
    expect(sceneAt(1600)).toBe("enter");
    expect(sceneAt(3000)).toBe("anatomy");
    expect(sceneAt(4400)).toBe("xray");
    expect(sceneAt(5000)).toBe("highlight");
    expect(sceneAt(6000)).toBe("brand");
  });

  it("clamps an out-of-range elapsed time safely", () => {
    expect(sceneAt(-100)).toBe("patient");
    expect(sceneAt(999999)).toBe("brand");
  });

  it("runs each scene in order, reveals once, and completes at eight seconds", () => {
    const scene = vi.fn(), reveal = vi.fn(), complete = vi.fn();
    scheduleIntro(false, scene, reveal, complete);
    vi.advanceTimersByTime(INTRO_DURATION - 1);
    expect(scene.mock.calls.map(([value]) => value)).toEqual(INTRO_STAGES.map(({ scene }) => scene));
    expect(reveal).toHaveBeenCalledTimes(1);
    expect(complete).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(complete).toHaveBeenCalledTimes(1);
  });

  it.each(["skip", "Escape", "media failure", "unmount", "hidden tab"])("cancels pending scenes and speech on %s", () => {
    const scene = vi.fn(), reveal = vi.fn(), complete = vi.fn();
    const cancel = scheduleIntro(false, scene, reveal, complete);
    vi.advanceTimersByTime(1000);
    cancel();
    cancel();
    vi.advanceTimersByTime(10000);
    expect(scene).toHaveBeenCalledTimes(1);
    expect(reveal).not.toHaveBeenCalled();
    expect(complete).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("uses only the static brand in reduced-motion mode", () => {
    const scene = vi.fn(), reveal = vi.fn(), complete = vi.fn();
    scheduleIntro(true, scene, reveal, complete);
    expect(scene).toHaveBeenCalledExactlyOnceWith("brand");
    vi.advanceTimersByTime(INTRO_REDUCED_DURATION);
    expect(scene).toHaveBeenCalledTimes(1);
    expect(reveal).toHaveBeenCalledTimes(1);
    expect(complete).toHaveBeenCalledTimes(1);
  });

  it("supports effect cleanup and restart without duplicate callbacks", () => {
    const scene = vi.fn(), reveal = vi.fn(), complete = vi.fn();
    const cancel = scheduleIntro(false, scene, reveal, complete);
    cancel();
    scheduleIntro(false, scene, reveal, complete);
    vi.advanceTimersByTime(INTRO_DURATION);
    expect(reveal).toHaveBeenCalledTimes(1);
    expect(complete).toHaveBeenCalledTimes(1);
  });
});

describe("bundled intro audio", () => {
  afterEach(() => vi.unstubAllGlobals());
  it("continues silently when no recordings are configured", async () => {
    const audio = new IntroAudio();
    expect(await audio.start()).toBe(false);
    expect(() => { audio.announce(); audio.stop(); audio.stop(); }).not.toThrow();
  });
  it("handles blocked autoplay without speech synthesis or a queued late voice", async () => {
    const play = vi.fn(() => Promise.reject(new Error('autoplay blocked')));
    const pause = vi.fn();
    vi.stubGlobal('Audio', class { volume=0; preload=''; duration=8; currentTime=0; play=play; pause=pause; removeAttribute() {} load() {} });
    const audio = new IntroAudio({music:'/test-music.mp3',voice:'/test-voice.mp3'});
    expect(await audio.start()).toBe(false);
    audio.announce();
    expect(play).toHaveBeenCalledTimes(1);
    audio.stop();
    expect(await audio.setMuted(false)).toBe(false);
    audio.announce();
    expect(play).toHaveBeenCalledTimes(1);
  });
  it("plays the recorded word once and stops both recordings on exit", async () => {
    const play=vi.fn(() => Promise.resolve()); const pause=vi.fn();
    vi.stubGlobal('Audio', class { volume=0; preload=''; duration=8; currentTime=0; play=play; pause=pause; removeAttribute() {} load() {} });
    const audio=new IntroAudio({music:'/test-music.mp3',voice:'/test-voice.mp3'});
    expect(await audio.start()).toBe(true);
    audio.announce(); audio.announce();
    expect(play).toHaveBeenCalledTimes(2);
    audio.stop();
    expect(pause).toHaveBeenCalledTimes(2);
  });
});
