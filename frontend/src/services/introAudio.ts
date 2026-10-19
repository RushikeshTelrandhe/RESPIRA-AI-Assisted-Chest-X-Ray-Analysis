/** Plays bundled recordings only. No browser TTS or synthesized pronunciation. */
export class IntroAudio {
  private music: HTMLAudioElement | null = null;
  private voice: HTMLAudioElement | null = null;
  private stopped = false;
  private announced = false;
  private muted = true;
  private startedAt = 0;
  constructor(private files: { music: string | null; voice: string | null } = { music: null, voice: null }) {}
  async start(): Promise<boolean> {
    if (this.stopped || typeof Audio === 'undefined' || (!this.files.music && !this.files.voice)) return false;
    this.startedAt = performance.now();
    if (this.files.voice) { this.voice = new Audio(this.files.voice); this.voice.preload = 'auto'; this.voice.volume = 0.8; }
    if (!this.files.music) return false;
    this.music = new Audio(this.files.music); this.music.preload = 'auto'; this.music.volume = 0.17;
    try { await this.music.play(); if (this.stopped) { this.music.pause(); return false; } this.muted = false; return true; }
    catch { this.muted = true; return false; }
  }
  async setMuted(value: boolean): Promise<boolean> {
    if (this.stopped) return false;
    this.muted = value;
    if (value) { this.music?.pause(); this.voice?.pause(); return false; }
    if (!this.music) return !!this.voice;
    try {
      if (Number.isFinite(this.music.duration)) this.music.currentTime = Math.min((performance.now() - this.startedAt) / 1000, Math.max(0,this.music.duration - 0.1));
      await this.music.play();
      if (this.stopped || this.muted) { this.music.pause(); return false; }
      return true;
    } catch { this.muted = true; return false; }
  }
  announce() {
    if (this.announced || this.stopped || this.muted || !this.voice) return;
    this.announced = true;
    if (this.music) this.music.volume = 0.07;
    void this.voice.play().then(() => { if (this.stopped || this.muted) this.voice?.pause(); }).catch(() => undefined);
    // This one-word clip is never queued for playback after the introduction ends.
  }
  stop() {
    this.stopped = true;
    for (const item of [this.music,this.voice]) { if (item) { item.pause(); item.removeAttribute('src'); item.load(); } }
    this.music = null; this.voice = null;
  }
}
