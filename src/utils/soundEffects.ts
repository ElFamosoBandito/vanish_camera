import type { VanishMode } from './vfxEngine';

class SoundEngine {
  private ctx: AudioContext | null = null;
  private narutoAudio: HTMLAudioElement | null = null;
  public muted: boolean = false;

  constructor() {
    if (typeof window !== 'undefined' && typeof Audio !== 'undefined') {
      try {
        this.narutoAudio = new Audio('/assets/vanish.mp3');
        this.narutoAudio.preload = 'auto';
      } catch {
        this.narutoAudio = null;
      }
    }
  }

  private getContext(): AudioContext | null {
    if (this.muted || typeof window === 'undefined') return null;
    if (!this.ctx) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  public playModeSound(mode: VanishMode) {
    if (this.muted) return;

    if (mode === 'naruto' && this.narutoAudio) {
      try {
        this.narutoAudio.currentTime = 0;
        this.narutoAudio.play().catch(() => {});
        return;
      } catch {
        // Fallback to Web Audio if blocked
      }
    }

    const ctx = this.getContext();
    if (!ctx) return;
    const now = ctx.currentTime;

    if (mode === 'goku') {
      // Iconic Dragon Ball Z Shunkan Ido high-frequency FM teleportation sound
      const osc = ctx.createOscillator();
      const mod = ctx.createOscillator();
      const modGain = ctx.createGain();
      const mainGain = ctx.createGain();

      osc.type = 'sawtooth';
      mod.type = 'sine';
      mod.frequency.setValueAtTime(75, now);
      modGain.gain.setValueAtTime(520, now);

      osc.frequency.setValueAtTime(1650, now);
      osc.frequency.exponentialRampToValueAtTime(3200, now + 0.14);
      osc.frequency.exponentialRampToValueAtTime(210, now + 0.34);

      mainGain.gain.setValueAtTime(0.01, now);
      mainGain.gain.linearRampToValueAtTime(0.24, now + 0.03);
      mainGain.gain.exponentialRampToValueAtTime(0.001, now + 0.38);

      mod.connect(modGain);
      modGain.connect(osc.frequency);
      osc.connect(mainGain);
      mainGain.connect(ctx.destination);

      mod.start(now);
      osc.start(now);
      mod.stop(now + 0.4);
      osc.stop(now + 0.4);
    } else {
      // Normal Fade Out shimmer
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(520, now);
      osc.frequency.exponentialRampToValueAtTime(180, now + 0.8);
      gain.gain.setValueAtTime(0.16, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.85);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.9);
    }
  }
}

export const soundEngine = new SoundEngine();
