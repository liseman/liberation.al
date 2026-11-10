// Shift+wheel shuttles gently at first, accelerating with recent scroll distance rather than event count.
// One notch is 100 pixels, roughly six lines, or one page. A pause or reversal starts a new gesture.
const PAUSE_MS = 250;
const DECAY_MS = 120;
const MAX_GAIN = 16;

export interface ShuttleInput {
  deltaX: number;
  deltaY: number;
  deltaMode: number;
  width: number;
  span: number;
  now: number;
}

export class WheelShuttle {
  private last: number | null = null;
  private direction = 0;
  private distance = 0;

  /** Seconds to seek for this event. Small trackpad deltas contribute proportionally, even at high event rates. */
  step({ deltaX, deltaY, deltaMode, width, span, now }: ShuttleInput): number {
    const delta = Math.abs(deltaX) > Math.abs(deltaY) ? deltaX : deltaY;
    if (![delta, width, span, now].every(Number.isFinite) || !delta || width <= 0 || span <= 0) return 0;
    const pixels = delta * (deltaMode === 1 ? 16 : deltaMode === 2 ? 100 : 1);
    const units = Math.min(1, Math.abs(pixels) / 100);
    const direction = Math.sign(delta);
    const elapsed = this.last == null ? PAUSE_MS : now - this.last;
    if (elapsed < 0 || elapsed >= PAUSE_MS || direction !== this.direction) this.distance = 0;
    else this.distance *= Math.exp(-elapsed / DECAY_MS);
    this.distance += units;
    this.last = now;
    this.direction = direction;
    const gain = Math.min(MAX_GAIN, 1 + Math.max(0, this.distance - 1) ** 2);
    // A quarter-pixel of the visible scale, bounded from one video frame to two seconds per unaccelerated notch.
    const seconds = Math.max(1 / 30, Math.min(2, span / width / 4));
    return direction * units * seconds * gain;
  }
}
