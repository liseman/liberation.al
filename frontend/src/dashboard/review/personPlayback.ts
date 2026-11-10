// A bounded playback queue for one person's selected moments. Media and clocks stay in ReviewEngine;
// this controller owns only the immutable queue, its active step, and transitions at an exact endpoint.
import { createStore } from 'zustand/vanilla';
import type { PlaybackMoment } from './personMomentsModel';

export type PersonPlaybackMode = 'audio' | 'video';
export interface PersonPlaybackRun {
  person: string;
  mode: PersonPlaybackMode;
  moments: readonly PlaybackMoment[];
  index: number;
}
export interface PersonPlaybackHost {
  play: (moment: PlaybackMoment, mode: PersonPlaybackMode) => void;
  stop: () => void;
}

export class PersonPlayback {
  readonly ui = createStore<{ run: PersonPlaybackRun | null }>(() => ({ run: null }));
  private readonly host: PersonPlaybackHost;
  constructor(host: PersonPlaybackHost) {
    this.host = host;
  }

  get run(): PersonPlaybackRun | null {
    return this.ui.getState().run;
  }
  get current(): PlaybackMoment | null {
    const run = this.run;
    return run?.moments[run.index] ?? null;
  }
  start(person: string, mode: PersonPlaybackMode, moments: readonly PlaybackMoment[]): void {
    // Snapshot the selection: a refreshed transcript must not reorder a queue already playing.
    const snapshot = moments
      .filter(
        m =>
          Number.isFinite(m.start) &&
          Number.isFinite(m.end) &&
          m.start >= 0 &&
          m.end > m.start &&
          (mode === 'audio' || !!m.cam),
      )
      .map(m => ({ ...m }));
    if (!snapshot.length) {
      this.stop();
      return;
    }
    this.ui.setState({ run: { person, mode, moments: snapshot, index: 0 } });
    const first = this.current;
    if (first) this.host.play(first, mode);
  }
  /** A manual action takes over playback. Its caller decides whether to keep playing or pause. */
  cancel(): void {
    if (this.run) this.ui.setState({ run: null });
  }
  stop(): void {
    this.cancel();
    this.host.stop();
  }
  /** Check after reading the media clock, before normal media rollover. True means this queue handled it. */
  atEnd(ts: number): boolean {
    const moment = this.current;
    if (!moment || !Number.isFinite(ts) || ts < moment.end) return false;
    return this.next();
  }
  /** A line clip ended before a frame tick could see its endpoint. Room audio under video is not a queue end. */
  audioEnded(): boolean {
    return this.run?.mode === 'audio' ? this.next() : false;
  }
  /** Also used when the current recording is unavailable, so it cannot strand the queue. */
  next(): boolean {
    const run = this.run;
    if (!run) return false;
    const index = run.index + 1;
    const next = run.moments[index];
    if (!next) this.stop();
    else {
      this.ui.setState({ run: { ...run, index } });
      this.host.play(next, run.mode);
    }
    return true;
  }
}
