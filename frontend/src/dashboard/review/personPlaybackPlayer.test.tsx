import { act, screen, waitFor, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { EngineContext } from './context';
import { ReviewEngine, type VideoLike } from './engine';
import { Player } from './Player';
import { momentsOf, playbackOf } from './personMomentsModel';
import type { Chip } from './types';

const DAY = '2026-10-27';
const T = new Date(2026, 9, 27, 10).getTime() / 1000;
const PERSON: Chip = {
  id: 'person:alice',
  kind: 'person',
  label: 'Alice',
  spans: [[T + 10, T + 20]],
  cams: { rear: [[T + 10, T + 20]] },
  total_s: 10,
  count: 1,
};

/** The decoder boundary: seeks finish, frames remain still until the engine asks them to play. */
class DecoderVideo extends EventTarget implements VideoLike {
  src = '';
  preload = 'auto';
  readyState = 4;
  duration = 60;
  error = null;
  seeking = false;
  paused = true;
  ended = false;
  playbackRate = 1;
  defaultPlaybackRate = 1;
  private time = 0;
  private readonly canvas: HTMLCanvasElement;
  constructor(canvas: HTMLCanvasElement) {
    super();
    this.canvas = canvas;
  }
  get currentTime(): number {
    return this.time;
  }
  set currentTime(value: number) {
    this.time = value;
    queueMicrotask(() => this.dispatchEvent(new Event('seeked')));
  }
  async play(): Promise<void> {
    const wasPaused = this.paused;
    this.paused = false;
    if (wasPaused) this.dispatchEvent(new Event('playing'));
  }
  pause(): void {
    this.paused = true;
  }
  load(): void {}
  close(): void {
    this.pause();
  }
  getAttribute(name: string): string | null {
    return name === 'src' ? this.src || null : this.canvas.getAttribute(name);
  }
  removeAttribute(name: string): void {
    if (name === 'src') this.src = '';
    else this.canvas.removeAttribute(name);
  }
  toggleAttribute(name: string, force?: boolean): boolean {
    return this.canvas.toggleAttribute(name, force);
  }
}

const engines: ReviewEngine[] = [];

async function open() {
  history.replaceState(null, '', `/review#day=${DAY}&t=${T + 5}`);
  const engine = new ReviewEngine({
    cams: ['front', 'rear'],
    now: () => T + 86400,
    fetchNotes: async () => ({ cams: {} }),
  });
  engines.push(engine);
  engine.start();
  engine.setDayData(DAY, engine.ui.getState().gen, {
    entries: [],
    spans: [],
    segments: ['front', 'rear'].map(cam => ({
      cam,
      start: T,
      end: T + 60,
      url: `gateway:${cam}:${T * 1e6}:${(T + 60) * 1e6}`,
    })),
  });
  engine.setChips(DAY, engine.ui.getState().gen, [PERSON]);
  const decoders: DecoderVideo[] = [];
  renderPage(
    <EngineContext value={engine}>
      <Player
        makeVideo={canvas => {
          const decoder = new DecoderVideo(canvas);
          decoders.push(decoder);
          return decoder;
        }}
      />
    </EngineContext>,
  );
  await act(async () => {});
  // The person's plan uses their own evidence, including cameras outside the current Show only filter.
  const plan = playbackOf(
    momentsOf('Alice', [], () => '', [PERSON], []),
    'video',
    engine.dayStart,
  );
  return { engine, decoders, plan };
}

beforeEach(() => {
  vi.stubGlobal('NAMES', { views: { front: 'front camera', rear: 'rear camera' } });
  server.use(
    http.get('/review/boxes', () => HttpResponse.json({ boxes: [] })),
    http.get('/ambient', () => HttpResponse.json({ devices: {} })),
  );
});
afterEach(() => {
  for (const engine of engines.splice(0)) engine.stop();
  vi.unstubAllGlobals();
});

it('reveals a queued rear-camera source in the real Player despite a front-only camera filter', async () => {
  const { engine, plan } = await open();
  act(() => engine.pickCameras(['front']));
  expect(screen.getByLabelText('Front camera video')).toBeVisible();
  expect(screen.getByLabelText('Rear camera video')).not.toBeVisible();
  expect(plan).toEqual([{ start: T + 8, end: T + 20, cam: 'rear' }]);

  act(() => engine.personPlayback.start('Alice', 'video', plan));

  expect(engine.ui.getState().camera).toBe('rear');
  await waitFor(() => expect(screen.getByLabelText('Rear camera video')).toBeVisible());
  const rear = screen.getByLabelText('Rear camera video').parentElement as HTMLElement;
  expect(within(rear).getByText('Rear camera')).toBeVisible();
  expect(within(rear).queryByRole('button', { name: 'Show the rear camera large' })).not.toBeInTheDocument();
  expect(engine.personPlayback.current).toEqual({ start: T + 8, end: T + 20, cam: 'rear' });
});

it('stops the active queue when a manual camera filter hides its source', async () => {
  const { engine, decoders, plan } = await open();
  act(() => engine.personPlayback.start('Alice', 'video', plan));
  await waitFor(() => {
    expect(engine.playing).toBe(true);
    expect(decoders.some(decoder => !decoder.paused)).toBe(true);
  });
  expect(screen.getByLabelText('Rear camera video')).toBeVisible();

  act(() => engine.pickCameras(['front']));

  expect(engine.personPlayback.run).toBeNull();
  expect(engine.playing).toBe(false);
  expect(engine.ui.getState().playing).toBe(false);
  expect(decoders.every(decoder => decoder.paused)).toBe(true);
  expect(screen.getByLabelText('Rear camera video')).not.toBeVisible();
  expect(screen.getByLabelText('Front camera video')).toBeVisible();
});
