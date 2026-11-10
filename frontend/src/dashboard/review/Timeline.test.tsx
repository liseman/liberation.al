import { act, fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { PREFS_KEY, readPrefs, writePrefs } from '../../shared/prefs';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { usePrefs } from '../lib/prefs';
import { EngineContext } from './context';
import { ReviewEngine } from './engine';
import { activityChips, layout } from './lanes';
import { ChunkLine, Timeline, TimelineTools } from './Timeline';
import { RULER } from './time';
import type { Chip } from './types';

const DAY = '2026-10-27';
const D0 = new Date(2026, 9, 27).getTime() / 1000;

function setup(mode: 'compact' | 'expanded' | null = 'expanded') {
  if (mode) writePrefs({ ...readPrefs(), reviewTimeline: mode });
  usePrefs.setState(readPrefs(), true); // the store reads storage once; this test's choice must reach it
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.pickDay(DAY);
  engine.setDayData(DAY, engine.ui.getState().gen, { entries: [], segments: [], spans: [] });
  engine.setView(D0 + 36000, D0 + 39600); // an hour across
  const page = renderPage(
    <EngineContext value={engine}>
      <Timeline />
      <TimelineTools />
    </EngineContext>,
  );
  const canvas = page.container.querySelector('canvas');
  if (!canvas) throw new Error('no canvas');
  return { engine, canvas };
}
const span = (e: ReviewEngine) => e.view.t1 - e.view.t0;

beforeEach(() => {
  server.use(http.get('/speakers', () => HttpResponse.json({ speakers: [], merged: {} })));
});

it('dims time outside the cross-facet intersection across every lane while Skip is off', () => {
  const { engine, canvas } = setup();
  const track = canvas.closest('[role="slider"]');
  if (!track) throw new Error('no timeline');
  Object.defineProperty(track, 'clientWidth', { value: 1000 });
  const masks: number[][] = [];
  const g = {
    globalAlpha: 1,
    clearRect() {},
    fillRect(x: number, y: number, width: number, height: number) {
      if (this.globalAlpha === 0.55 && y === RULER) masks.push([x, width, height]);
    },
    fillText() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    fill() {},
    strokeRect() {},
    setLineDash() {},
  };
  vi.spyOn(canvas, 'getContext').mockReturnValue(g as unknown as CanvasRenderingContext2D);
  const chip = (id: string, a: number, b: number): Chip => ({
    id,
    kind: id.split(':')[0] ?? '',
    label: id,
    spans: [[D0 + a, D0 + b]],
    total_s: b - a,
    count: 1,
  });
  act(() => {
    engine.setChips(DAY, engine.ui.getState().gen, [
      chip('person:alice', 36900, 38700),
      chip('object:mug', 37800, 39600),
    ]);
    engine.toggleChip('person:alice');
    engine.toggleChip('object:mug');
    engine.ui.setState({ skip: false });
    engine.setView(D0 + 36000, D0 + 39600, true);
  });
  act(() => engine.frame(performance.now()));
  expect(masks.map(([x, width]) => [x, width])).toEqual([
    [0, 500],
    [750, 250],
  ]);
  expect(masks.every(([, , height]) => height === layout(['cam0'], {}).height - RULER)).toBe(true);
});

describe('timeline layout', () => {
  it('defaults to a thin compact strip and expands without changing its canvas, time, or filters', () => {
    const { engine, canvas } = setup(null);
    const track = screen.getByRole('slider');
    expect(screen.getByRole('button', { name: 'Camera lanes' })).toHaveAttribute('aria-pressed', 'false');
    expect(track).toHaveAttribute('data-layout', 'compact');
    expect(screen.queryByText('Who spoke')).not.toBeInTheDocument();
    const height = Number.parseFloat(track.style.height);
    engine.jump(D0 + 37800);
    const time = engine.t;
    const view = { ...engine.view };
    const selection = engine.ui.getState().sel;
    fireEvent.click(screen.getByRole('button', { name: 'Camera lanes' }));
    expect(screen.getByText('Who spoke')).toBeInTheDocument();
    expect(Number.parseFloat(track.style.height)).toBeGreaterThan(height);
    expect(track.querySelector('canvas')).toBe(canvas);
    expect(engine.t).toBe(time);
    expect(engine.view).toEqual(view);
    expect(engine.ui.getState().sel).toBe(selection);
    expect(readPrefs().reviewTimeline).toBe('expanded');
    fireEvent.click(screen.getByRole('button', { name: 'Camera lanes' }));
    expect(track).toHaveStyle({ height: `${height}px` });
    expect(readPrefs().reviewTimeline).toBe('compact');
  });

  it('restores the saved choice when the timeline opens again', () => {
    writePrefs({ reviewTimeline: 'expanded', theme: 'dark' });
    setup(null);
    expect(screen.getByRole('button', { name: 'Camera lanes' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Camera lanes' }));
    expect(JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}')).toMatchObject({
      reviewTimeline: 'compact',
      theme: 'dark',
    });
  });

  it('still switches layout when browser storage cannot be read or written', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('full');
    });
    setup(null);
    expect(screen.getByRole('button', { name: 'Camera lanes' })).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(screen.getByRole('button', { name: 'Camera lanes' }));
    expect(screen.getByRole('slider')).toHaveAttribute('data-layout', 'expanded');
  });

  it.each(['compact', 'expanded'] as const)('seeks an exact empty timestamp in %s mode', mode => {
    const { engine, canvas } = setup(mode);
    const track = screen.getByRole('slider');
    Object.defineProperty(track, 'clientWidth', { value: 1000 });
    vi.spyOn(canvas, 'getContext').mockReturnValue(null);
    act(() => engine.frame(performance.now()));
    fireEvent.pointerDown(track, { pointerId: 1, clientX: 375, clientY: 28 });
    fireEvent.pointerUp(track, { pointerId: 1, clientX: 375, clientY: 28 });
    expect(engine.t).toBe(D0 + 36000 + 3600 * 0.375);
  });
});

it('shows a neutral activity count in the hover title', () => {
  const { engine, canvas } = setup();
  const track = canvas.closest('[role="slider"]');
  if (!track) throw new Error('no timeline');
  Object.defineProperty(track, 'clientWidth', { value: 1000 });
  vi.spyOn(canvas, 'getContext').mockReturnValue(null);
  act(() =>
    engine.setChips(
      DAY,
      engine.ui.getState().gen,
      activityChips([
        {
          id: 'model-span',
          camera: 'cam0',
          start: D0 + 37800,
          end: D0 + 37860,
          kind: 'reading',
          name: null,
          count: 3,
          person: null,
          promoted: false,
        },
      ]),
    ),
  );
  act(() => engine.frame(performance.now()));
  fireEvent.pointerMove(track, {
    pointerId: 1,
    pointerType: 'mouse',
    clientX: 500,
    clientY: layout(['cam0'], {}).events[0] + 1,
  });
  expect(screen.getByText(/Reading.*Count: 3/)).toBeVisible();
  expect(screen.queryByText(/repetitions/)).not.toBeInTheDocument();
});

// A hover before the first paint measured the track used to throw "Invalid time value" in the clock.
it('shows the hovered time when the pointer arrives before the first paint', () => {
  const { canvas } = setup();
  const track = canvas.closest('[role="slider"]');
  if (!track) throw new Error('no timeline');
  Object.defineProperty(track, 'clientWidth', { value: 1000 });
  vi.spyOn(canvas, 'getContext').mockReturnValue(null);
  const errors: unknown[] = [];
  const onError = (e: ErrorEvent) => {
    errors.push(e.error);
    e.preventDefault();
  };
  window.addEventListener('error', onError);
  try {
    fireEvent.pointerMove(track, { pointerId: 1, pointerType: 'mouse', clientX: 500, clientY: RULER + 1 });
  } finally {
    window.removeEventListener('error', onError);
  }
  expect(errors).toEqual([]);
  expect(screen.getByText(/^10:30:00/)).toBeInTheDocument(); // halfway across 10:00 to 11:00
});

describe('timeline wheel', () => {
  it('zooms in when the wheel turns up over the canvas, and out when it turns down', () => {
    const { engine, canvas } = setup();
    fireEvent.wheel(canvas, { deltaY: -100 });
    expect(span(engine)).toBeLessThan(3600);
    const zoomed = span(engine);
    fireEvent.wheel(canvas, { deltaY: 300 });
    expect(span(engine)).toBeGreaterThan(zoomed);
  });

  it('pans when the wheel moves mostly sideways (a trackpad swipe)', () => {
    const { engine, canvas } = setup();
    const t0 = engine.view.t0;
    fireEvent.wheel(canvas, { deltaX: 50, deltaY: 5 });
    expect(engine.view.t0).toBeGreaterThan(t0);
    expect(span(engine)).toBe(3600);
  });

  it.each([
    { deltaY: 50, deltaX: 0, deltaMode: 0, seconds: 0.45 },
    { deltaY: 5, deltaX: -50, deltaMode: 0, seconds: -0.45 },
    { deltaY: -3, deltaX: 0, deltaMode: 1, seconds: -0.432 },
    { deltaY: 1, deltaX: 0, deltaMode: 2, seconds: 0.9 },
  ])('shuttles the cursor with Shift wheel without zooming: %j', async ({ seconds, ...wheel }) => {
    const { engine, canvas } = setup();
    const track = canvas.closest('[role="slider"]');
    if (!track) throw new Error('no timeline');
    Object.defineProperty(track, 'clientWidth', { value: 1000 });
    engine.jump(D0 + 37800);
    const view = { ...engine.view };
    expect(fireEvent.wheel(canvas, { ...wheel, shiftKey: true })).toBe(false);
    await waitFor(() => expect(engine.t).toBeCloseTo(D0 + 37800 + seconds));
    expect(span(engine)).toBe(3600);
    if (Math.abs(seconds) < 1800) expect(engine.view).toEqual(view);
  });

  it('combines a burst into one seek per animation frame', () => {
    const { engine, canvas } = setup();
    const track = canvas.closest('[role="slider"]');
    if (!track) throw new Error('no timeline');
    Object.defineProperty(track, 'clientWidth', { value: 1000 });
    engine.jump(D0 + 37800);
    const jump = vi.spyOn(engine, 'jump');
    let frame: FrameRequestCallback | undefined;
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      frame = callback;
      return 99;
    });
    vi.spyOn(performance, 'now').mockReturnValue(100);
    fireEvent.wheel(canvas, { deltaY: 50, shiftKey: true });
    fireEvent.wheel(canvas, { deltaY: 50, shiftKey: true });
    expect(jump).not.toHaveBeenCalled();
    act(() => frame?.(100));
    expect(jump).toHaveBeenCalledTimes(1);
    expect(engine.t).toBeCloseTo(D0 + 37800.9);
  });

  it('drops a pending seek when another day opens', () => {
    const { engine, canvas } = setup();
    engine.jump(D0 + 37800);
    let frame: FrameRequestCallback | undefined;
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      frame = callback;
      return 99;
    });
    fireEvent.wheel(canvas, { deltaY: 100, shiftKey: true });
    engine.pickDay('2026-10-28');
    const jump = vi.spyOn(engine, 'jump');
    act(() => frame?.(100));
    expect(jump).not.toHaveBeenCalled();
  });

  it('does not consume an empty Shift wheel event', () => {
    const { canvas } = setup();
    expect(fireEvent.wheel(canvas, { shiftKey: true, deltaY: 0, deltaX: 0 })).toBe(true);
  });

  it('keeps the page from scrolling while it zooms', () => {
    const { canvas } = setup();
    expect(fireEvent.wheel(canvas, { deltaY: -100 })).toBe(false); // the default was prevented
  });
});

describe('timeline drag intent', () => {
  function ready() {
    const { engine, canvas } = setup();
    const track = canvas.closest('[role="slider"]');
    if (!track) throw new Error('no timeline track');
    Object.defineProperty(track, 'clientWidth', { configurable: true, value: 1000 });
    engine.jump(D0 + 37800);
    return { engine, track };
  }

  it.each(['mouse', 'touch'])('pans the time ruler without moving the cursor for %s', pointerType => {
    const { engine, track } = ready();
    const before = { ...engine.view };
    const cursor = engine.t;
    fireEvent.pointerDown(track, { pointerId: 1, pointerType, clientX: 100, clientY: 4 });
    fireEvent.pointerMove(track, { pointerId: 1, pointerType, clientX: 200, clientY: 4 });
    fireEvent.pointerUp(track, { pointerId: 1, pointerType, clientX: 200, clientY: 4 });
    expect(engine.view).toEqual({ t0: before.t0 - 360, t1: before.t1 - 360 });
    expect(engine.t).toBe(cursor);
  });

  it.each(['mouse', 'touch'])('moves the cursor across the body without panning for %s', pointerType => {
    const { engine, track } = ready();
    const before = { ...engine.view };
    fireEvent.pointerDown(track, { pointerId: 1, pointerType, clientX: 100, clientY: 50 });
    fireEvent.pointerMove(track, { pointerId: 1, pointerType, clientX: 200, clientY: 50 });
    fireEvent.pointerUp(track, { pointerId: 1, pointerType, clientX: 200, clientY: 50 });
    expect(engine.t).toBe(before.t0 + 720);
    expect(engine.view).toEqual(before);
  });

  it('pans with a middle drag over the lanes, and never seeks', () => {
    const { engine, track } = ready();
    const before = { ...engine.view };
    const cursor = engine.t;
    const down = new PointerEvent('pointerdown', {
      pointerId: 1,
      pointerType: 'mouse',
      button: 1,
      clientX: 100,
      clientY: 50,
      bubbles: true,
      cancelable: true,
    });
    track.dispatchEvent(down);
    expect(down.defaultPrevented).toBe(true);
    fireEvent.pointerMove(track, { pointerId: 1, pointerType: 'mouse', buttons: 4, clientX: 200, clientY: 50 });
    fireEvent.pointerUp(track, { pointerId: 1, pointerType: 'mouse', button: 1, clientX: 200, clientY: 50 });
    expect(engine.view).toEqual({ t0: before.t0 - 360, t1: before.t1 - 360 });
    expect(engine.t).toBe(cursor);
  });

  it('cancels a middle press, so the browser never starts its autoscroll', () => {
    const { track } = ready();
    expect(fireEvent.mouseDown(track, { button: 1 })).toBe(false);
    expect(fireEvent(track, new MouseEvent('auxclick', { button: 1, bubbles: true, cancelable: true }))).toBe(false);
    expect(fireEvent.mouseDown(track, { button: 0 })).toBe(true); // a left press is left alone
  });

  it('leaves a middle click in place: no seek', () => {
    const { engine, track } = ready();
    const cursor = engine.t;
    fireEvent.pointerDown(track, { pointerId: 1, pointerType: 'mouse', button: 1, clientX: 100, clientY: 50 });
    fireEvent.pointerUp(track, { pointerId: 1, pointerType: 'mouse', button: 1, clientX: 100, clientY: 50 });
    expect(engine.t).toBe(cursor);
  });
});

describe('the days still loading under the timeline', () => {
  it('keeps one status line when nothing is loading, so a day arriving never moves the page', () => {
    renderPage(<ChunkLine chunkStatus={{ '2026-11-05': 'ready' }} range={null} />);
    expect(screen.getByRole('status')).toHaveTextContent(/^$/);
  });

  it('names every loading and failed day in view on that one line, as people say them', () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date(2026, 10, 5, 15));
    renderPage(
      <ChunkLine
        chunkStatus={{
          '2026-11-05': 'loading',
          '2026-11-04': 'loading',
          '2026-11-03': 'error',
          '2026-11-01': 'loading',
        }}
        range={{ t0: new Date(2026, 10, 3, 12).getTime() / 1000, t1: new Date(2026, 10, 5, 12).getTime() / 1000 }}
      />,
    );
    expect(screen.getAllByRole('status')).toHaveLength(1);
    expect(screen.getByRole('status')).toHaveTextContent(/^Loading today and yesterday\. Could not load \D*3\D*\.$/);
    vi.useRealTimers();
  });
});
