import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readPrefs, writePrefs } from '../../shared/prefs';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { hms } from '../lib/format';
import { usePrefs } from '../lib/prefs';
import { EngineContext, useEngine, useReview } from './context';
import { EventForm, parseClock, parseNotes } from './EventForm';
import { ReviewEngine } from './engine';
import { Timeline, TimelineTools } from './Timeline';

const DAY = '2026-10-27';
const D0 = new Date(2026, 9, 27).getTime() / 1000;
const W = 1000; // the track's width: jsdom lays nothing out, so each test gives it this

let marked: unknown[];
const paint = { clearRect: vi.fn(), fillRect: vi.fn(), fillText: vi.fn() };
const engines: ReviewEngine[] = [];
afterEach(() => {
  for (const engine of engines.splice(0)) engine.stop();
});

/** Review's timeline and the event form, wired as Review wires them. */
function Marking() {
  const engine = useEngine();
  const span = useReview(s => s.eventSpan);
  return (
    <>
      <Timeline onSpan={s => engine.markEventSpan(s)} />
      <TimelineTools mark />
      <EventForm span={span} onClose={() => engine.closeEvent()} />
    </>
  );
}

function open(mode: 'compact' | 'expanded' = 'compact') {
  writePrefs({ ...readPrefs(), reviewTimeline: mode });
  usePrefs.setState(readPrefs(), true);
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.start();
  engines.push(engine);
  engine.pickDay(DAY);
  const gen = engine.ui.getState().gen;
  engine.setDayData(DAY, gen, { entries: [], segments: [], spans: [] });
  engine.setChips(DAY, gen, []);
  const page = renderPage(
    <EngineContext value={engine}>
      <Marking />
    </EngineContext>,
  );
  const track = page.container.querySelector('[role="slider"]') as HTMLElement;
  Object.defineProperty(track, 'clientWidth', { configurable: true, value: W });
  const label = [...page.container.querySelectorAll('span[style*="top"]')].find(l => l.textContent === 'cam0');
  const camLane = Number.parseFloat((label as HTMLElement | undefined)?.style.top ?? '20') + 6;
  engine.setView(D0 + 3600, D0 + 3700); // 100 s across 1,000 px: 10 px a second
  const x = (ts: number) => ((ts - engine.view.t0) / (engine.view.t1 - engine.view.t0)) * W;
  const drag = (from: number, to: number, opts: { shiftKey?: boolean } = {}) => {
    fireEvent.pointerDown(track, { pointerId: 1, clientX: x(from), clientY: camLane, ...opts });
    fireEvent.pointerMove(track, { pointerId: 1, clientX: x(to), clientY: camLane, ...opts });
    fireEvent.pointerUp(track, { pointerId: 1, clientX: x(to), clientY: camLane, ...opts });
  };
  const mark = (from: number, to: number) => {
    engine.jump(from);
    fireEvent.keyDown(track, { key: 'i' });
    engine.jump(to);
    fireEvent.keyDown(track, { key: 'o' });
  };
  // A drag at any height on the track (not just a camera's lane), for the pointer tool.
  const dragAt = (from: number, to: number, y: number) => {
    fireEvent.pointerDown(track, { pointerId: 1, clientX: x(from), clientY: y });
    fireEvent.pointerMove(track, { pointerId: 1, clientX: x(to), clientY: y });
    fireEvent.pointerUp(track, { pointerId: 1, clientX: x(to), clientY: y });
  };
  return { engine, track, drag, dragAt, mark, page };
}

beforeEach(() => {
  marked = [];
  paint.clearRect.mockClear();
  paint.fillRect.mockClear();
  paint.fillText.mockClear();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(paint as unknown as CanvasRenderingContext2D);
  server.use(
    http.get('/speakers', () => HttpResponse.json({ speakers: [], merged: {} })),
    http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }] })),
    http.post('/measurement/mark', async ({ request }) => {
      marked.push(await request.json());
      return HttpResponse.json({ ok: true });
    }),
  );
});

describe('marking an event (#287, #518)', () => {
  it('marks with the pointer tool from anywhere on the timeline, with a crosshair, for the shown camera', async () => {
    const { track, dragAt, page } = open();
    await userEvent.click(within(page.container).getByRole('button', { name: 'Mark an event' }));
    expect(track).toHaveAttribute('data-marking');
    dragAt(D0 + 3610, D0 + 3650, 2); // on the time ruler, above every lane
    const form = await screen.findByRole('form', { name: 'Mark an event' });
    expect(within(form).getByLabelText(/^From/)).toHaveValue(hms(D0 + 3610));
    expect(within(form).getByLabelText(/^To/)).toHaveValue(hms(D0 + 3650));
    expect(within(form).getByLabelText('Camera')).toHaveValue('cam0');
    expect(track).not.toHaveAttribute('data-marking');
  });

  it('opens the form for I and O, saves the label, and clears the marks', async () => {
    const { engine, mark } = open();
    mark(D0 + 3610, D0 + 3650);
    const form = await screen.findByRole('form', { name: 'Mark an event' });
    expect(within(form).getByLabelText(/^From/)).toHaveValue(hms(D0 + 3610));
    expect(within(form).getByLabelText(/^To/)).toHaveValue(hms(D0 + 3650));
    expect(within(form).getByLabelText('Camera')).toHaveValue('cam0');
    await userEvent.type(within(form).getByRole('combobox', { name: /^Who/ }), 'Da');
    await userEvent.click(await screen.findByRole('option', { name: 'Dana' }));
    expect(marked).toEqual([]); // choosing a person does not submit the event
    await userEvent.type(within(form).getByLabelText(/^What happened/), 'opened the fridge');
    await userEvent.clear(within(form).getByLabelText(/^How many times/));
    await userEvent.type(within(form).getByLabelText(/^How many times/), '3');
    await userEvent.click(within(form).getByRole('button', { name: 'Add a measurement' }));
    await userEvent.type(within(form).getByLabelText('Measurement'), 'door open');
    await userEvent.type(within(form).getByLabelText('Value'), '12');
    await userEvent.type(within(form).getByLabelText('Unit'), 's');
    await userEvent.type(within(form).getByLabelText(/^Notes/), '2: only half in view{Enter}light on');
    await userEvent.click(within(form).getByRole('button', { name: 'Save' }));
    expect(await within(form).findByRole('status')).toHaveTextContent('Saved: Dana, opened the fridge, 3 times.');
    expect(engine.ui.getState().eventMarks).toBeNull();
    expect(marked).toEqual([
      {
        kind: 'event',
        person: 'Dana',
        camera: 'cam0',
        what: 'opened the fridge',
        start: D0 + 3610,
        end: D0 + 3650,
        count: 3,
        attributes: { 'door open': { value: 12, unit: 's' } },
        notes: [{ nth: 2, note: 'only half in view' }, { note: 'light on' }],
      },
    ]);
  });

  it('saves an event that involved no one, once by default', async () => {
    const { engine, mark } = open();
    mark(D0 + 3610, D0 + 3650);
    const form = await screen.findByRole('form', { name: 'Mark an event' });
    await userEvent.type(within(form).getByLabelText(/^What happened/), 'the door was left open');
    await userEvent.click(within(form).getByRole('button', { name: 'Save' }));
    expect(await within(form).findByRole('status')).toHaveTextContent('Saved: the door was left open, 1 time.');
    expect(engine.ui.getState().eventMarks).toBeNull();
    expect(marked).toEqual([
      expect.objectContaining({ kind: 'event', what: 'the door was left open', count: 1, attributes: {}, notes: [] }),
    ]);
    expect(marked[0]).not.toHaveProperty('person');
  });

  it.each(['compact', 'expanded'] as const)('seeks and marks with a drag in %s mode', async mode => {
    const { engine, drag } = open(mode);
    const before = { ...engine.view };
    drag(D0 + 3610, D0 + 3650);
    expect(engine.t).toBe(D0 + 3650);
    expect(engine.view).toEqual(before); // scrubbing does not pan
    expect(screen.queryByRole('form', { name: 'Mark an event' })).not.toBeInTheDocument();
    const mark = screen.getByRole('button', { name: 'Mark an event' });
    await userEvent.click(mark);
    expect(mark).toHaveAttribute('aria-pressed', 'true');
    const view = { ...engine.view };
    drag(view.t0 + 10, view.t0 + 30);
    expect(await screen.findByRole('form', { name: 'Mark an event' })).toBeInTheDocument();
    expect(engine.view).toEqual(view); // marking doesn't move the view
    expect(mark).toHaveAttribute('aria-pressed', 'false'); // one span at a time
  });

  it.each(['compact', 'expanded'] as const)('Shift-drag scrubs and Escape clears a partial mark in %s mode', mode => {
    const { engine, track, drag } = open(mode);
    drag(D0 + 3610, D0 + 3650, { shiftKey: true });
    expect(engine.t).toBe(D0 + 3650);
    expect(screen.queryByRole('form', { name: 'Mark an event' })).not.toBeInTheDocument();
    fireEvent.keyDown(track, { key: 'i' });
    expect(engine.ui.getState().eventMarks?.in).toBe(D0 + 3650);
    fireEvent.keyDown(track, { key: 'Escape' });
    expect(engine.ui.getState().eventMarks).toBeNull();
  });

  it('opens reversed marks in time order, and Cancel clears the range', async () => {
    const { engine, mark } = open();
    mark(D0 + 3650, D0 + 3610);
    const form = await screen.findByRole('form', { name: 'Mark an event' });
    expect(within(form).getByLabelText(/^From/)).toHaveValue(hms(D0 + 3610));
    expect(within(form).getByLabelText(/^To/)).toHaveValue(hms(D0 + 3650));
    await userEvent.click(within(form).getByRole('button', { name: 'Cancel' }));
    expect(engine.ui.getState().eventMarks).toBeNull();
    expect(screen.queryByRole('form', { name: 'Mark an event' })).not.toBeInTheDocument();
  });

  it("turns the bar's Mark off with Escape, even with focus on the button", async () => {
    open();
    const mark = screen.getByRole('button', { name: 'Mark an event' });
    await userEvent.click(mark);
    expect(mark).toHaveAttribute('aria-pressed', 'true');
    await userEvent.keyboard('{Escape}');
    expect(mark).toHaveAttribute('aria-pressed', 'false');
  });

  it('paints the range and both ruler handles, and Escape closes the form and clears them', async () => {
    const { engine, mark } = open();
    mark(D0 + 3610, D0 + 3650);
    await screen.findByRole('form', { name: 'Mark an event' });
    engine.frame(0);
    expect(paint.fillRect).toHaveBeenCalledWith(100, expect.any(Number), 400, expect.any(Number));
    expect(paint.fillText).toHaveBeenCalledWith('I', 105, expect.any(Number));
    expect(paint.fillText).toHaveBeenCalledWith('O', 505, expect.any(Number));
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('form', { name: 'Mark an event' })).not.toBeInTheDocument());
    expect(engine.ui.getState().eventMarks).toBeNull();
    paint.fillRect.mockClear();
    engine.frame(0);
    expect(paint.fillRect).not.toHaveBeenCalledWith(100, expect.any(Number), 400, expect.any(Number));
  });

  it('says what is wrong instead of saving: a span that ends first, or no count', async () => {
    const { mark } = open();
    mark(D0 + 3610, D0 + 3650);
    const form = await screen.findByRole('form', { name: 'Mark an event' });
    await userEvent.type(within(form).getByLabelText(/^What happened/), 'sat down');
    await userEvent.clear(within(form).getByLabelText(/^How many times/));
    fireEvent.submit(form);
    expect(await within(form).findByRole('alert')).toHaveTextContent('Count how many times it happened');
    await userEvent.clear(within(form).getByLabelText(/^To/));
    await userEvent.type(within(form).getByLabelText(/^To/), hms(D0 + 3600));
    await userEvent.type(within(form).getByLabelText(/^How many times/), '2');
    fireEvent.submit(form);
    await waitFor(() => expect(within(form).getByRole('alert')).toHaveTextContent('The span ends after it starts.'));
    expect(marked).toEqual([]);
  });
});

describe('parseClock and parseNotes', () => {
  it('reads a time on the same day, or nothing that is not one', () => {
    expect(parseClock('01:00:10', D0 + 3600)).toBe(D0 + 3610);
    expect(parseClock('1:00', D0 + 3600)).toBe(D0 + 3600);
    expect(parseClock('25:00:00', D0)).toBeNull();
    expect(parseClock('soon', D0)).toBeNull();
  });

  it('reads one note a line, with which time a line starts with', () => {
    expect(parseNotes('2: only half in view\n\n  light on  \n10:last one quick')).toEqual([
      { nth: 2, note: 'only half in view' },
      { note: 'light on' },
      { nth: 10, note: 'last one quick' },
    ]);
    expect(parseNotes('18:05 came back')).toEqual([{ note: '18:05 came back' }]); // a time, not the 18th
  });
});
