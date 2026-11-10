import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { createStore } from 'zustand/vanilla';
import { newQueryClient, Providers } from '../../shared/Providers';
import { server } from '../../test/server';
import type { TranscriptEntry } from '../live/types';
import { HEAD_PX, LINE_PX, OVERSCAN } from '../transcript/Transcript';
import { EngineContext } from './context';
import type { ReviewEngine, ReviewUi, TranscriptHooks } from './engine';
import { Transcript } from './Transcript';

// jsdom has no layout: a turn is its header and then a LINE_PX row per line (the list's own estimate, so turns not
// yet measured sit where they will), each where the list puts it (its translateY), and the list shows VIEW px. A
// scroll set by the list fires its scroll event, as a browser's does.
const TURN = HEAD_PX + LINE_PX; // these tests' turns are one line each
const VIEW = 100;
const CENTRED = (VIEW - LINE_PX) / 2; // a centred line's top
const scroll = new WeakMap<Element, number>();
const isList = (el: Element) => el.getAttribute('aria-label') === 'Transcript of the day';
const listOf = (el: Element) => el.closest('[aria-label="Transcript of the day"]');
const height = (list: Element) =>
  Number.parseFloat(list.querySelector<HTMLElement>('[data-turns]')?.style.height ?? '0') || 0;
const slotTop = (slot: Element | null) =>
  Number.parseFloat(/translateY\((-?[\d.]+)px\)/.exec((slot as HTMLElement | null)?.style.transform ?? '')?.[1] ?? '0');

const line = (i: number, ts = 1000 + i * 5): TranscriptEntry => ({
  ts_start: ts,
  ts_end: ts + 2,
  line: `l${ts}`,
  text: `line ${i}`,
  speaker_id: (i & 1) + 1, // a turn per line: alternating voices
});
const lines = (n: number) => Array.from({ length: n }, (_, i) => line(i));

let hooks: TranscriptHooks | null;
let attached: (boolean | undefined)[];
let current: number; // the engine's current line, marked on every attach
/** The engine moving its current line (playback or a jump), as setCur hears it. */
const moveTo = (i: number, centre: boolean) =>
  act(() => {
    current = i;
    hooks?.setCur(i, centre);
  });
let engine: ReviewEngine;

beforeEach(() => {
  hooks = null;
  attached = [];
  current = 0;
  const ui = createStore<ReviewUi>()(() => ({ day: '2026-11-04', person: null }) as unknown as ReviewUi);
  engine = {
    ui,
    seekPlay: vi.fn(),
    attachTranscript(t: TranscriptHooks | null, centre?: boolean) {
      hooks = t;
      if (!t) return;
      attached.push(centre);
      t.setCur(current, centre ?? true);
    },
  } as unknown as ReviewEngine;
  server.use(
    http.get('/speakers', () => HttpResponse.json({ speakers: [], merged: {} })),
    http.get('/people', () => HttpResponse.json({ people: [] })),
  );
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function (this: HTMLElement) {
    return isList(this) ? VIEW : 0;
  });
  vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(function (this: HTMLElement) {
    return isList(this) ? height(this) : 0;
  });
  vi.spyOn(Element.prototype, 'scrollTop', 'get').mockImplementation(function (this: Element) {
    return scroll.get(this) ?? 0;
  });
  vi.spyOn(Element.prototype, 'scrollTop', 'set').mockImplementation(function (this: Element, v: number) {
    const to = Math.max(0, Math.min(v, height(this) - VIEW));
    if (to === (scroll.get(this) ?? 0)) return;
    scroll.set(this, to);
    queueMicrotask(() => this.dispatchEvent(new Event('scroll')));
  });
  // The list measures each turn as it's observed: as tall as its lines.
  vi.stubGlobal(
    'ResizeObserver',
    class {
      readonly cb: ResizeObserverCallback;
      constructor(cb: ResizeObserverCallback) {
        this.cb = cb;
      }
      observe(el: Element) {
        if (!el.hasAttribute('data-index')) return;
        const blockSize = HEAD_PX + el.querySelectorAll('[data-key]').length * LINE_PX;
        const entry = { target: el, borderBoxSize: [{ blockSize, inlineSize: 300 }] } as unknown as ResizeObserverEntry;
        queueMicrotask(() => this.cb([entry], this as unknown as ResizeObserver));
      }
      unobserve() {}
      disconnect() {}
    },
  );
  vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockImplementation(function (this: HTMLElement) {
    return this.hasAttribute('data-index') ? HEAD_PX + this.querySelectorAll('[data-key]').length * LINE_PX : 0;
  });
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    const list = listOf(this);
    if (!list || isList(this)) return DOMRect.fromRect({ x: 0, y: 0, width: 300, height: VIEW });
    const slot = this.closest('[data-index]');
    const at = slot ? [...slot.querySelectorAll('[data-key]')].indexOf(this) : 0;
    const y = slotTop(slot) + HEAD_PX + at * LINE_PX - (scroll.get(list) ?? 0);
    return DOMRect.fromRect({ x: 0, y, width: 300, height: LINE_PX });
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  server.resetHandlers();
});

/** Let queued scroll events, and the renders they cause, run. */
const settle = () => act(async () => {});

function show(first: TranscriptEntry[]) {
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false } });
  const ui = (entries: TranscriptEntry[]) => (
    <Providers client={client} test>
      <EngineContext.Provider value={engine}>
        <Transcript entries={entries} />
      </EngineContext.Provider>
    </Providers>
  );
  const view = render(ui(first));
  const list = screen.getByRole('region', { name: 'Transcript of the day' });
  return {
    list,
    update: (entries: TranscriptEntry[]) => view.rerender(ui(entries)),
    scrollTo: async (top: number) => {
      list.scrollTop = top;
      fireEvent.scroll(list);
      await settle();
    },
  };
}

it('holds its place when new lines land while scrolled away, and says how many (#2286)', async () => {
  const { list, update, scrollTo } = show(lines(20));
  await settle();
  expect(attached).toEqual([true]); // shown: the playhead's line is centred once
  await scrollTo(100);
  update(lines(23));
  await settle();
  expect(list.scrollTop).toBe(100);
  expect(attached.at(-1)).toBe(false); // the same list re-rendered: marked again, not centred
  await userEvent.click(screen.getByRole('button', { name: '3 new lines' }));
  await settle();
  expect(list.scrollTop).toBe(23 * TURN - VIEW);
  expect(screen.queryByRole('button', { name: /new line/ })).toBeNull();
});

it('keeps the same line at the top when a line lands above it', async () => {
  const { list, update, scrollTo } = show(lines(20));
  await scrollTo(5 * TURN); // line 5's turn at the top
  current = 1; // the playhead's line is one further down the day's lines now
  update([line(-1, 990), ...lines(20)]);
  await settle();
  expect(list.scrollTop).toBe(6 * TURN);
  expect(screen.getByText('line 5').closest('[data-key]')?.getBoundingClientRect().top).toBe(HEAD_PX);
});

it('follows the newest line at the live end', async () => {
  const { list, update, scrollTo } = show(lines(20));
  await scrollTo(20 * TURN - VIEW);
  update(lines(23));
  await settle();
  expect(list.scrollTop).toBe(23 * TURN - VIEW);
  expect(screen.queryByRole('button', { name: /new line/ })).toBeNull();
});

it("scrolls to the playhead's line only when it moves to another line, or after a jump", async () => {
  const { list, scrollTo } = show(lines(20));
  await scrollTo(400);
  moveTo(0, false); // the same line marked again after a refresh
  await settle();
  expect(list.scrollTop).toBe(400);
  moveTo(1, false); // playback reached the next line, out of view
  await settle();
  expect(list.scrollTop).toBe(TURN + HEAD_PX - CENTRED);
  await scrollTo(400);
  moveTo(1, true); // a jump
  await settle();
  expect(list.scrollTop).toBe(TURN + HEAD_PX - CENTRED);
});

it('mounts only the turns in view and a few either side, and the rest as they scroll near (#2382)', async () => {
  const { scrollTo } = show(lines(500));
  await settle();
  const most = Math.ceil(VIEW / TURN) + 1 + 2 * OVERSCAN;
  const mounted = () => document.querySelectorAll('[data-key]').length;
  expect(mounted()).toBeLessThanOrEqual(most);
  expect(screen.queryByText('line 400')).toBeNull();
  await scrollTo(400 * TURN);
  expect(screen.getByText('line 400')).toBeInTheDocument();
  expect(mounted()).toBeLessThanOrEqual(most);
});

it("marks and centres the playhead's line when a jump lands on one scrolled far away", async () => {
  const { list } = show(lines(500));
  await settle();
  moveTo(300, true);
  const row = (await screen.findByText('line 300')).closest('[data-key]');
  await waitFor(() => expect(row?.getBoundingClientRect().top).toBe(CENTRED));
  expect(row).toHaveAttribute('aria-current', 'time');
  expect(list.scrollTop).toBe(300 * TURN + HEAD_PX - CENTRED);
});

it('keeps a hidden dense day bounded until its turns have layout (#3290)', async () => {
  // Review's inactive Mantine panel is display:none: connected nodes measure zero.
  vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(0);
  const observed = new Map<Element, ResizeObserverCallback>();
  vi.stubGlobal(
    'ResizeObserver',
    class {
      readonly cb: ResizeObserverCallback;
      constructor(cb: ResizeObserverCallback) {
        this.cb = cb;
      }
      observe(el: Element) {
        observed.set(el, this.cb);
      }
      unobserve(el: Element) {
        observed.delete(el);
      }
      disconnect() {
        observed.clear();
      }
    },
  );
  const resize = (size: number) =>
    act(() => {
      for (const [el, cb] of [...observed]) {
        if (!el.hasAttribute('data-index')) continue;
        cb(
          [{ target: el, borderBoxSize: [{ blockSize: size, inlineSize: 300 }] } as unknown as ResizeObserverEntry],
          {} as ResizeObserver,
        );
      }
    });
  current = 500;
  const { list } = show(lines(1000));
  await settle();
  expect(list).toBeInTheDocument();
  expect(document.querySelectorAll('[data-index]').length).toBeLessThan(30);
  expect(height(list)).toBe(1000 * TURN);
  resize(0); // hidden ResizeObserver delivery must not collapse the estimates either
  await settle();
  expect(height(list)).toBe(1000 * TURN);
  resize(80); // showing the panel measures real rows instead of retaining their estimates forever
  await settle();
  expect(height(list)).toBeGreaterThan(1000 * TURN);
});

it('keeps a turn being named mounted while scrolled away', async () => {
  const { scrollTo } = show(lines(200));
  await settle();
  const [first] = screen.getAllByRole('button', { name: 'Unmatched: say who this is' });
  if (first) await userEvent.click(first);
  expect(await screen.findByRole('combobox')).toBeInTheDocument();
  await scrollTo(150 * TURN);
  expect(screen.getByText('line 150')).toBeInTheDocument();
  expect(screen.getByRole('combobox')).toBeInTheDocument();
});
