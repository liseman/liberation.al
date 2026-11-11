import { onlineManager } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { Activity, type ComponentProps, useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { newQueryClient, Providers } from '../../shared/Providers';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { AskBar } from '../AskBar';
import { useAsk, useAskPage } from '../ask/openAsk';
import { jobServer } from '../ask/testing';
import { dayWords, hm, hms, isoDay } from '../lib/format';
import { useRoute } from '../router';
import { reviewKey, stillComputing, working } from './api';
import { ReviewEngine } from './engine';
import { daySegments } from './gatewaySegments';
import { Review } from './Review';
import { REVIEW_KEYS } from './reviewKeys';

// The day's video comes from Continuum's gateway: its list is mocked per test, and each canvas's player is a stand-in
// that behaves as jsdom's <video> did for these tests (it takes a src and reports metadata, but never seeks or plays),
// with every one made kept in `players`.
const players = vi.hoisted(() => [] as { getAttribute(name: string): string | null; src: string }[]);
const citedMarks = vi.hoisted(() => ({ current: [] as readonly number[] }));
vi.mock('./Timeline', async original => {
  const actual = await original<typeof import('./Timeline')>();
  return {
    ...actual,
    Timeline: (props: ComponentProps<typeof actual.Timeline>) => {
      citedMarks.current = props?.marks ?? [];
      return <actual.Timeline {...props} />;
    },
  };
});
vi.mock('./gatewaySegments', async original => ({
  ...(await original<typeof import('./gatewaySegments')>()),
  daySegments: vi.fn(),
}));
vi.mock('./gatewayVideo', async orig => {
  class QuietVideo extends EventTarget {
    preload = 'auto';
    playbackRate = 1;
    defaultPlaybackRate = 1;
    readyState = 0;
    duration = Number.NaN;
    error = null;
    seeking = false;
    paused = true;
    ended = false;
    private url = '';
    private at = 0;
    readonly canvas: HTMLCanvasElement;
    constructor(canvas: HTMLCanvasElement) {
      super();
      this.canvas = canvas;
      players.push(this);
    }
    get src() {
      return this.url;
    }
    set src(u: string) {
      this.url = u;
      this.readyState = u ? 1 : 0;
      if (u) queueMicrotask(() => this.dispatchEvent(new Event('loadedmetadata')));
    }
    get currentTime() {
      return this.at;
    }
    set currentTime(t: number) {
      this.at = t;
    }
    play() {
      return Promise.resolve();
    }
    pause() {}
    load() {}
    close() {}
    getAttribute(name: string) {
      return name === 'src' ? this.url || null : this.canvas.getAttribute(name);
    }
    removeAttribute(name: string) {
      if (name === 'src') this.url = '';
      else this.canvas.removeAttribute(name);
    }
    toggleAttribute(name: string, force?: boolean) {
      return this.canvas.toggleAttribute(name, force);
    }
  }
  return { ...(await orig<typeof import('./gatewayVideo')>()), GatewayVideo: QuietVideo };
});

const DAY = '2026-10-27';
const D0 = new Date(2026, 9, 27).getTime() / 1000;
const at = (s: number) => D0 + s;

const ENTRIES = [
  { ts_start: at(36010), ts_end: at(36013), text: 'Morning, Dana.', speaker: 'Alice', speaker_id: 1 },
  { ts_start: at(36015), ts_end: at(36018), text: 'Coffee is ready.', speaker: 'Alice', speaker_id: 1 },
  { ts_start: at(36400), ts_end: at(36404), text: 'Where did the dog go?', speaker: 'Dana', speaker_id: 2 },
];
const SEGMENTS = [
  { cam: 'cam0', url: 'gateway:1:1', start: at(36000), end: at(36300) },
  { cam: 'cam0', url: 'gateway:1:2', start: at(36300), end: at(36600) },
];
const CHIPS = [
  { id: 'person:dana', kind: 'person', label: 'Dana', spans: [[at(36010), at(36020)]], total_s: 10, count: 1 },
  {
    id: 'object:dog',
    kind: 'object',
    label: 'dog',
    spans: [
      [at(36100), at(36130)],
      [at(36500), at(36520)],
    ],
    total_s: 50,
    count: 2,
  },
];

const OTHER_DAY = '2026-10-26';
const RECORDED = [
  { day: '2026-10-20', video: false, audio: true, transcript: false },
  { day: OTHER_DAY, video: true, audio: false, transcript: true },
  { day: DAY, video: true, audio: true, transcript: true },
];
let calls: Record<string, number>;
let hold: Promise<void> = Promise.resolve(); // transcript responses wait for this

function serve({ notes = 200 }: { notes?: 200 | 403 } = {}) {
  const count = (k: string) => {
    calls[k] = (calls[k] ?? 0) + 1;
  };
  server.use(
    http.get('/transcript/day', async ({ request }) => {
      const day = new URL(request.url).searchParams.get('day') ?? DAY;
      count('transcript');
      count(`transcript:${day}`);
      await hold;
      const offset = new Date(`${day}T00:00:00`).getTime() / 1000 - D0;
      return HttpResponse.json({
        day,
        entries: ENTRIES.map(e => ({ ...e, ts_start: e.ts_start + offset, ts_end: e.ts_end + offset })),
      });
    }),
    http.get('/audio/spans', ({ request }) => {
      const day = new URL(request.url).searchParams.get('day') ?? DAY;
      const start = new Date(`${day}T00:00:00`).getTime() / 1000;
      return HttpResponse.json({ day, spans: [{ start: start + 36000, end: start + 36600 }] });
    }),
    http.get('/review/presence', ({ request }) => {
      const day = new URL(request.url).searchParams.get('day') ?? DAY;
      const offset = new Date(`${day}T00:00:00`).getTime() / 1000 - D0;
      return HttpResponse.json({
        chips: CHIPS.map(c => ({ ...c, spans: c.spans.map(([a, b]) => [(a ?? 0) + offset, (b ?? 0) + offset]) })),
      });
    }),
    http.get('/review/boxes', () => HttpResponse.json({ boxes: [] })), // the large camera's boxes
    http.get('/review/summary', ({ request }) => {
      const q = new URL(request.url).searchParams; // the summary by time: nothing in this stretch
      return HttpResponse.json({
        start: Number(q.get('start')),
        end: Number(q.get('end')),
        level: 3600,
        status: 'ready',
        blocks: [],
      });
    }),
    http.get('/summary', () => {
      count('summary');
      return HttpResponse.json({ day: DAY, md: 'Legacy day summary', status: 'ready' });
    }),
    http.get('/history/at', ({ request }) =>
      notes === 403
        ? HttpResponse.json({ error: 'camera history is only shown on this machine' }, { status: 403 })
        : HttpResponse.json({
            ts: Number(new URL(request.url).searchParams.get('ts')),
            cams: { cam0: { ts: at(36011), dt: 1, caption: 'Two people at the counter', faces: [{ name: 'Dana' }] } },
          }),
    ),
  );
}

function open(hash = `#day=${DAY}&t=${at(36011)}`) {
  history.replaceState(null, '', `/review${hash}`);
  return renderPage(
    <>
      <AskBar tab="review" setup={false} />
      <Review />
    </>,
  );
}

const longDay = (day: string) =>
  new Date(`${day}T12:00:00`).toLocaleDateString(undefined, {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });

/** Pick a day from the calendar, as a person would. */
async function chooseDay(day: string) {
  await userEvent.click(screen.getByLabelText(/^Day\b/));
  await userEvent.click(await screen.findByRole('button', { name: new RegExp(`^${longDay(day)}`) }));
}

/** The day the picker shows, as it shows it ("Tue 27 Oct 2026"). */
const shownDay = () => screen.getByLabelText(/^Day\b/).textContent;
/** The day's lines, video, and audio have loaded: the status line no longer says it's loading. */
const dayLoaded = () =>
  waitFor(() => {
    expect(screen.queryByText(/Loading the day/)).toBeNull();
    expect(screen.getByRole('slider', { name: /^Timeline/ })).not.toHaveAttribute('aria-busy');
  });
const shown = (day: string) =>
  new Date(`${day}T12:00:00`)
    .toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })
    .replace(/,/g, '');

const timeline = () => screen.getByRole('slider', { name: /Timeline/ });
const playheadAt = async (ts: number) =>
  waitFor(() => expect(timeline()).toHaveAttribute('aria-valuetext', `${isoDay(new Date(ts * 1000))} ${hms(ts)}`));

beforeEach(() => {
  server.use(http.get('/ambient', () => HttpResponse.json({ devices: {} })));
  useAsk.setState({ open: false, request: null });
  useAskPage.setState({ day: null, seek: null, mark: null });
  useRoute.setState({ tab: 'review' });
  citedMarks.current = [];
  server.use(http.get('/activities', () => HttpResponse.json({ items: [], truncated: false })));
  hold = Promise.resolve();
  players.length = 0;
  vi.mocked(daySegments).mockReset();
  vi.mocked(daySegments).mockImplementation(async start =>
    SEGMENTS.map(s => ({ ...s, start: s.start + start - D0, end: s.end + start - D0, url: `${s.url}:${start}` })),
  );
  server.use(
    http.get('/speakers', () => HttpResponse.json({ speakers: [], merged: {} })),
    // Voice naming refreshes the shared Live snapshot four seconds later, even after its originating turn closes.
    http.get('/transcript', () => HttpResponse.json({ entries: [], size: 0, reset: true })),
    http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }, { name: 'Kara' }] })), // the name boxes
    http.get('/review/days', () => HttpResponse.json({ days: RECORDED, today: DAY })),
    http.get('/audio/range', () => new HttpResponse(new Blob(['room']), { headers: { 'Content-Type': 'audio/wav' } })),
    http.get('/clip', () => new HttpResponse(new Blob(['line']), { headers: { 'Content-Type': 'audio/wav' } })),
    http.post('/playback/start', () => HttpResponse.json({ ok: true, token: 'tok' })),
    http.post('/playback/stop', () => HttpResponse.json({ ok: true })),
  );
  calls = {};
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Alice', cams: ['cam0'], hotkeys: false });
  // jsdom can't play media; the engine's own tests cover what it asks of the <video>s.
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(() => Promise.resolve());
  Object.assign(URL, { createObjectURL: () => 'blob:clip', revokeObjectURL: () => {} });
});
afterEach(() => {
  document.getElementById('test-dashboard-topbar')?.remove();
  vi.unstubAllGlobals();
  history.replaceState(null, '', '/');
});

describe('the days with recordings', () => {
  it('opens the latest recorded day when today has nothing', async () => {
    vi.useFakeTimers({ toFake: ['Date'], now: new Date(2026, 9, 30, 9) });
    try {
      serve();
      open('');
      await waitFor(() => expect(shownDay()).toBe(shown(DAY)));
      expect(location.hash).toContain(`day=${DAY}`);
      await dayLoaded();
    } finally {
      vi.useRealTimers();
    }
  });

  it('steps with Previous day and Next day over the days with nothing recorded', async () => {
    serve();
    open(`#day=${OTHER_DAY}`);
    const prev = await screen.findByRole('button', { name: 'Previous day' });
    const next = screen.getByRole('button', { name: 'Next day' });
    await waitFor(() => expect(prev).toBeEnabled());
    await userEvent.click(prev);
    await waitFor(() => expect(shownDay()).toBe(shown('2026-10-20')));
    expect(prev).toBeDisabled(); // nothing recorded before it
    await userEvent.click(next);
    await userEvent.click(next);
    await waitFor(() => expect(shownDay()).toBe(shown(DAY)));
    expect(next).toBeDisabled();
  });

  it('marks the recorded days in the calendar, saying what each has', async () => {
    serve();
    open(`#day=${DAY}`);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next day' })).toBeDisabled()); // the days are in
    await userEvent.click(screen.getByLabelText(/^Day\b/));
    const recorded = await screen.findByRole('button', { name: `${longDay(OTHER_DAY)}, video and transcript` });
    expect(recorded).toHaveAttribute('data-recorded');
    expect(screen.getByRole('button', { name: longDay('2026-10-25') })).not.toHaveAttribute('data-recorded');
  });

  it('keeps the last list of days when a refresh fails', async () => {
    serve();
    const page = open(`#day=${DAY}`);
    const next = screen.getByRole('button', { name: 'Next day' });
    await waitFor(() => expect(next).toBeDisabled()); // nothing recorded after DAY
    server.use(http.get('/review/days', () => HttpResponse.json({ error: 'busy' }, { status: 500 })));
    await page.client.refetchQueries({ queryKey: ['review', 'days'] });
    await waitFor(() => expect(page.client.getQueryState(['review', 'days'])?.status).toBe('error'));
    await new Promise(r => setTimeout(r, 50)); // let the page draw the failed state
    expect(next).toBeDisabled(); // not one calendar day on, as if the days were unknown
  });

  it('steps one calendar day while the recorded days are unknown', async () => {
    serve();
    server.use(http.get('/review/days', () => HttpResponse.json({ error: 'boom' }, { status: 500 })));
    open(`#day=${OTHER_DAY}`);
    await userEvent.click(await screen.findByRole('button', { name: 'Next day' }));
    await waitFor(() => expect(shownDay()).toBe(shown(DAY)));
  });
});

describe('the room audio element', () => {
  it('stays attached to the engine across re-renders, so it keeps the room audio it holds', async () => {
    const attach = vi.spyOn(ReviewEngine.prototype, 'setAudio');
    serve();
    const { unmount } = open();
    await dayLoaded();
    await userEvent.click(screen.getByRole('button', { name: /Morning, Dana\./ })); // a re-render or more
    // Each attach forgets the room audio the element holds (loadedRoom), so it happens once, not once per render.
    expect(attach.mock.calls.filter(([el]) => el === null)).toEqual([]);
    expect(attach.mock.calls.filter(([el]) => el !== null)).toHaveLength(1);
    unmount();
    expect(attach).toHaveBeenLastCalledWith(null);
  });
});

describe('opening a day', () => {
  it('names the day picker with the day it shows, not only "Day"', async () => {
    serve();
    open();
    await waitFor(() => expect(shownDay()).toBe(shown(DAY)));
    expect(screen.getByLabelText(/^Day\b/)).toHaveAccessibleName(`Day, ${shown(DAY)}`);
  });

  it('shows the day from the link, at its moment', async () => {
    serve();
    open();
    await dayLoaded();
    expect(shownDay()).toBe(shown(DAY));
    await playheadAt(at(36011));
    expect(screen.getByRole('button', { name: /Morning, Dana\./ })).toHaveAttribute('aria-current', 'time');
  });

  it('shows what the large camera saw at the playhead as its caption', async () => {
    serve();
    open();
    const video = await screen.findByRole('region', { name: 'Video' });
    expect(await within(video).findByText('Two people at the counter')).toBeInTheDocument();
    expect(within(video).getByText('Seen: Dana')).toBeInTheDocument();
  });

  it('says when camera notes are only shown on the machine itself', async () => {
    serve({ notes: 403 });
    open();
    expect(
      await screen.findByText(/What the cameras saw is only shown on the machine Liberational runs on\./),
    ).toBeInTheDocument();
  });

  it('says so when nothing was recorded', async () => {
    vi.mocked(daySegments).mockResolvedValue([]);
    server.use(
      http.get('/transcript/day', () => HttpResponse.json({ day: DAY, entries: [] })),
      http.get('/audio/spans', () => HttpResponse.json({ day: DAY, spans: [] })),
      http.get('/review/presence', () => HttpResponse.json({ chips: [] })),
      http.get('/review/summary', () => HttpResponse.json({ status: 'ready', blocks: [] })),
    );
    open(`#day=${DAY}`);
    expect(await screen.findByText('Nothing was transcribed this day.')).toBeInTheDocument();
    // Said once, in the player, with no row of empty tiles. The player says it once every source has
    // answered, which can be after the transcript did: wait for it rather than look at once.
    expect(await screen.findByText(/^Nothing was recorded (today|on .+)\.$/)).toBeInTheDocument();
    expect(screen.queryByText('No video now')).not.toBeInTheDocument();
  });

  it('reports a source that failed to load and shows the rest', async () => {
    serve();
    server.use(http.get('/audio/spans', () => HttpResponse.json({ error: 'disk' }, { status: 500 })));
    open();
    expect(await screen.findByText(/Could not load the audio list\./)).toHaveAttribute('role', 'status');
    expect(screen.getByRole('button', { name: /Coffee is ready\./ })).toBeInTheDocument();
  });
});

describe('came-into-view chips', () => {
  // The presence chips: an event chip per camera, padded spans to play, and no time "in all".
  const PRESENCE = {
    id: 'presence:cam0',
    kind: 'event',
    label: 'cam0 came into view',
    spans: [
      [at(36200), at(36210)],
      [at(36450), at(36460)],
    ],
    total_s: 0,
    count: 2,
    notes: ['Sampled every 30 s: the start may be up to 30 s earlier.', 'Coverage unknown before this.'],
  };

  it('says how many times, with the coverage note, and no misleading duration', async () => {
    serve();
    server.use(http.get('/review/presence', () => HttpResponse.json({ chips: [...CHIPS, PRESENCE] })));
    open();
    const chip = await screen.findByRole('checkbox', { name: /cam0 came into view/ });
    await userEvent.hover(chip);
    expect(await screen.findByRole('tooltip')).toHaveTextContent(
      'cam0 came into view: 2 times. Sampled every 30 s: the start may be up to 30 s earlier.',
    );
  });

  it('can be picked and stepped through like any chip', async () => {
    serve();
    server.use(http.get('/review/presence', () => HttpResponse.json({ chips: [...CHIPS, PRESENCE] })));
    open();
    const filters = await screen.findByRole('group', { name: 'Show only' });
    await userEvent.click(await within(filters).findByRole('checkbox', { name: /cam0 came into view/ }));
    fireEvent.keyDown(document.body, { key: ']' });
    await playheadAt(at(36200));
  });
});

describe('moving through the day', () => {
  it('moves the playhead to a transcript line when it is clicked, and keeps it in the link', async () => {
    const clock = vi.spyOn(performance, 'now').mockReturnValue(10000);
    serve();
    open();
    const line = await screen.findByRole('button', { name: /Where did the dog go\?/ });
    await userEvent.click(line);
    // Advance past the hash throttle without racing waitFor's one-second deadline.
    clock.mockReturnValue(11001);
    await playheadAt(at(36400));
    expect(line).toHaveAttribute('aria-current', 'time');
    await waitFor(() => expect(location.hash).toBe(`#day=${DAY}&t=${at(36400)}`));
  });

  it('keeps only the time-block summary, with no header summary or day-summary request', async () => {
    serve();
    open();
    await screen.findByRole('button', { name: /Coffee is ready\./ });
    expect(screen.queryByRole('button', { name: 'Full summary' })).not.toBeInTheDocument();
    expect(screen.queryByText(/Recorded highlights:|Recorded so far:/)).not.toBeInTheDocument();
    expect(calls.summary ?? 0).toBe(0);
    expect(screen.getByRole('heading', { level: 1, name: 'Review' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Ask' })).toHaveAttribute('aria-haspopup', 'dialog');
    expect(screen.queryByRole('textbox', { name: 'Your question' })).not.toBeInTheDocument();
    expect(screen.getByRole('complementary', { name: 'Review details' })).toBeInTheDocument();
  });

  it('steps through the time a selected chip covers', async () => {
    serve();
    server.use(
      http.get('/review/presence', ({ request }) =>
        HttpResponse.json({ chips: new URL(request.url).searchParams.get('day') === DAY ? CHIPS : [] }),
      ),
    );
    open();
    const filters = await screen.findByRole('group', { name: 'Show only' });
    const dog = await within(filters).findByRole('checkbox', { name: /dog/ });
    expect(
      within(within(filters).getByRole('group', { name: 'People' })).getByRole('button', { name: /Dana/ }),
    ).toBeInTheDocument(); // people are chips in Show only
    expect(screen.queryByLabelText('Moment navigation shortcuts')).not.toBeInTheDocument();
    await userEvent.click(dog);
    expect(dog).toBeChecked();
    await userEvent.keyboard(']');
    await playheadAt(at(36100));
    await waitFor(() =>
      expect(screen.getByLabelText('Moment navigation shortcuts')).toHaveTextContent('1 of 2 next ]'),
    );
    expect(within(screen.getByLabelText('Moment navigation shortcuts')).queryByText('[')).toBeNull();
    await userEvent.keyboard(']');
    await playheadAt(at(36500));
    await waitFor(() =>
      expect(screen.getByLabelText('Moment navigation shortcuts')).toHaveTextContent('[ prev 2 of 2'),
    );
    expect(within(screen.getByLabelText('Moment navigation shortcuts')).queryByText(']')).toBeNull();
    fireEvent.keyDown(document.body, { key: '[' });
    await playheadAt(at(36100));
    await userEvent.click(within(filters).getByRole('button', { name: 'Show all' }));
    expect(dog).not.toBeChecked();
    expect(screen.queryByLabelText('Moment navigation shortcuts')).not.toBeInTheDocument();
  });

  it('opens another day when one is picked', async () => {
    serve();
    open();
    await screen.findByRole('button', { name: /Morning, Dana\./ });
    await chooseDay('2026-10-26');
    await waitFor(() => expect(location.hash).toContain('day=2026-10-26'));
  });

  it('reloads the day when Today is picked while today is showing, keeping the transcript meanwhile', async () => {
    const today = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    const iso = `${today.getFullYear()}-${pad(today.getMonth() + 1)}-${pad(today.getDate())}`;
    serve();
    open(`#day=${iso}`);
    await screen.findByRole('button', { name: /Coffee is ready\./ });
    let release = () => {};
    hold = new Promise(r => {
      release = r;
    });
    await userEvent.click(screen.getByRole('button', { name: 'Today' }));
    await waitFor(() => expect(calls[`transcript:${iso}`]).toBe(2));
    expect(screen.getByRole('button', { name: /Coffee is ready\./ })).toBeInTheDocument();
    release();
  });

  it('keeps the day it opened with when the network comes back, so the transcript and the engine agree', async () => {
    serve();
    open();
    await screen.findByRole('button', { name: /Coffee is ready\./ });
    onlineManager.setOnline(false);
    onlineManager.setOnline(true);
    await act(() => new Promise(resolve => setTimeout(resolve, 50)));
    expect(calls[`transcript:${DAY}`]).toBe(1);
  });

  it('reuses a loaded day when it is picked again after another', async () => {
    serve();
    open();
    await screen.findByRole('button', { name: /Coffee is ready\./ });
    await chooseDay('2026-10-26');
    await waitFor(() => expect(calls['transcript:2026-10-26']).toBe(1));
    await chooseDay(DAY);
    await waitFor(() => expect(shownDay()).toBe(shown(DAY)));
    expect(calls[`transcript:${DAY}`]).toBe(1);
  });
});

describe('keyboard shortcuts', () => {
  it('lists exactly the keys Review handles', async () => {
    serve();
    open();
    fireEvent.click(await screen.findByRole('button', { name: 'Shortcuts and timeline help' }));
    const help = await screen.findByRole('table', { name: 'Keyboard shortcuts' });
    expect(
      within(help)
        .getAllByText((_, el) => el?.tagName === 'KBD')
        .map(k => k.textContent),
    ).toEqual(REVIEW_KEYS.flatMap(k => k.shown));
    // Two columns: the keys in one cell, what they do in the next, under their section.
    const row = (name: RegExp) => within(help).getByRole('row', { name });
    expect(within(row(/^\[\s*or\s*\]/)).getAllByRole('cell')[1]).toHaveTextContent(
      'Previous or next moment Show only picks',
    );
    expect(within(row(/^s /)).getAllByRole('cell')[1]).toHaveTextContent('Skip quiet time on or off');
    expect(
      within(help)
        .getAllByRole('rowheader')
        .map(h => h.textContent),
    ).toEqual(['Play', 'Move around', 'Mark an event']);
  });

  it('handles every listed key, and leaves the rest alone', async () => {
    serve();
    open();
    await screen.findByRole('button', { name: /Where did the dog go\?/ });
    const press = (key: string) => fireEvent.keyDown(document.body, { key }); // false: Review took it
    const plain = REVIEW_KEYS.filter(k => !['play', 'mark', 'clear'].includes(k.id)).flatMap(k => k.keys);
    expect(plain.filter(press)).toEqual([]);
    expect(press('i')).toBe(false); // a mark, so Escape has one to clear
    expect(press('Escape')).toBe(false);
    expect(press('Escape')).toBe(true); // nothing left to clear
    expect(press('x')).toBe(true);
    expect(press('I')).toBe(true); // Shift and i isn't a mark
  });
});

it("takes Review's keys from a focused transcript line too", async () => {
  serve();
  open();
  const line = await screen.findByRole('button', { name: /Where did the dog go\?/ });
  line.focus();
  expect(fireEvent.keyDown(line, { key: 'j' })).toBe(false); // false: Review took it
});

describe('playing', () => {
  it('plays and pauses from the button', async () => {
    serve();
    open();
    await playheadAt(at(36011));
    // Wait for the whole day to load: the playhead comes from the link first, and the day's data landing after a
    // click would seek again without playing.
    await dayLoaded();
    await userEvent.click(screen.getByRole('button', { name: 'Play' }));
    expect(await screen.findByRole('button', { name: 'Pause' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Pause' }));
    expect(await screen.findByRole('button', { name: 'Play' })).toBeInTheDocument();
  });

  it('plays from a transcript line when it is clicked', async () => {
    serve();
    open();
    await userEvent.click(await screen.findByRole('button', { name: /Coffee is ready\./ }));
    expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();
  });
});

describe('sound', () => {
  it('says on the status line when the browser blocks the sound', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) {
      // video is muted, so it may play; sound needs the browser's permission
      return this instanceof HTMLAudioElement
        ? Promise.reject(new DOMException('play() failed because the user did not interact first.', 'NotAllowedError'))
        : Promise.resolve();
    });
    serve();
    // The first Play, before the room audio has come: the click unlocks the empty <audio> with silence, and that's
    // what the browser refuses. Paused 300 ms, the engine preloads the room audio (PRELOAD_MS), and a click after that
    // waits on the picture, which jsdom never plays: held here, the click always comes first, however slow the run.
    let release = () => {};
    const held = new Promise<void>(resolve => {
      release = resolve;
    });
    onTestFinished(() => release());
    server.use(
      http.get('/audio/range', async () => {
        await held;
        return new HttpResponse(new Blob(['room']), { headers: { 'Content-Type': 'audio/wav' } });
      }),
    );
    open();
    await playheadAt(at(36011));
    await userEvent.click(screen.getByRole('button', { name: 'Play' }));
    expect(await screen.findByText('The browser blocked the sound; click Play again.')).toHaveAttribute(
      'role',
      'status',
    );
  });
});

describe('a person', () => {
  it('has a removable filter, and once removed it stays off on other days', async () => {
    serve();
    open(`#person=Dana&day=${DAY}`);
    await userEvent.click(await screen.findByRole('button', { name: 'Stop showing only Dana' }));
    expect(location.hash).not.toContain('person=');
    expect(screen.queryByRole('button', { name: 'Stop showing only Dana' })).not.toBeInTheDocument();
    await chooseDay(OTHER_DAY);
    const filters = await screen.findByRole('group', { name: 'Show only' });
    const people = await within(filters).findByRole('group', { name: 'People' });
    expect(await within(people).findByRole('button', { name: /Dana/ })).toHaveAttribute('aria-pressed', 'false');
  });

  it("opens on the person's day: their visits, and a transcript of only their lines, the others a switch away", async () => {
    serve();
    open(`#person=Dana&day=${DAY}`);
    const tab = await screen.findByRole('tab', { name: 'Dana' });
    expect(tab).toHaveAttribute('aria-selected', 'true');
    const visits = await screen.findByRole('list', { name: "Dana's visits" });
    expect(within(visits).getAllByRole('listitem')).toHaveLength(1); // the one line Dana said
    expect(within(visits).getByText('Said 1 line')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: 'Transcript' }));
    const lines = screen.getByRole('region', { name: 'Transcript of the day' });
    expect(await within(lines).findByText('Where did the dog go?')).toBeInTheDocument();
    expect(within(lines).queryByText('Coffee is ready.')).not.toBeInTheDocument();
    await userEvent.click(within(lines).getByRole('switch', { name: "Show the others' lines too, dimmed" }));
    expect(within(lines).getByText('Coffee is ready.').closest('article')).toHaveAttribute('data-other');
    expect(within(lines).getByText('Where did the dog go?').closest('article')).not.toHaveAttribute('data-other');
  });

  it('says when the day has nothing of the person a link named', async () => {
    serve();
    open(`#person=Eve&day=${DAY}`);
    expect(await screen.findByText(`Nothing of Eve on ${DAY}.`, { exact: false })).toHaveAttribute('role', 'status');
  });

  it("shows a person's time when a link names them, as the contact card's See all in Review does", async () => {
    serve();
    open(`#person=Dana&day=${DAY}`);
    const filters = await screen.findByRole('group', { name: 'Show only' });
    const people = await within(filters).findByRole('group', { name: 'People' });
    const dana = await within(people).findByRole('button', { name: /Dana/ });
    await waitFor(() => expect(dana).toHaveAttribute('aria-pressed', 'true'));
    expect(screen.queryByRole('heading', { name: /clips/ })).not.toBeInTheDocument(); // no clip browser any more
  });
});

describe('ask', () => {
  const askBox = async () => {
    if (!screen.queryByRole('dialog', { name: 'Ask' }))
      await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
    return screen.findByRole('textbox', { name: 'Your question' });
  };
  const ANSWER = {
    status: 'completed',
    question: 'What did the dog do?',
    created_at: at(40000),
    day: DAY,
    result: {
      status: 'ok',
      scope: 'history',
      answer: 'It went out of the front door [C1].',
      evidence: [
        { id: 'C1', ts: at(36400), camera: 'cam0', source: 'camera', text: 'A dog walks out.' },
        { id: 'C2', ts: at(36500), camera: 'cam0', source: 'camera', text: 'Not cited.' },
      ],
    },
  };

  it('asks about the day that is showing, with the day as a hint in the one Ask', async () => {
    serve();
    const jobs = jobServer();
    open();
    const box = await askBox();
    await userEvent.type(box, 'What did the dog do?{Enter}');
    await waitFor(() => expect(jobs.posts).toEqual([{ question: 'What did the dog do?', day: DAY }]));
    expect(screen.getByRole('switch', { name: /^Deep research/ })).toBeInTheDocument();
  });

  it('answers with the moments it cites as time chips that play from there, and asks if it was right', async () => {
    vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Alice', cams: ['cam0'], views: { cam0: 'front camera' } });
    serve();
    const jobs = jobServer();
    jobs.replies.push(ANSWER);
    open();
    await userEvent.type(await askBox(), 'What did the dog do?{Enter}');
    const cited = await screen.findByRole('list', { name: 'Moments it cites' });
    await waitFor(() => expect(citedMarks.current).toEqual([at(36400)]));
    expect(
      within(cited)
        .getAllByRole('button')
        .map(b => b.textContent),
    ).toEqual([`${hm(at(36400))} front camera`]);
    await userEvent.click(within(cited).getByRole('button'));
    await playheadAt(at(36400));
    const mark = screen.getByRole('region', { name: 'Mark this answer' });
    expect(mark).toHaveTextContent('Was this right?');
    expect(within(mark).getByRole('button', { name: 'Yes' })).toBeEnabled();
    expect(within(mark).getByRole('button', { name: 'No, correct it' })).toBeEnabled();
  });

  it('keeps following a question (no Activity pause in the real app)', async () => {
    serve();
    const jobs = jobServer(); // stays running
    history.replaceState(null, '', `/review#day=${DAY}&t=${at(36011)}`);
    render(
      <Providers client={newQueryClient()}>
        <AskBar tab="review" setup={false} />
        <Review />
      </Providers>,
    ); // not Mantine's test mode, which never hides a tab with <Activity>
    await userEvent.type(await askBox(), 'What did the dog do?{Enter}');
    await waitFor(() => expect(jobs.posts).toHaveLength(1));
    const polled = jobs.polls.length;
    await waitFor(() => expect(jobs.polls.length).toBeGreaterThan(polled));
  });

  it('suggests questions to start with, and asks one when picked', async () => {
    serve();
    const jobs = jobServer();
    open();
    await askBox();
    const suggested = await screen.findByRole('list', { name: 'Suggested questions' });
    const first = within(suggested).getAllByRole('button')[0] as HTMLElement;
    await userEvent.click(first);
    await waitFor(() => expect(jobs.posts).toEqual([expect.objectContaining({ question: first.textContent })]));
    expect(screen.queryByRole('list', { name: 'Suggested questions' })).not.toBeInTheDocument();
  });

  it('keeps the draft, playhead, and selected detail tab when Ask closes and reopens', async () => {
    serve();
    const jobs = jobServer();
    open();
    await playheadAt(at(36011));
    await userEvent.click(screen.getByRole('tab', { name: 'Summary' }));
    const trigger = screen.getByRole('button', { name: 'Ask' });
    await userEvent.click(trigger);
    await userEvent.type(await screen.findByRole('textbox', { name: 'Your question' }), 'A draft about this day');
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Ask' })).not.toBeInTheDocument());
    await userEvent.click(trigger);
    expect(await screen.findByRole('textbox', { name: 'Your question' })).toHaveValue('A draft about this day');
    await userEvent.keyboard('{Escape}');
    expect(screen.getByRole('tab', { name: 'Summary' })).toHaveAttribute('aria-selected', 'true');
    await playheadAt(at(36011));
    expect(jobs.posts).toEqual([]);
  });

  it('keeps polling an in-flight answer while the modal is closed', async () => {
    serve();
    const jobs = jobServer();
    history.replaceState(null, '', `/review#day=${DAY}&t=${at(36011)}`);
    render(
      <Providers client={newQueryClient()}>
        <AskBar tab="review" setup={false} />
        <Review />
      </Providers>,
    );
    await userEvent.type(await askBox(), 'What did the dog do?{Enter}');
    await waitFor(() => expect(jobs.posts).toHaveLength(1));
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Ask' })).not.toBeInTheDocument());
    const polled = jobs.polls.length;
    await waitFor(() => expect(jobs.polls.length).toBeGreaterThan(polled));
    await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
    expect(await screen.findByText('What did the dog do?')).toBeInTheDocument();
    expect(jobs.posts).toHaveLength(1);
  });
});

describe('summary by time', () => {
  it('sits beside the video and reads what the timeline shows', async () => {
    serve();
    let range: { start: number; end: number } | null = null;
    server.use(
      http.get('/review/summary', ({ request }) => {
        const q = new URL(request.url).searchParams;
        if (!q.has('day')) range = { start: Number(q.get('start')), end: Number(q.get('end')) };
        const block = { start: at(36000), end: at(36600), text: 'Coffee and the dog', tags: ['object:dog'] };
        return HttpResponse.json({ ...range, level: 600, status: 'ready', blocks: [block] });
      }),
    );
    open();
    await userEvent.click(screen.getByRole('tab', { name: 'Summary' }));
    const column = await screen.findByRole('region', { name: 'Summary by time' });
    const summaries = await within(column).findByRole('list');
    expect(within(summaries).getByText('Coffee and the dog')).toBeInTheDocument();
    const shown = range as { start: number; end: number } | null;
    expect(shown && shown.start <= at(36011) && at(36011) <= shown.end).toBe(true); // the range around the playhead
  });

  it('jumps to a summary block across midnight without replacing the player or timeline', async () => {
    serve();
    const target = D0 - 30;
    server.use(
      http.get('/review/summary', ({ request }) => {
        const q = new URL(request.url).searchParams;
        return HttpResponse.json({
          start: Number(q.get('start')),
          end: Number(q.get('end')),
          level: 600,
          status: 'ready',
          blocks: q.has('day')
            ? []
            : [{ start: target, end: D0 + 30, text: 'A conversation across midnight', tags: [] }],
        });
      }),
    );
    open();
    await playheadAt(at(36011));
    const oldPlayers = [...players];
    const oldTimeline = timeline();
    await userEvent.click(screen.getByRole('tab', { name: 'Summary' }));
    const column = await screen.findByRole('region', { name: 'Summary by time' });
    const summaries = await within(column).findByRole('list');
    const seekPlay = vi.spyOn(ReviewEngine.prototype, 'seekPlay');
    onTestFinished(() => seekPlay.mockRestore());
    await userEvent.click(within(summaries).getByRole('button', { name: /A conversation across midnight/ }));
    // The click plays from the block's start. With no picture or room audio there, playback goes on with the
    // next spoken line, so the playhead itself moves on; the list, player, and timeline stay put.
    expect(seekPlay).toHaveBeenCalledWith(target);
    expect(timeline()).toBe(oldTimeline);
    expect(players).toEqual(oldPlayers);
    expect(screen.queryByTestId('timeline-skeleton')).not.toBeInTheDocument();
    expect(screen.queryByTestId('player-skeleton')).not.toBeInTheDocument();
  });

  it("clears the previous day's person summary while the new day's summary is loading", async () => {
    serve();
    let release = () => {};
    const waiting = new Promise<void>(done => {
      release = done;
    });
    server.use(
      http.get('/review/summary', async ({ request }) => {
        const day = new URL(request.url).searchParams.get('day');
        if (day === OTHER_DAY) await waiting;
        return HttpResponse.json({
          status: 'ready',
          blocks: day
            ? [
                {
                  start: at(36000),
                  end: at(36600),
                  text: day === DAY ? 'Dana planned the garden' : 'Dana read a book',
                  tags: ['person:dana'],
                },
              ]
            : [],
        });
      }),
    );
    open(`#person=Dana&day=${DAY}&t=${at(36011)}`);
    expect(await screen.findByRole('list', { name: 'Summary mentions of Dana' })).toHaveTextContent(
      'Dana planned the garden',
    );
    await chooseDay(OTHER_DAY);
    await waitFor(() => expect(shownDay()).toBe(shown(OTHER_DAY)));
    expect(screen.queryByRole('list', { name: 'Summary mentions of Dana' })).not.toBeInTheDocument();
    release();
    expect(await screen.findByRole('list', { name: 'Summary mentions of Dana' })).toHaveTextContent('Dana read a book');
  });
});

describe('detector storage availability', () => {
  it.each([
    [true, 'Detector history is loading from local storage.'],
    [false, 'Detector history is unavailable.'],
  ])('shows an explicit storage notice when pending is %s', async (pending, notice) => {
    serve();
    server.use(
      http.get('/review/presence', () =>
        HttpResponse.json({ chips: [], partial: true, pending, error: pending ? 'StoragePending' : 'StorageError' }),
      ),
    );
    open();
    expect(await screen.findByText(new RegExp(notice))).toHaveAttribute('role', 'status');
    expect(!!screen.queryByTestId('chips-skeleton')).toBe(pending);
    expect(!!screen.queryByText("Could not load the day's objects and events.")).toBe(!pending);
    expect(await screen.findByRole('button', { name: /Coffee is ready\./ })).toBeInTheDocument();
  });

  it('hides old chips during pending refresh and restores them after recovery', async () => {
    serve();
    server.use(http.get('/review/days', () => HttpResponse.json({ days: RECORDED, today: '2026-10-28' })));
    const { client } = open();
    const filters = await screen.findByRole('group', { name: 'Show only' });
    expect(await within(filters).findByRole('checkbox', { name: /dog/ })).toBeInTheDocument();
    server.use(
      http.get('/review/presence', () =>
        HttpResponse.json({ chips: [], partial: true, pending: true, error: 'StoragePending' }),
      ),
    );
    await client.refetchQueries({ queryKey: ['review', 'day', DAY, 'chips'] });
    await screen.findByText(/Detector history is loading from local storage\./);
    expect(screen.queryByRole('group', { name: 'Show only' })).not.toBeInTheDocument();
    expect(screen.getByTestId('chips-skeleton')).toBeInTheDocument();
    // The video's boxes and Skip quiet time stay usable meanwhile, under the player rather than on its bar, so the
    // bar keeps its height.
    expect(screen.getByRole('checkbox', { name: 'Skip quiet time' })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Review markers' })).toBeInTheDocument();
    const video = screen.getByRole('region', { name: 'Video' });
    expect(within(video).queryByRole('checkbox', { name: 'Skip quiet time' })).not.toBeInTheDocument();
    server.use(http.get('/review/presence', () => HttpResponse.json({ chips: CHIPS })));
    const recoveredFilters = await screen.findByRole('group', { name: 'Show only' });
    expect(await within(recoveredFilters).findByRole('checkbox', { name: /dog/ })).toBeInTheDocument();
    expect(screen.queryByTestId('chips-skeleton')).not.toBeInTheDocument();
    expect(screen.queryByText(/Detector history is loading from local storage\./)).not.toBeInTheDocument();
  });

  it.each(['storage', 'http'] as const)('automatically recovers an ended day after a hard %s error', async kind => {
    serve();
    let fail = true;
    server.use(
      http.get('/review/days', () => HttpResponse.json({ days: RECORDED, today: '2026-10-28' })),
      http.get('/review/presence', () =>
        fail
          ? HttpResponse.json({ chips: [], error: 'StorageError' }, { status: kind === 'http' ? 500 : 200 })
          : HttpResponse.json({ chips: CHIPS }),
      ),
    );
    open();
    expect(await screen.findByText("Could not load the day's objects and events.")).toBeInTheDocument();
    expect(screen.queryByTestId('chips-skeleton')).not.toBeInTheDocument();
    fail = false;
    const filters = await screen.findByRole('group', { name: 'Show only' });
    expect(await within(filters).findByRole('checkbox', { name: /dog/ })).toBeInTheDocument();
    expect(screen.queryByText("Could not load the day's objects and events.")).not.toBeInTheDocument();
  });
});

describe('loading and errors', () => {
  it('covers the player, the timeline, and the transcript with placeholders until the day loads', async () => {
    let answer: () => void = () => {};
    hold = new Promise(r => {
      answer = r;
    });
    serve();
    open();
    expect(screen.getByTestId('player-skeleton')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-skeleton')).toBeInTheDocument();
    expect(screen.getByLabelText('Loading the transcript')).toHaveAttribute('aria-busy', 'true');
    expect(screen.getByText('Loading the day…')).toBeInTheDocument();
    answer();
    await dayLoaded();
    expect(screen.queryByTestId('player-skeleton')).not.toBeInTheDocument();
    expect(screen.queryByTestId('timeline-skeleton')).not.toBeInTheDocument();
    expect(screen.queryByText('Loading the day…')).not.toBeInTheDocument();
  });

  it('loads the rest of the day while the chips are still being found', async () => {
    serve();
    server.use(http.get('/review/presence', () => new Promise<never>(() => {}))); // never answers
    open();
    await dayLoaded();
    expect(screen.getByTestId('chips-skeleton')).toBeInTheDocument();
    expect(screen.queryByTestId('timeline-skeleton')).not.toBeInTheDocument();
    await playheadAt(at(36011)); // the timeline works meanwhile
  });

  it('shows a retry when the chips fail, and the chips after it', async () => {
    serve();
    let fail = true;
    server.use(
      http.get('/review/presence', () =>
        fail ? HttpResponse.json({ error: 'broken' }, { status: 500 }) : HttpResponse.json({ chips: CHIPS }),
      ),
    );
    open();
    const message = await screen.findByText("Could not load the day's objects and events.");
    const alert = message.closest<HTMLElement>('[role="alert"]') as HTMLElement;
    fail = false;
    await userEvent.click(within(alert).getByRole('button', { name: 'Retry' }));
    await waitFor(() =>
      expect(screen.queryByText("Could not load the day's objects and events.")).not.toBeInTheDocument(),
    );
    expect(screen.queryByTestId('chips-skeleton')).not.toBeInTheDocument();
    const filters = screen.getByRole('group', { name: 'Show only' });
    expect(await within(filters).findByRole('checkbox', { name: /dog/ })).toBeInTheDocument();
  });

  it('keeps loading, not failing, while a proxy gives up on a long day', async () => {
    serve();
    stillComputing.retryMs = 20;
    let answered = 0;
    server.use(
      http.get('/review/presence', ({ request }) =>
        new URL(request.url).searchParams.get('day') !== DAY
          ? HttpResponse.json({ chips: [] })
          : ++answered < 3
            ? new HttpResponse(null, { status: 504 })
            : HttpResponse.json({ chips: CHIPS }),
      ),
    );
    try {
      open();
      expect(await screen.findByTestId('chips-skeleton')).toBeInTheDocument();
      const filters = await screen.findByRole('group', { name: 'Show only' });
      expect(await within(filters).findByRole('checkbox', { name: /dog/ })).toBeInTheDocument();
      expect(screen.queryByText("Could not load the day's objects and events.")).not.toBeInTheDocument();
      expect(answered).toBe(3);
    } finally {
      stillComputing.retryMs = 5000;
    }
  });

  it('says how long an ended day has left while the server works through it, then shows its chips', async () => {
    serve();
    working.pollMs = 20;
    let asked = 0;
    let complete = false;
    server.use(
      http.get('/review/presence', ({ request }) =>
        new URL(request.url).searchParams.get('day') !== DAY
          ? HttpResponse.json({ chips: [] })
          : ++asked && !complete
            ? HttpResponse.json({ chips: [], working: { elapsed_s: 10, estimate_s: 75, left_s: 65 } })
            : HttpResponse.json({ chips: CHIPS }),
      ),
    );
    try {
      open();
      await waitFor(() => expect(screen.getByText(/^Working through .+, about a minute left\.$/)).toBeInTheDocument());
      expect(screen.getByTestId('chips-skeleton')).toBeInTheDocument();
      await waitFor(() => expect(asked).toBeGreaterThan(1));
      complete = true;
      const filters = await screen.findByRole('group', { name: 'Show only' });
      expect(await within(filters).findByRole('checkbox', { name: /dog/ })).toBeInTheDocument();
      expect(screen.queryByText(/Working through/)).not.toBeInTheDocument();
      expect(asked).toBeGreaterThanOrEqual(3);
    } finally {
      working.pollMs = 3000;
    }
  });

  it("keeps a day's chips while the server works it out again, and says so once", async () => {
    serve();
    working.pollMs = 50;
    let busy = false;
    server.use(
      http.get('/review/presence', () =>
        busy
          ? HttpResponse.json({ chips: [], working: { elapsed_s: 1, estimate_s: 75, left_s: 74 } })
          : HttpResponse.json({ chips: CHIPS }),
      ),
    );
    try {
      const { client } = open();
      const filters = await screen.findByRole('group', { name: 'Show only' });
      expect(await within(filters).findByRole('checkbox', { name: /dog/ })).toBeInTheDocument();
      busy = true; // a restart after a merge (new code), or rows added to the day
      await client.invalidateQueries({ queryKey: reviewKey.chips(DAY) });
      expect(await screen.findByText(/Working through .+, about a minute left\./)).toBeInTheDocument();
      expect(
        within(screen.getByRole('group', { name: 'Show only' })).getByRole('checkbox', { name: /dog/ }),
      ).toBeInTheDocument();
      expect(screen.getAllByText(/Working through/)).toHaveLength(1);
    } finally {
      working.pollMs = 3000;
    }
  });

  it('retries the part of the day that failed, and the timeline and the player get it', async () => {
    serve();
    let fail = true;
    vi.mocked(daySegments).mockImplementation(async () => {
      if (fail) throw new Error('the gateway is down');
      return SEGMENTS;
    });
    open();
    expect(await screen.findByText(/Could not load the video list\./)).toBeInTheDocument();
    expect(screen.getByLabelText('cam0 video').parentElement).not.toHaveAttribute('data-video');
    fail = false;
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await dayLoaded();
    await waitFor(() => expect(screen.getByLabelText('cam0 video').parentElement).toHaveAttribute('data-video'));
  });

  it("loads an uncached day's details without covering the continuous timeline", async () => {
    serve();
    open();
    expect(await screen.findByText('Morning, Dana.')).toBeInTheDocument();
    hold = new Promise(() => {}); // the new day's transcript never answers
    await chooseDay('2026-10-20');
    expect(await screen.findByLabelText('Loading the transcript')).toHaveAttribute('aria-busy', 'true');
    expect(screen.queryByTestId('timeline-skeleton')).not.toBeInTheDocument();
    expect(screen.queryByText('Morning, Dana.')).not.toBeInTheDocument();
  });

  it('shows the error with a retry when nothing of the day loads, and the day after it', async () => {
    serve();
    let fail = true;
    const refuse = (ok: () => Response) => () => (fail ? HttpResponse.json({ error: 'down' }, { status: 500 }) : ok());
    vi.mocked(daySegments).mockImplementation(async () => {
      if (fail) throw new Error('the gateway is down');
      return SEGMENTS;
    });
    server.use(
      http.get(
        '/transcript/day',
        refuse(() => HttpResponse.json({ day: DAY, entries: ENTRIES })),
      ),
      http.get(
        '/audio/spans',
        refuse(() => HttpResponse.json({ day: DAY, spans: [] })),
      ),
    );
    open();
    const message = await screen.findByText(
      new RegExp(`^Could not load ${DAY}: the transcript, the video list, and the audio list all failed\\.`),
    );
    const alert = message.closest<HTMLElement>('[role="alert"]') as HTMLElement;
    expect(screen.queryByTestId('player-skeleton')).not.toBeInTheDocument(); // never an endless placeholder
    fail = false;
    await userEvent.click(within(alert).getByRole('button', { name: 'Retry' }));
    await dayLoaded();
    // The engine took the day's data: the camera with video at the playhead gets its tile.
    await waitFor(() => expect(screen.getByLabelText('cam0 video').parentElement).not.toHaveAttribute('hidden'));
  });
});

let show: (on: boolean) => void = () => {};
/** Review in a tab that can be hidden, as App shows it. */
function Tabs() {
  const [on, setOn] = useState(true);
  show = setOn;
  return (
    <Activity mode={on ? 'visible' : 'hidden'}>
      <Review />
    </Activity>
  );
}

describe('hidden behind another tab', () => {
  /** Review as the dashboard shows it: its query defaults (fresh for 30 s), inside a tab that can be hidden. */
  function openTab(hash = `#day=${DAY}&t=${at(36011)}`) {
    history.replaceState(null, '', `/review${hash}`);
    const client = newQueryClient();
    client.setDefaultOptions({ queries: { ...client.getDefaultOptions().queries, retry: false } });
    return render(
      <Providers client={client} test>
        <Tabs />
      </Providers>,
    );
  }

  it('stops its sound and lets go of its videos', async () => {
    serve();
    const { container } = openTab();
    await screen.findByRole('button', { name: /Coffee is ready\./ });
    const audio = container.querySelector('audio') as HTMLAudioElement;
    Object.defineProperty(audio, 'paused', { value: false, configurable: true }); // the room is playing
    const videos = players.slice(-2); // this page's camera: its two players
    if (videos[0]) videos[0].src = 'gateway:1:1';
    const pause = vi.mocked(HTMLMediaElement.prototype.pause);
    pause.mockClear();
    await act(async () => show(false));
    expect(pause.mock.contexts).toContain(audio);
    expect(videos.map(v => v.getAttribute('src'))).toEqual([null, null]); // no connection kept (engine.release)
  });

  it('opens on what it had after more than 30 s away, without loading the day again', async () => {
    serve();
    openTab();
    await screen.findByRole('button', { name: /Coffee is ready\./ });
    await act(async () => show(false));
    const now = Date.now();
    vi.spyOn(Date, 'now').mockReturnValue(now + 31_000);
    await act(async () => show(true));
    await act(() => new Promise(r => setTimeout(r, 50)));
    expect(calls[`transcript:${DAY}`]).toBe(1);
    expect(screen.queryByTestId('player-skeleton')).toBeNull();
  });

  it('opens a day afresh when a link names it with no moment', async () => {
    const today = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    const iso = `${today.getFullYear()}-${pad(today.getMonth() + 1)}-${pad(today.getDate())}`;
    serve();
    openTab(`#day=${iso}`);
    await screen.findByRole('button', { name: /Coffee is ready\./ });
    await waitFor(() => expect(location.hash).toContain('&t=')); // the playhead is in the link, as it always is
    await act(async () => show(false));
    history.replaceState(null, '', `/review#day=${iso}`);
    await act(async () => show(true));
    await waitFor(() => expect(calls[`transcript:${iso}`]).toBe(2));
  });
});

describe('naming and correcting a speaker from the transcript', () => {
  const KEYED = [
    { ...ENTRIES[0], line: 'mic.36010000' },
    { ...ENTRIES[1], line: 'mic.36015000' },
    { ...ENTRIES[2], line: 'mic.36400000' },
    { ts_start: at(36500), ts_end: at(36502), text: 'Anyone home?', speaker: 'unknown', speaker_id: null },
  ];
  /** The day's lines, as a function so a test can change what a second read says. */
  function lines(read: () => object[]) {
    server.use(http.get('/transcript/day', () => HttpResponse.json({ day: DAY, entries: read() })));
  }
  /** Why this name's reply for whatever line is asked about. */
  const explains = (name: string) =>
    server.use(
      http.get('/identity/why', () =>
        HttpResponse.json({
          name,
          subject: 'line',
          via: 'heard',
          basis: 'heard',
          asserted_at: null,
          source: null,
          confidence: 1,
          similarity: null,
          threshold: 0.55,
          voice: null,
          mark: { kind: 'line', ts_start: at(36400), ts_end: at(36404) },
        }),
      ),
    );
  const turn = (name: string) =>
    screen.getByRole('region', { name: 'Transcript of the day' }).querySelector(`article:has([aria-label="${name}"])`);

  it('corrects exactly the visible turn, keeps partial failure visible, and retries only failed lines', async () => {
    serve();
    lines(() => KEYED);
    const corrected: unknown[] = [];
    const undone: unknown[] = [];
    let failed = true;
    let undoFailed = true;
    server.use(
      http.post('/people/correct', async ({ request }) => {
        const body = (await request.json()) as { id: string };
        corrected.push(body);
        if (body.id === 'mic.36015000' && failed)
          return HttpResponse.json({ ok: false, error: 'Reference unavailable' });
        return HttpResponse.json({ ok: true, undo: `undo:${body.id}`, to: 'Kara' });
      }),
      http.post('/people/undo', async ({ request }) => {
        const body = (await request.json()) as { undo: string };
        undone.push(body);
        if (body.undo === 'undo:mic.36010000' && undoFailed)
          return HttpResponse.json({ ok: false, errors: ['Storage unavailable'] });
        return HttpResponse.json({ ok: true });
      }),
    );
    open();
    const t = await screen.findByRole('region', { name: 'Transcript of the day' });
    await within(t).findByText('Morning, Dana.');
    await userEvent.click(within(t).getByRole('button', { name: 'Not Alice' }));
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    expect(within(who).getByText("It's not Alice in this turn. Who is it?")).toBeInTheDocument();
    await userEvent.click(within(who).getByRole('button', { name: 'Kara' }));
    await screen.findByText(/Changed 1 of 2 lines/);
    expect(corrected).toEqual([
      { name: 'Alice', kind: 'voice', id: 'mic.36010000', action: 'move', to: 'Kara' },
      { name: 'Alice', kind: 'voice', id: 'mic.36015000', action: 'move', to: 'Kara' },
    ]);
    expect(screen.getByText(/Reference unavailable/)).toBeInTheDocument();
    failed = false;
    await userEvent.click(screen.getByRole('button', { name: 'Retry failed lines' }));
    await screen.findByText('Moved 2 lines to Kara');
    expect(corrected).toHaveLength(3);
    expect(corrected[2]).toMatchObject({ id: 'mic.36015000' });
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    await screen.findByText('Undo failed for 1 of 2 lines. Storage unavailable');
    undoFailed = false;
    // The toast it replaced leaves a tick later (lib/Toast)
    await waitFor(() => expect(screen.getAllByRole('button', { name: 'Undo' })).toHaveLength(1));
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    await screen.findByText('Undone');
    expect(undone).toEqual([
      { undo: 'undo:mic.36015000' },
      { undo: 'undo:mic.36010000' },
      { undo: 'undo:mic.36010000' },
    ]);
  });

  it('corrects only the line whose person-minus was picked and offers to retry a denied one', async () => {
    serve();
    lines(() => KEYED);
    const sent: unknown[] = [];
    server.use(
      http.post('/people/correct', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ok: false, error: 'Controls unavailable' }, { status: 403 });
      }),
    );
    open();
    const t = await screen.findByRole('region', { name: 'Transcript of the day' });
    const line = await within(t).findByRole('button', { name: `Not Alice in the line at ${hms(at(36015))}` });
    await userEvent.click(line);
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    expect(within(who).getByText("It's not Alice in this line. Who is it?")).toBeInTheDocument();
    await userEvent.click(within(who).getByRole('button', { name: 'Not sure, just remove it from Alice' }));
    await screen.findByText(/Changed 0 of 1 lines\. .*Controls unavailable/);
    expect(screen.getByRole('button', { name: 'Retry failed lines' })).toBeInTheDocument();
    expect(sent).toEqual([{ name: 'Alice', kind: 'voice', id: 'mic.36015000', action: 'remove' }]);
    expect(within(t).queryByRole('button', { name: /Confirm / })).toBeNull();
  });

  it('removes a numbered voice from every exact line in its turn and reports partial failure', async () => {
    serve();
    lines(() => KEYED);
    server.use(
      http.get('/speakers', () => HttpResponse.json({ speakers: [{ id: 1, label: 'Speaker 1' }], merged: {} })),
    );
    const sent: unknown[] = [];
    server.use(
      http.post('/people/correct', async ({ request }) => {
        const body = await request.json();
        sent.push(body);
        return sent.length === 1
          ? HttpResponse.json({ ok: true, undo: 'u1' })
          : HttpResponse.json({ ok: false, error: 'Controls unavailable' }, { status: 403 });
      }),
    );
    open();
    const t = await screen.findByRole('region', { name: 'Transcript of the day' });
    await within(t).findByText('Coffee is ready.');
    await userEvent.click(within(t).getByRole('button', { name: 'Not this voice' }));
    // An unmatched voice's turn goes through Who is this? by the voice's label, as a person's does.
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    await userEvent.click(
      within(who).getByRole('button', { name: 'Not sure, just remove it from the unmatched voice' }),
    );
    await screen.findByText(/Changed 1 of 2 lines\. .*Controls unavailable/);
    expect(sent).toEqual([
      { name: 'Speaker 1', kind: 'voice', id: 'mic.36010000', action: 'remove' },
      { name: 'Speaker 1', kind: 'voice', id: 'mic.36015000', action: 'remove' },
    ]);
    // The line that went through can be undone, and the one that didn't, retried.
    expect(screen.getByRole('button', { name: 'Undo' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry failed lines' })).toBeInTheDocument();
  });

  it('trashes exactly one visible turn and uses the transcript Undo endpoint', async () => {
    serve();
    let removed = false;
    lines(() => (removed ? KEYED.slice(2) : KEYED));
    const sent: unknown[] = [];
    server.use(
      http.post('/transcript/reject', async ({ request }) => {
        sent.push(await request.json());
        removed = true;
        return HttpResponse.json({ ok: true, undo: 'rejected-turn' });
      }),
      http.post('/transcript/undo', async ({ request }) => {
        sent.push(await request.json());
        removed = false;
        return HttpResponse.json({ ok: true });
      }),
    );
    open();
    const t = await screen.findByRole('region', { name: 'Transcript of the day' });
    const first = (await within(t).findByText('Morning, Dana.')).closest('article');
    if (!first) throw new Error('Missing turn');
    await userEvent.click(within(first).getByRole('button', { name: 'Not transcribed correctly' }));
    await screen.findByText('Marked as not transcribed correctly');
    expect(sent).toEqual([{ lines: ['mic.36010000', 'mic.36015000'] }]);
    await waitFor(() => expect(within(t).queryByText('Morning, Dana.')).toBeNull());
    expect(within(t).getByText('Where did the dog go?')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    await within(t).findByText('Morning, Dana.');
    expect(sent[1]).toEqual({ undo: 'rejected-turn' });
  });

  it('corrects a line put down to the wrong person, shows the new name, and undoes it', async () => {
    serve();
    let moved = false;
    lines(() =>
      moved ? KEYED.map(e => (e.ts_start === at(36400) ? { ...e, speaker: 'Kara', speaker_id: 3 } : e)) : KEYED,
    );
    const sent: unknown[] = [];
    server.use(
      http.post('/people/correct', async ({ request }) => {
        sent.push(await request.json());
        moved = true;
        return HttpResponse.json({ ok: true, undo: 'u1', to: 'Kara' });
      }),
      http.post('/people/undo', async ({ request }) => {
        sent.push(await request.json());
        moved = false;
        return HttpResponse.json({ ok: true });
      }),
    );
    open();
    const t = await screen.findByRole('region', { name: 'Transcript of the day' });
    await within(t).findByText('Where did the dog go?');
    await userEvent.click(within(t).getByRole('button', { name: 'Not Dana' }));
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    expect(within(who).getByText("It's not Dana in this line. Who is it?")).toBeInTheDocument();
    await userEvent.click(within(who).getByRole('button', { name: 'Kara' }));

    await waitFor(() =>
      expect(sent[0]).toEqual({ name: 'Dana', kind: 'voice', id: 'mic.36400000', action: 'move', to: 'Kara' }),
    );
    expect(await screen.findByText('Moved to Kara')).toBeInTheDocument();
    expect(await within(t).findByRole('button', { name: 'Why Kara' })).toBeInTheDocument(); // read again
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    await waitFor(() => expect(sent[1]).toEqual({ undo: 'u1' }));
    expect(await screen.findByText('Undone')).toBeInTheDocument();
    expect(await within(t).findByRole('button', { name: 'Why Dana' })).toBeInTheDocument();
  });

  it('corrects a renamed voice using its current displayed name, as the endpoint validates', async () => {
    serve();
    lines(() => KEYED);
    explains('Dana');
    // Dana's voice (2) was renamed Dee since: the turn shows Dee, but the line is still Dana's in the log.
    server.use(http.get('/speakers', () => HttpResponse.json({ speakers: [{ id: 2, label: 'Dee' }], merged: {} })));
    let body: unknown;
    server.use(
      http.post('/people/correct', async ({ request }) => {
        const correction = (await request.json()) as { name: string };
        body = correction;
        if (correction.name !== 'Dee')
          return HttpResponse.json({ error: 'Line is not assigned to that person' }, { status: 404 });
        return HttpResponse.json({ ok: true, undo: 'u1', to: 'Kara' });
      }),
    );
    open();
    const t = await screen.findByRole('region', { name: 'Transcript of the day' });
    const dee = await within(t).findByRole('button', { name: 'Why Dee' });
    expect(within(t).getByRole('button', { name: `Not Dee in the line at ${hms(at(36400))}` })).toBeInTheDocument();
    expect(within(t).queryByRole('button', { name: `Not Dana in the line at ${hms(at(36400))}` })).toBeNull();
    await userEvent.click(dee);
    const why = await screen.findByRole('dialog', { name: 'Why Dee' });
    // Its fixes come with the explanation, and they too go by the name shown. (Looking for no "Not" button at once
    // only passed while the explanation was still loading.)
    expect(await within(why).findByRole('button', { name: 'Not Dee' })).toBeInTheDocument();
    await userEvent.keyboard('{Escape}');
    await userEvent.click(within(t).getByRole('button', { name: 'Not Dee' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Who is this?' })).getByRole('button', { name: 'Kara' }),
    );
    await waitFor(() => expect(body).toMatchObject({ name: 'Dee', id: 'mic.36400000', action: 'move', to: 'Kara' }));
    expect(await screen.findByText('Moved to Kara')).toBeInTheDocument();
  });

  it('opens Why this name and keeps naming and correction in the turn header', async () => {
    serve();
    lines(() => KEYED);
    const asked: string[] = [];
    let body: unknown;
    server.use(
      http.get('/identity/why', ({ request }) => {
        asked.push(new URL(request.url).searchParams.get('line') ?? '');
        return HttpResponse.json({
          name: 'Dana',
          subject: 'line',
          via: 'voice',
          basis: 'owner_tag',
          asserted_at: at(30000),
          source: 'dashboard',
          confidence: 1,
          similarity: 0.71,
          threshold: 0.55,
          voice: { id: 2, label: 'Dana' },
          mark: { kind: 'line', ts_start: at(36400), ts_end: at(36404) },
        });
      }),
      http.post('/people/correct', async ({ request }) => {
        body = await request.json();
        return HttpResponse.json({ ok: true, undo: 'u1', to: 'Kara' });
      }),
    );
    open();
    const t = await screen.findByRole('region', { name: 'Transcript of the day' });
    const name = await within(t).findByRole('button', { name: 'Why Dana' });
    expect(name).toHaveAttribute('aria-haspopup', 'dialog');
    await userEvent.click(name);
    const why = await screen.findByRole('dialog', { name: 'Why Dana' });
    expect(name).toHaveAttribute('aria-expanded', 'true');
    expect(await within(why).findByText(/Its voice was named Dana/)).toBeInTheDocument();
    expect(asked).toEqual(['mic.36400000']); // the line the turn's name comes from

    await userEvent.keyboard('{Escape}');
    await userEvent.click(within(t).getByRole('button', { name: 'Not Dana' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Who is this?' })).getByRole('button', { name: 'Kara' }),
    );
    await waitFor(() => expect(body).toMatchObject({ name: 'Dana', id: 'mic.36400000', action: 'move', to: 'Kara' }));
    await waitFor(() => expect(within(t).getByRole('button', { name: 'Not Dana' })).toHaveFocus());

    const dana = within(t).getByRole('button', { name: 'Why Dana' }).closest('article');
    if (!dana) throw new Error('Missing Dana turn');
    await userEvent.click(within(dana).getByRole('button', { name: 'Why Dana' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Rename this voice…' }));
    expect(await within(t).findByRole('combobox', { name: 'Speaker name' })).toHaveValue('Dana');
  });

  it('offers no correction for a line nobody is named on, and names that voice as Live does', async () => {
    serve();
    let tagged = false;
    lines(() => (tagged ? KEYED.map(e => (e.speaker === 'unknown' ? { ...e, speaker: 'Kara' } : e)) : KEYED));
    let body: unknown;
    server.use(
      http.post('/speakers/tag', async ({ request }) => {
        body = await request.json();
        tagged = true;
        return HttpResponse.json({ ok: true });
      }),
    );
    open();
    const t = await screen.findByRole('region', { name: 'Transcript of the day' });
    await within(t).findByText('Anyone home?');
    expect(within(t).queryByRole('button', { name: `More for the line at ${hms(at(36500))}` })).toBeNull();
    const unknown = within(t).getByText('Anyone home?').closest('article');
    if (!unknown) throw new Error('Missing unknown turn');
    await userEvent.click(within(unknown).getByRole('button', { name: 'Unmatched: say who this is' }));
    const box = await within(t).findByRole('combobox', { name: 'Speaker name' });
    await userEvent.type(box, 'Ka');
    await userEvent.click(await screen.findByRole('option', { name: 'Kara' }));
    await waitFor(() => expect(body).toEqual({ lines: [at(36500).toFixed(3)], name: 'Kara' }));
    expect(await screen.findByText('Tagging as Kara')).toBeInTheDocument();
    expect(await within(t).findByRole('button', { name: 'Why Kara' })).toBeInTheDocument(); // read again, now named
    expect(turn('Why Kara')).toHaveTextContent('Anyone home?');
    // A line with no key cannot offer an exact-key correction.
    explains('Kara');
    await userEvent.click(within(t).getByRole('button', { name: 'Why Kara' }));
    const pop = await screen.findByRole('dialog', { name: 'Why Kara' });
    expect(within(pop).queryByRole('button', { name: /^Not / })).toBeNull();
    // Lines tagged without a voice are renamed from here (no separate Name this voice).
    expect(within(pop).getByRole('button', { name: 'Say who this is…' })).toBeInTheDocument();
  });

  it('shows a name given to an unmatched turn at once, while its lines are still being relabelled', async () => {
    serve();
    lines(() => KEYED); // the day's lines come back unchanged: the microphone service hasn't relabelled them yet
    server.use(http.post('/speakers/tag', () => HttpResponse.json({ ok: true })));
    open();
    const t = await screen.findByRole('region', { name: 'Transcript of the day' });
    const unknown = (await within(t).findByText('Anyone home?')).closest('article') as HTMLElement;
    await userEvent.click(within(unknown).getByRole('button', { name: 'Unmatched: say who this is' }));
    await userEvent.type(within(t).getByRole('combobox', { name: 'Speaker name' }), 'Kara{Enter}');
    expect(await within(unknown).findByText('Tagging as Kara')).toBeInTheDocument();
    expect(within(unknown).getByRole('button', { name: 'Kara: say who this is' })).toHaveTextContent(/^Kara$/);
    expect(unknown).not.toHaveAttribute('data-unmatched');
  });

  describe('a name that matches nobody', () => {
    /** Speaker 1's turn, with what naming it posts recorded by route. */
    function numbered() {
      serve();
      lines(() => KEYED);
      const posted: Record<string, unknown[]> = {};
      const record = (path: string, reply: object) =>
        http.post(path, async ({ request }) => {
          posted[path] = [...(posted[path] ?? []), await request.json()];
          return HttpResponse.json(reply);
        });
      server.use(
        http.get('/speakers', () => HttpResponse.json({ speakers: [{ id: 1, label: 'Speaker 1' }], merged: {} })),
        record('/people/add', { name: 'Sam' }),
        record('/people/claim', { ok: true }),
        record('/speakers/tag', { ok: true }),
      );
      return posted;
    }
    async function nameSpeaker1(typed: string) {
      const { client } = open();
      const t = await screen.findByRole('region', { name: 'Transcript of the day' });
      const turn = (await within(t).findByText('Coffee is ready.')).closest('article');
      if (!turn) throw new Error('Missing Speaker 1 turn');
      await userEvent.click(within(turn).getByRole('button', { name: 'Unmatched: say who this is' }));
      await userEvent.type(await within(t).findByRole('combobox', { name: 'Speaker name' }), `${typed}{Enter}`);
      return { client, t };
    }

    it('asks first, then adds them with just the name and names the voice as them', async () => {
      const posted = numbered();
      const { client } = await nameSpeaker1('Sam');
      const waiting = { groups: [{ id: 1 }, { id: 4 }], count: 2, skipped: [], pending: false };
      client.setQueryData(['voices', 'unknown', 'groups'], waiting);
      const ask = await screen.findByRole('dialog', { name: 'New person' });
      expect(ask).toHaveTextContent('Add Sam as a new person?');
      expect(posted['/people/add']).toBeUndefined();
      await userEvent.click(within(ask).getByRole('button', { name: 'Add' }));
      await waitFor(() => expect(posted['/speakers/tag']).toHaveLength(1));
      expect(posted['/people/add']).toEqual([{ name: 'Sam', number: '' }]);
      expect(posted['/people/claim']).toEqual([{ kind: 'voice', id: 1, name: 'Sam' }]);
      // The waiting voices drop at once, as any voice named from a transcript does.
      expect(client.getQueryData(['voices', 'unknown', 'groups'])).toMatchObject({ groups: [{ id: 4 }], count: 1 });
    });

    it('leaves the box holding the name on Cancel, and adds nobody', async () => {
      const posted = numbered();
      const { t } = await nameSpeaker1('Sam');
      const ask = await screen.findByRole('dialog', { name: 'New person' });
      await userEvent.click(within(ask).getByRole('button', { name: 'Cancel' }));
      await waitFor(() => expect(screen.queryByRole('dialog', { name: 'New person' })).toBeNull());
      expect(await within(t).findByRole('combobox', { name: 'Speaker name' })).toHaveValue('Sam');
      expect(posted).toEqual({});
    });

    it('names the voice as the person an exact match in any case is, with no prompt', async () => {
      const posted = numbered();
      await nameSpeaker1('kara');
      await waitFor(() => expect(posted['/speakers/tag']).toHaveLength(1));
      expect(posted['/people/claim']).toEqual([{ kind: 'voice', id: 1, name: 'Kara' }]);
      expect(posted['/people/add']).toBeUndefined();
      expect(screen.queryByRole('dialog', { name: 'New person' })).toBeNull();
    });
  });
});

it('keeps Space playback available beside Ask while leaving Space in its input alone', async () => {
  serve();
  open();
  await dayLoaded();
  await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
  const panel = await screen.findByRole('dialog', { name: 'Ask' });
  expect(panel).toHaveAttribute('aria-modal', 'false');
  const input = within(panel).getByRole('textbox');
  await waitFor(() => expect(input).toHaveFocus());
  await userEvent.keyboard(' ');
  expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('heading', { level: 1, name: 'Review' }));
  await userEvent.keyboard(' ');
  expect(await screen.findByRole('button', { name: 'Pause' })).toBeInTheDocument();
  expect(panel).toBeVisible();
});

describe('Space from a focused control', () => {
  it('plays without clicking the focused button', async () => {
    serve();
    open();
    await dayLoaded();
    const ask = screen.getByRole('button', { name: 'Ask' });
    ask.focus();
    await userEvent.keyboard(' ');
    expect(await screen.findByRole('button', { name: 'Pause' })).toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: 'Ask' })).toBeNull();
    expect(ask).toHaveFocus();
  });

  it('plays without toggling the focused Show only chip', async () => {
    serve();
    open();
    const filters = await screen.findByRole('group', { name: 'Show only' });
    const dog = await within(filters).findByRole('checkbox', { name: /dog/ });
    dog.focus();
    await userEvent.keyboard(' ');
    expect(await screen.findByRole('button', { name: 'Pause' })).toBeInTheDocument();
    expect(dog).not.toBeChecked();
  });

  it('lets go of a chip a click focused, so Space lights no focus ring', async () => {
    serve();
    open();
    const filters = await screen.findByRole('group', { name: 'Show only' });
    const dog = await within(filters).findByRole('checkbox', { name: /dog/ });
    await userEvent.click(dog); // a pointer focuses it (and picks it)
    expect(dog).toHaveFocus();
    await userEvent.keyboard(' ');
    expect(await screen.findByRole('button', { name: 'Pause' })).toBeInTheDocument();
    expect(dog).not.toHaveFocus();
    expect(dog).toBeChecked(); // the click's pick stands; Space only played
  });

  it('keeps focus, and its ring, on a control reached from the keyboard', async () => {
    serve();
    open();
    const filters = await screen.findByRole('group', { name: 'Show only' });
    const dog = await within(filters).findByRole('checkbox', { name: /dog/ });
    dog.focus(); // as Tab would: no pointer
    await userEvent.keyboard(' ');
    expect(await screen.findByRole('button', { name: 'Pause' })).toBeInTheDocument();
    expect(dog).toHaveFocus();
  });
});

it('searches unseen full-day summary blocks from the transcript tab and seeks the result', async () => {
  serve();
  const requestedDays: (string | null)[] = [];
  const semantic = vi.fn();
  server.use(
    http.get('/review/summary', ({ request }) => {
      const params = new URL(request.url).searchParams;
      requestedDays.push(params.get('day'));
      return HttpResponse.json({
        day: DAY,
        status: 'ready',
        blocks:
          params.get('day') === DAY ? [{ start: at(36015), end: at(36020), text: 'Garden sandwiches', tags: [] }] : [],
      });
    }),
    http.get('/review/search', () => {
      semantic();
      return HttpResponse.json({ results: [] });
    }),
  );
  open();
  await screen.findByRole('slider', { name: /Timeline/ });
  await waitFor(() => expect(requestedDays).toContain(DAY));
  const input = screen.getByRole('combobox', { name: 'Search transcript and summary' });
  fireEvent.change(input, { target: { value: 'sandwiches' } });
  const hit = await screen.findByRole('option', { name: /Summary.*Garden sandwiches/ });
  await userEvent.click(hit);
  await playheadAt(at(36015));
  expect(semantic).not.toHaveBeenCalled();
  fireEvent.keyDown(input, { key: 'Escape' });
  expect(input).toHaveValue('');
  expect(input).toHaveFocus();
});

it("keeps the day's transcript under the search results, dimmed to the matches and highlighted", async () => {
  serve();
  open();
  const t = await screen.findByRole('region', { name: 'Transcript of the day' });
  await within(t).findByText('Coffee is ready.');
  fireEvent.change(screen.getByRole('combobox', { name: 'Search transcript and summary' }), {
    target: { value: 'dog' },
  });
  expect(await screen.findByRole('option', { name: /Where did the dog go\?/ })).toBeInTheDocument(); // results stay
  expect(t).toBeVisible();
  const turn = (text: string) => [...t.querySelectorAll('article')].find(a => a.textContent?.includes(text));
  await waitFor(() => expect(turn('Coffee is ready.')).toHaveAttribute('data-missed'));
  expect(turn('Where did the dog go?')).not.toHaveAttribute('data-missed');
  expect(turn('Where did the dog go?')?.querySelector('mark')).toHaveTextContent(/^dog$/);
});

it('adds fetched activity kinds to Events filters and discloses bounded marker coverage', async () => {
  serve();
  server.use(
    http.get('/activities', () =>
      HttpResponse.json({
        items: [
          {
            id: 'reading-one',
            camera: 'cam0',
            start: at(36200),
            end: at(36202),
            count: null,
            name: null,
            kind: 'reading_book',
            person: null,
            promoted: false,
          },
          {
            id: 'reading-two',
            camera: 'cam0',
            start: at(36400),
            end: at(36402),
            count: null,
            name: null,
            kind: 'reading_book',
            person: null,
            promoted: false,
          },
        ],
        truncated: true,
      }),
    ),
  );
  open();
  const filters = await screen.findByRole('group', { name: 'Show only' });
  const chip = await within(filters).findByRole('checkbox', { name: /Reading book/ });
  expect(chip).toHaveAccessibleName(/Reading book.*2$/);
  expect(within(filters).getAllByRole('checkbox', { name: /Reading book/ })).toHaveLength(1);
  await userEvent.click(chip);
  expect(chip).toBeChecked();
  // The note shows once it has held for 1.5 s, so the landing's pass through the day before doesn't flash it.
  expect(
    await screen.findByText(/Activity markers show only the latest 200 for/, undefined, { timeout: 3000 }),
  ).toBeInTheDocument();
});

it('shows waiting state and discloses incomplete activity markers from a visible neighboring day', async () => {
  serve();
  const views = vi.spyOn(ReviewEngine.prototype, 'setView');
  server.use(
    http.get('/activities', ({ request }) =>
      HttpResponse.json({ items: [], truncated: new URL(request.url).searchParams.get('day') === OTHER_DAY }),
    ),
  );
  const { client } = open();
  await playheadAt(at(36011));
  const engine = views.mock.contexts[0];
  if (!(engine instanceof ReviewEngine)) throw new Error('missing Review engine');
  act(() => {
    engine.ui.setState({ waiting: true });
    engine.setView(D0 - 600, D0 + 600, true);
  });
  // First on the status line, so a phone's truncation can't hide it behind the day's stats.
  expect(screen.getByText(/^Waiting for recordings at the playhead…/)).toBeInTheDocument();
  expect(await screen.findByText(new RegExp(`Activity markers.*${dayWords(OTHER_DAY)}`))).toBeInTheDocument();
  server.use(http.get('/activities', () => HttpResponse.json({ error: 'Unavailable' }, { status: 400 })));
  // The hook still observes the cached neighbor, so a failed refresh must disclose the missing lane there.
  await act(async () => {
    await client.invalidateQueries({ queryKey: reviewKey.activities(OTHER_DAY) });
  });
  expect(
    await screen.findByText(new RegExp(`Activity markers unavailable for.*${dayWords(OTHER_DAY)}`)),
  ).toBeInTheDocument();
  act(() => engine.ui.setState({ waiting: false }));
});

it('retries recording sources without restarting summaries or detector analysis', async () => {
  serve();
  vi.mocked(daySegments).mockRejectedValue(new Error('No video listing'));
  const { client } = open();
  const retry = await screen.findByRole('button', { name: 'Retry recordings' });
  const invalidates = vi.spyOn(client, 'invalidateQueries');
  vi.mocked(daySegments).mockResolvedValue(SEGMENTS);
  await userEvent.click(retry);
  expect(invalidates.mock.calls.length).toBeGreaterThan(0);
  expect(
    invalidates.mock.calls
      .map(([options]) => options?.queryKey?.[3])
      .every(key => ['transcript', 'segments', 'spans'].includes(String(key))),
  ).toBe(true);
});

it('passes backend kind promotion status through fetched activity chips to Show-only tooltips', async () => {
  serve();
  server.use(
    http.get('/activities', () =>
      HttpResponse.json({
        items: [
          {
            id: 'rated-candidate',
            camera: 'cam0',
            start: at(36200),
            end: at(36202),
            count: null,
            name: null,
            kind: 'reading_book',
            person: null,
            promoted: false,
          },
        ],
        truncated: false,
        promotion: [
          {
            kind: 'reading_book',
            camera: null,
            status: 'needs_correct_ratings',
            remaining_ratings: null,
            rated: 10,
            right: 7,
            precision: 0.7,
          },
        ],
      }),
    ),
  );
  open();
  const filters = await screen.findByRole('group', { name: 'Show only' });
  const chip = await within(filters).findByRole('checkbox', { name: /Reading book/ });
  act(() => chip.focus());
  const tooltip = await screen.findByRole('tooltip');
  expect(tooltip).toHaveTextContent('Not counted in Ask: more Yes answers in Known, Activities');
  expect(tooltip).not.toHaveTextContent('needs 0');
});
