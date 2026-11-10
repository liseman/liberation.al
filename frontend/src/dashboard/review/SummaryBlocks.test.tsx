import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createStore } from 'zustand/vanilla';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { hm } from '../lib/format';
import type { Block, Blocks, DayRollup, DetectedActivity } from './api';
import { EngineContext } from './context';
import type { ReviewEngine, ReviewUi } from './engine';
import { nextAsk, partOfDay, picked, SummaryBlocks } from './SummaryBlocks';
import type { Chip } from './types';

const DAY = '2026-10-27';
const D0 = new Date(2026, 9, 27).getTime() / 1000;
const at = (h: number, m = 0) => D0 + h * 3600 + m * 60;

const BLOCKS: Block[] = [
  // as written by the server: no time, no full stop, and person: and topic: tags
  {
    start: at(8),
    end: at(11),
    text: 'Worked with Dana at the desk for about 3 hours',
    tags: ['person:dana', 'topic:work'],
  },
  { start: at(12), end: at(13), text: 'Lunch with Dana in the kitchen', tags: ['person:dana', 'topic:meal'] },
  { start: at(15), end: at(16), text: '3 decisions made about the garden', tags: ['topic:decision', 'topic:planning'] },
];

let asked: { start: number; end: number }[];
let replies: (Partial<Blocks> | { httpStatus: number })[];
let activities: DetectedActivity[];

const activity = (start: number, camera: string, kind: string, promoted = true): DetectedActivity => ({
  id: `${camera}:${start}`,
  camera,
  start,
  end: start + 20,
  count: null,
  name: null,
  kind,
  person: null,
  promoted,
});

function serveBlocks() {
  server.use(
    http.get('/review/summary', ({ request }) => {
      const q = new URL(request.url).searchParams;
      const range = { start: Number(q.get('start')), end: Number(q.get('end')) };
      asked.push(range);
      const r = replies.shift() ?? {};
      if ('httpStatus' in r) return HttpResponse.json({ error: 'broken' }, { status: r.httpStatus });
      return HttpResponse.json({ ...range, level: 3600, status: 'ready', blocks: BLOCKS, ...r });
    }),
    http.get('/activities', () => HttpResponse.json({ items: activities, truncated: false, promotion: [] })),
  );
}

/** A stand-in for the Review engine: the state the column reads, and the two calls it makes. */
function show(state: Partial<ReviewUi> = {}, chips: Chip[] = []) {
  const ui = createStore<Partial<ReviewUi>>(() => ({
    range: { t0: at(0), t1: at(24) },
    sel: new Set<string>(),
    cameras: [],
    t: null,
    rangeIntent: 0,
    ...state,
  }));
  const engine = { ui, navigationVersion: 0, seekPlay: vi.fn(), seekActivity: vi.fn(), zoomTo: vi.fn(), chips };
  engine.seekPlay.mockImplementation(() => {
    engine.navigationVersion++;
  });
  renderPage(
    <EngineContext value={engine as unknown as ReviewEngine}>
      <SummaryBlocks day={DAY} />
    </EngineContext>,
  );
  return engine;
}

const rows = () => within(screen.getByRole('list'));
const stretch = (text: RegExp) => rows().getByRole('button', { name: text });
const findStretch = async (text: RegExp) =>
  within(await screen.findByRole('list')).findByRole('button', { name: text });
const findSummaryText = async (text: string) => within(await screen.findByRole('list')).findByText(text);

beforeEach(() => {
  asked = [];
  replies = [];
  activities = [];
  serveBlocks();
});
afterEach(() => vi.useRealTimers());

describe('the summary by time', () => {
  it('asks for what the timeline shows, and reads it as stretches under parts of the day', async () => {
    show();
    await findSummaryText('Worked with Dana at the desk for about 3 hours');
    expect(asked).toEqual([{ start: at(0), end: at(24) }]);
    const items = within(screen.getByRole('list')).getAllByRole('listitem');
    expect(items.map(li => li.textContent)).toEqual([
      'Morning',
      `${hm(at(8))} to ${hm(at(11))}Worked with Dana at the desk for about 3 hours`,
      'Afternoon',
      `${hm(at(12))} to ${hm(at(13))}Lunch with Dana in the kitchen`,
      `${hm(at(15))} to ${hm(at(16))}3 decisions made about the garden`,
    ]);
    expect(screen.getByText('the whole day on the timeline, by the hour')).toBeInTheDocument();
  });

  it('puts each activity in its stretch as a chip, the rest at their own time, and plays either from its start', async () => {
    activities = [
      activity(at(12, 20), 'kitchen', 'drinking'),
      activity(at(14, 5), 'patio', 'walked_past', false), // between stretches, and unconfirmed
      activity(at(12, 40), 'den', 'drinking'), // another camera than Show only picked
    ];
    const engine = show({ cameras: ['kitchen', 'patio'] });
    await findSummaryText('Lunch with Dana in the kitchen');
    const items = await waitFor(() => {
      const found = within(screen.getByRole('list')).getAllByRole('listitem');
      expect(found).toHaveLength(6);
      return found;
    });
    expect(items.map(li => li.textContent)).toEqual([
      'Morning',
      `${hm(at(8))} to ${hm(at(11))}Worked with Dana at the desk for about 3 hours`,
      'Afternoon',
      `${hm(at(12))} to ${hm(at(13))}Lunch with Dana in the kitchen${hm(at(12, 20))} Drinking · kitchen`,
      `${hm(at(14, 5))}${hm(at(14, 5))} Walked past? · patio`,
      `${hm(at(15))} to ${hm(at(16))}3 decisions made about the garden`,
    ]);
    await userEvent.click(rows().getByRole('button', { name: /Walked past\? · patio/ }));
    await userEvent.click(rows().getByRole('button', { name: /Drinking · kitchen/ }));
    expect(engine.seekActivity.mock.calls).toEqual([
      [at(14, 5), 'patio', { play: true }],
      [at(12, 20), 'kitchen', { play: true }],
    ]);
    expect(engine.seekPlay).not.toHaveBeenCalled();
  });

  it('waits for the day to load before asking', async () => {
    show({ range: null });
    expect(await screen.findByLabelText('Loading the summary')).toBeInTheDocument();
    expect(asked).toEqual([]);
  });

  it('asks again for the new range when the timeline zooms, keeping the last stretches meanwhile', async () => {
    const engine = show();
    await findSummaryText('Lunch with Dana in the kitchen');
    replies.push({ blocks: [{ start: at(12), end: at(12, 30), text: 'Made sandwiches', tags: [] }] });
    act(() => engine.ui.setState({ range: { t0: at(12), t1: at(13) } }));
    expect(rows().getByText('Lunch with Dana in the kitchen')).toBeInTheDocument(); // until the finer ones arrive
    await findSummaryText('Made sandwiches');
    expect(asked.at(-1)).toEqual({ start: at(12), end: at(13) });
  });

  it('seeks a picked stretch without refetching the list or losing its selection on later seeks', async () => {
    const engine = show();
    engine.zoomTo.mockImplementation((start: number, end: number) =>
      engine.ui.setState({ range: { t0: start, t1: end } }),
    );
    const lunch = await findStretch(/Lunch with Dana in the kitchen/);
    const list = screen.getByRole('list');
    await userEvent.click(lunch);
    expect(engine.seekPlay).toHaveBeenCalledWith(at(12)); // plays from there
    expect(engine.zoomTo).not.toHaveBeenCalled();
    expect(lunch).toHaveAttribute('aria-pressed', 'true');
    act(() => engine.ui.setState({ t: at(15, 5) }));
    expect(screen.getByRole('list')).toBe(list);
    expect(stretch(/Lunch/)).toBe(lunch);
    expect(lunch).toHaveAttribute('aria-pressed', 'true');
    expect(stretch(/3 decisions/)).toHaveAttribute('aria-current', 'time');
    expect(asked).toEqual([{ start: at(0), end: at(24) }]);
  });

  it('plays a stretch on every click, the same one twice and two in turn, without refetching the list', async () => {
    const engine = show();
    const lunch = await findStretch(/Lunch with Dana in the kitchen/);
    const list = screen.getByRole('list');
    await userEvent.click(lunch);
    act(() => engine.ui.setState({ t: at(12) })); // already at its start: a second click still plays it
    await userEvent.click(lunch);
    await userEvent.click(stretch(/3 decisions/));
    await userEvent.click(lunch);
    expect(engine.seekPlay.mock.calls.map(([t]) => t)).toEqual([at(12), at(12), at(15), at(12)]);
    expect(screen.getByRole('list')).toBe(list);
    expect(asked).toEqual([{ start: at(0), end: at(24) }]);
  });

  it('keeps the selected block by its start when refinement inserts earlier blocks', async () => {
    const engine = show();
    const lunch = await findStretch(/Lunch with Dana in the kitchen/);
    await userEvent.click(lunch);
    replies.push({
      blocks: [{ start: at(6), end: at(7), text: 'Breakfast outside', tags: [] }, ...BLOCKS],
    });
    act(() => engine.ui.setState({ range: { t0: at(6), t1: at(18) }, rangeIntent: 1 }));
    await findSummaryText('Breakfast outside');
    expect(stretch(/Lunch/)).toBe(lunch);
    expect(lunch).toHaveAttribute('aria-pressed', 'true');
    expect(stretch(/Worked with Dana at the desk/)).toHaveAttribute('aria-pressed', 'false');
    expect(asked).toHaveLength(2);
  });

  it('keeps the selected summary list when its own seek moves the viewport across midnight', async () => {
    replies.push({
      blocks: [{ start: D0 - 120, end: D0 + 60, text: 'Late conversation across midnight', tags: [] }, ...BLOCKS],
    });
    const engine = show();
    engine.seekPlay.mockImplementation(() => {
      engine.navigationVersion++;
      engine.ui.setState({ t: D0 - 120, range: { t0: D0 - 43200, t1: D0 + 43200 } });
    });
    const previous = await findStretch(/Late conversation across midnight/);
    const list = screen.getByRole('list');
    await userEvent.click(previous);
    expect(previous).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('list')).toBe(list);
    expect(asked).toEqual([{ start: D0, end: at(24) }]);
  });

  it('highlights the stretches with a chip Show only picked, and Only these hides the rest', async () => {
    show({ sel: new Set(['event:decisions']) }); // the Decisions section's chip: stretches tagged topic:decision
    await findSummaryText('Lunch with Dana in the kitchen');
    expect(stretch(/3 decisions/)).toHaveAttribute('data-picked');
    expect(stretch(/Lunch/)).not.toHaveAttribute('data-picked');
    await userEvent.click(screen.getByRole('switch', { name: /Show only the stretches/ }));
    expect(screen.queryByText('Lunch with Dana in the kitchen')).not.toBeInTheDocument();
    expect(rows().getByText('3 decisions made about the garden')).toBeInTheDocument();
  });

  it('highlights a person picked in Show only', async () => {
    show({ sel: new Set(['person:dana']) });
    await findSummaryText('Lunch with Dana in the kitchen');
    expect(stretch(/Lunch/)).toHaveAttribute('data-picked');
    expect(stretch(/Worked with Dana at the desk/)).toHaveAttribute('data-picked');
    expect(stretch(/3 decisions/)).not.toHaveAttribute('data-picked');
  });

  it('does not highlight the patio objects merely because Alice has presence all day', async () => {
    const text = 'Saw black container and orange toy vehicle on patio';
    replies.push({ blocks: [{ start: at(8), end: at(9), text, tags: [] }] });
    show({ sel: new Set(['person:alice']) }, [
      {
        id: 'person:alice',
        kind: 'person',
        label: 'Alice',
        spans: [[at(0), at(24)]],
        total_s: 86400,
        count: 1,
      },
    ]);
    await findSummaryText(text);
    expect(stretch(/Saw black container/)).not.toHaveAttribute('data-picked');
    await userEvent.click(screen.getByRole('switch', { name: /Show only the stretches/ }));
    expect(screen.queryByText(text)).not.toBeInTheDocument();
    expect(screen.getByText(/No confirmed summary matches/)).toBeInTheDocument();
  });

  it('marks missing camera attribution as unverified and cannot match it through a person tag', async () => {
    show({ sel: new Set(['person:dana']), cameras: ['patio'] });
    await findSummaryText('Lunch with Dana in the kitchen');
    expect(stretch(/Lunch/)).not.toHaveAttribute('data-picked');
    expect(screen.getByText(/Camera attribution is unavailable/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('switch', { name: /Show only the stretches/ }));
    expect(screen.queryByText('Lunch with Dana in the kitchen')).not.toBeInTheDocument();
    expect(screen.getByText(/No confirmed summary matches/)).toBeInTheDocument();
  });

  it('requires the selected person and event together, with alternatives inside each facet', async () => {
    show({ sel: new Set(['person:dana', 'person:alice', 'event:meals', 'event:visits']) });
    await findSummaryText('Lunch with Dana in the kitchen');
    expect(stretch(/Lunch/)).toHaveAttribute('data-picked');
    expect(stretch(/Worked/)).not.toHaveAttribute('data-picked');
    expect(stretch(/3 decisions/)).not.toHaveAttribute('data-picked');
  });

  it('does not substitute semantic topics for missing speech or action attribution', async () => {
    show({ sel: new Set(['event:speech', 'event:actions']) });
    await findSummaryText('Lunch with Dana in the kitchen');
    expect(stretch(/Lunch/)).not.toHaveAttribute('data-picked');
    expect(screen.getByText(/Some summary blocks lack attribution/)).toBeInTheDocument();
  });

  it('keeps activity attribution unverified when the block only has semantic topics', async () => {
    show({ sel: new Set(['event:activity:crossing']) });
    await findSummaryText('Lunch with Dana in the kitchen');
    expect(stretch(/Worked/)).not.toHaveAttribute('data-picked');
    expect(screen.getByText(/Some summary blocks lack attribution/)).toBeInTheDocument();
  });

  it('has no Only these while Show only picks nothing', async () => {
    show();
    await findSummaryText('Lunch with Dana in the kitchen');
    expect(screen.queryByRole('switch')).not.toBeInTheDocument();
    expect(stretch(/3 decisions/)).not.toHaveAttribute('data-picked');
  });

  it('marks the stretch at the playhead', async () => {
    const engine = show({ t: at(12, 20) });
    await findSummaryText('Lunch with Dana in the kitchen');
    expect(stretch(/Lunch/)).toHaveAttribute('aria-current', 'time');
    act(() => engine.ui.setState({ t: at(15, 5) }));
    await waitFor(() => {
      expect(stretch(/3 decisions/)).toHaveAttribute('aria-current', 'time');
      expect(stretch(/Lunch/)).not.toHaveAttribute('aria-current');
    });
  });

  it('says when a stretch is still being written, and asks again until it is', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    replies.push({ status: 'pending', blocks: [] });
    show();
    expect(await screen.findByText('Summarizing this stretch…')).toBeInTheDocument();
    await act(() => vi.advanceTimersByTimeAsync(5000));
    await findSummaryText('Lunch with Dana in the kitchen');
    expect(asked).toHaveLength(2);
  });

  it('says when no summary is written without claiming nothing was recorded', async () => {
    replies.push({ blocks: [] });
    show();
    expect(await screen.findByText('No written summary for this stretch.')).toBeInTheDocument();
  });

  it('says when the summary could not load, and tries again', async () => {
    replies.push({ httpStatus: 500 });
    show();
    await userEvent.click(await screen.findByRole('button', { name: /Retry/ }));
    await findSummaryText('Lunch with Dana in the kitchen');
    await waitFor(() => expect(asked).toHaveLength(2));
  });
});

describe('summary attribution facets', () => {
  const block: Block = {
    start: at(8),
    end: at(9),
    text: 'Lunch with Alice beside the container',
    tags: ['person:alice', 'object:container', 'topic:meal', 'camera:patio'],
  };

  it.each([
    [['person:alice', 'person:dana'], true],
    [['person:alice', 'object:container'], true],
    [['person:dana', 'object:container'], false],
    [['person:alice', 'object:vehicle'], false],
    [['person:alice', 'event:meals'], true],
    [['person:alice', 'event:speech'], false],
    [['person:alice', 'event:actions'], false],
    [['object:container', 'event:work'], false],
  ] as const)('matches selected chip facets %j: %s', (ids, expected) => {
    expect(picked(block, new Set(ids))).toBe(expected);
  });

  it('requires one explicitly attributed camera and all selected chip facets', () => {
    expect(picked(block, new Set(['person:alice']), ['patio', 'kitchen'])).toBe(true);
    expect(picked(block, new Set(['person:alice']), ['kitchen'])).toBe(false);
    expect(picked(block, new Set(['person:dana']), ['patio'])).toBe(false);
  });

  it('matches an evidence event kind only when the block explicitly attributes it', () => {
    expect(picked({ ...block, tags: ['event:speech'] }, new Set(['event:speech']))).toBe(true);
    expect(picked({ ...block, tags: ['event:actions'] }, new Set(['event:actions']))).toBe(true);
    expect(picked({ ...block, tags: ['topic:activity:crossing'] }, new Set(['event:activity:crossing']))).toBe(false);
  });
});

describe('zooming and playing', () => {
  it('keeps the stretches showing, marked Refining…, until finer ones come, then swaps them in', async () => {
    const engine = show();
    await findSummaryText('Lunch with Dana in the kitchen');
    replies.push({ status: 'pending', blocks: [] }); // the finer level isn't written yet
    replies.push({ blocks: [{ start: at(12), end: at(12, 30), text: 'Made sandwiches', tags: [] }] });
    vi.useFakeTimers({ shouldAdvanceTime: true });
    act(() => engine.ui.setState({ range: { t0: at(12), t1: at(13) } }));
    await waitFor(() => expect(asked).toHaveLength(2));
    expect(rows().getByText('Lunch with Dana in the kitchen')).toBeInTheDocument(); // not emptied
    expect(screen.getByRole('status')).toHaveTextContent('Refining…');
    await act(() => vi.advanceTimersByTimeAsync(5000)); // asked again while it's being written
    await findSummaryText('Made sandwiches');
    expect(screen.queryByText('Lunch with Dana in the kitchen')).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('');
  });

  it("doesn't ask again while the view follows the playhead within what's shown", async () => {
    const engine = show({ range: { t0: at(9), t1: at(12) } });
    await findSummaryText('Lunch with Dana in the kitchen');
    act(() => engine.ui.setState({ range: { t0: at(9, 30), t1: at(12, 30) } })); // panned, same span
    await new Promise(r => setTimeout(r, 50));
    expect(asked).toHaveLength(1);
  });
});

describe('nextAsk', () => {
  const v = (t0: number, t1: number) => ({ t0, t1 });
  const shown = [
    { start: 100, end: 200, text: '', tags: [] },
    { start: 200, end: 400, text: '', tags: [] },
  ];
  it.each([
    ['the first range', null, v(100, 300), v(100, 300)],
    ['a pan within what is shown', v(120, 320), v(150, 350), v(120, 320)],
    ['a pan past what is shown', v(120, 320), v(250, 450), v(250, 450)],
    ['a zoom', v(120, 320), v(150, 250), v(150, 250)],
  ] as const)('%s', (_why, asked, range, want) => {
    expect(nextAsk(asked, range, shown)).toEqual(want);
  });
  it('asks nothing before the day loads', () => expect(nextAsk(v(1, 2), null, shown)).toBeNull());
});

describe("a day's parts and its sentence", () => {
  const rolled = (parts: DayRollup['parts'], text: string | null = 'A work day at home'): DayRollup => ({
    day: DAY,
    start: at(0),
    end: at(24),
    text,
    status: text ? 'ready' : 'pending',
    parts,
  });
  const morning = {
    name: 'Morning',
    start: at(5),
    end: at(12),
    text: 'Worked at the desk with Dana all morning',
    status: 'ready' as const,
  };

  it("opens with the day's sentence and reads each part as one sentence, its stretches folded under it", async () => {
    activities = [activity(at(9, 30), 'kitchen', 'drinking')];
    replies = [{ days: [rolled([morning])] }];
    const engine = show();
    await findSummaryText('A work day at home');
    await waitFor(() => expect(screen.getByRole('button', { name: /Drinking/ })).toBeInTheDocument());
    const items = within(screen.getByRole('list')).getAllByRole('listitem');
    expect(items.map(li => li.textContent)).toEqual([
      'A work day at home',
      `MorningShow 1 stretch${hm(at(8))} to ${hm(at(11))}Worked at the desk with Dana all morning${hm(at(9, 30))} Drinking · kitchen`,
      'Afternoon', // no sentence for the afternoon yet: its stretches as before
      `${hm(at(12))} to ${hm(at(13))}Lunch with Dana in the kitchen`,
      `${hm(at(15))} to ${hm(at(16))}3 decisions made about the garden`,
    ]);
    expect(screen.getByText('the whole day on the timeline, by part of the day')).toBeInTheDocument();
    await userEvent.click(stretch(/^.*Worked at the desk with Dana all morning/));
    expect(engine.seekPlay).toHaveBeenLastCalledWith(at(8)); // from the part's first stretch
    await userEvent.click(screen.getByRole('button', { name: 'Show 1 stretch' }));
    expect(screen.getByRole('button', { name: 'Hide 1 stretch' })).toHaveAttribute('aria-expanded', 'true');
    expect(stretch(/Worked with Dana at the desk for about 3 hours/)).toBeInTheDocument();
  });

  it('says a part is being summed up while its sentence is written', async () => {
    replies = [{ status: 'partial', days: [rolled([{ ...morning, text: null, status: 'pending' }], null)] }];
    show();
    expect(await findSummaryText('Summing up the morning…')).toBeInTheDocument();
    expect(screen.queryByText('A work day at home')).not.toBeInTheDocument();
  });

  it('lists single stretches again under Only these', async () => {
    replies = [{ days: [rolled([morning])] }, { days: [rolled([morning])] }];
    show({ sel: new Set(['topic:work']) });
    await findSummaryText('A work day at home');
    await userEvent.click(screen.getByRole('switch', { name: /Show only/ }));
    expect(await findStretch(/Worked with Dana at the desk for about 3 hours/)).toBeInTheDocument();
    expect(screen.queryByText('A work day at home')).not.toBeInTheDocument();
  });
});

it('ends the day in the evening, as the server sums it up', () => {
  expect([4, 5, 12, 17, 23].map(h => partOfDay(at(h)))).toEqual([
    'Night',
    'Morning',
    'Afternoon',
    'Evening',
    'Evening',
  ]);
});
