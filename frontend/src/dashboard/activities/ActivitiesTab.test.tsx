import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { hm, isoDay } from '../lib/format';
import type { DetectedActivity } from '../review/api';
import { ActivitiesTab, thinnedText } from './ActivitiesTab';
import type { ActivityKind } from './kinds';
import { FILTER_KEY } from './recent';

const TODAY = isoDay(new Date());
const at = (h: number, m = 0) =>
  new Date(`${TODAY}T${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:00`).getTime() / 1000;
let items: DetectedActivity[];
let posts: { url: string; body: unknown }[];
let batches: string[][]; // the days of each batched /activities read
let defined: (ActivityKind & { deleted?: boolean })[];

const activity = (kind: string, name: string, more: Partial<ActivityKind> = {}): ActivityKind => ({
  kind,
  name,
  description: '',
  rule: false,
  classes: { ntu: [], kinetics: [] },
  learned: { ntu: [], kinetics: [] },
  answers: 0,
  promotion: null,
  ...more,
});

const clip = (id: string, start: number, kind: string, camera = 'kitchen'): DetectedActivity => ({
  id,
  camera,
  start,
  end: start + 6,
  count: null,
  name: null,
  kind,
  person: null,
  promoted: false,
});

beforeEach(() => {
  batches = [];
  items = [clip('early', at(7, 22), 'drinking'), clip('late', at(18, 40), 'squats', 'living-room')];
  posts = [];
  defined = [activity('drinking', 'Drinking', { answers: 6, classes: { ntu: ['drink water'], kinetics: [] } })];
  const change = (kind: string, deleted: boolean) => {
    defined = defined.map(k => (k.kind === kind ? { ...k, deleted } : k));
    if (deleted) items = items.filter(i => i.kind !== kind);
    const found = defined.find(k => k.kind === kind);
    return HttpResponse.json({ ok: true, kind: found });
  };
  server.use(
    http.get('/activity/kinds', () =>
      HttpResponse.json({
        kinds: defined.filter(k => !k.deleted),
        deleted: defined.filter(k => k.deleted).map(k => ({ kind: k.kind, name: k.name, answers: k.answers })),
      }),
    ),
    http.post('/activity/kinds', async ({ request }) => {
      const body = (await request.json()) as { name: string; description: string };
      posts.push({ url: '/activity/kinds', body });
      const made = activity(body.name.toLowerCase(), body.name, {
        description: body.description,
        classes: { ntu: ['squat down'], kinetics: ['squat'] },
      });
      defined = [...defined, made];
      return HttpResponse.json({ ok: true, kind: made });
    }),
    http.post('/activity/kinds/delete', async ({ request }) => {
      const body = (await request.json()) as { kind: string };
      posts.push({ url: '/activity/kinds/delete', body });
      return change(body.kind, true);
    }),
    http.post('/activity/kinds/restore', async ({ request }) => {
      const body = (await request.json()) as { kind: string };
      posts.push({ url: '/activity/kinds/restore', body });
      return change(body.kind, false);
    }),
    // The tab reads its week in one request (?days=a,b,...), and one day again after a change (?day=).
    http.get('/activities', ({ request }) => {
      const query = new URL(request.url).searchParams;
      const listing = (day: string) => ({
        items: day === TODAY ? items : [],
        truncated: false,
        promotion: [
          {
            kind: 'walked_past',
            camera: null,
            status: 'counted',
            remaining_ratings: 0,
            rated: 25,
            right: 23,
            precision: 0.92,
          },
          {
            kind: 'drinking',
            camera: null,
            status: 'needs_ratings',
            remaining_ratings: 4,
            rated: 6,
            right: 6,
            precision: 1,
          },
        ],
      });
      const days = query.get('days');
      if (days === null) return HttpResponse.json(listing(query.get('day') ?? ''));
      batches.push(days.split(','));
      return HttpResponse.json({ days: Object.fromEntries(days.split(',').map(day => [day, listing(day)])) });
    }),
    http.post('/activity/feedback', async ({ request }) => {
      const body = (await request.json()) as { activity: string; verdict: 'right' | 'wrong' };
      posts.push({ url: '/activity/feedback', body });
      items = items.map(item => (item.id === body.activity ? { ...item, feedback: body.verdict } : item));
      return HttpResponse.json({ ok: true, activity: body.activity, verdict: body.verdict });
    }),
    http.post('/activity/name', async ({ request }) => {
      const body = await request.json();
      posts.push({ url: '/activity/name', body });
      return HttpResponse.json({ ok: true });
    }),
  );
});

const card = (name: RegExp) => screen.getByRole('listitem', { name });

it('lists the kinds by how far each is from counting, and the clips to check newest first', async () => {
  renderPage(<ActivitiesTab />);
  await kindsListed();
  expect(await screen.findByText('2 clips to check ›')).toBeInTheDocument();
  const kinds = within(screen.getByRole('complementary', { name: 'Activity kinds' }));
  expect(kinds.getByText('Counted in Ask')).toBeInTheDocument();
  // Sentence-case names and one progress wording.
  expect(kinds.getByText('Drinking')).toBeInTheDocument();
  expect(kinds.getByText('4 more answers to count')).toBeInTheDocument();
  const asks = screen.getAllByText(/^Is this /).map(q => q.textContent);
  expect(asks).toEqual(['Is this “Squats”?', 'Is this “Drinking”?']);
  // The week comes in one request, not one per day (#52).
  expect(batches).toHaveLength(1);
  expect(batches[0]).toHaveLength(7);
  expect(batches[0]?.[0]).toBe(TODAY);
  expect(within(card(/Drinking at/)).getByRole('button', { name: /Play Drinking at/ })).toBeInTheDocument();
  // Yes, No, and Not sure weigh the same: none is the filled one.
  const answers = within(card(/Drinking at/)).getAllByRole('button', { name: /^(Yes|No|Not sure)$/ });
  expect(answers.map(b => [b.textContent, b.dataset.variant])).toEqual([
    ['Yes', 'default'],
    ['No', 'default'],
    ['Not sure', 'default'],
  ]);
});

it('saves a Yes as a right rating and takes the clip off the list', async () => {
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Drinking”?');
  await userEvent.click(within(card(/Drinking at/)).getByRole('button', { name: 'Yes' }));
  await waitFor(() => expect(screen.queryByText('Is this “Drinking”?')).not.toBeInTheDocument());
  expect(posts).toEqual([
    { url: '/activity/feedback', body: { activity: 'early', start: at(7, 22), verdict: 'right' } },
  ]);
  expect(screen.getByText('1 clip to check ›')).toBeInTheDocument();
});

it('sets a clip aside on Not sure without saving anything', async () => {
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Squats”?');
  await userEvent.click(within(card(/Squats at/)).getByRole('button', { name: 'Not sure' }));
  expect(screen.queryByText('Is this “Squats”?')).not.toBeInTheDocument();
  expect(posts).toEqual([]);
});

it('saves a name in the owner words for one clip, trimmed, when the field is left', async () => {
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Drinking”?');
  // The name field opens on demand, with focus in it.
  expect(within(card(/Drinking at/)).queryByRole('combobox', { name: 'Call it' })).toBeNull();
  await userEvent.click(within(card(/Drinking at/)).getByRole('button', { name: 'Call it something else' }));
  const field = within(card(/Drinking at/)).getByRole('combobox', { name: 'Call it' });
  expect(field).toHaveFocus();
  await userEvent.type(field, '  morning coffee ');
  await userEvent.tab();
  await waitFor(() =>
    expect(posts).toEqual([
      { url: '/activity/name', body: { activity: 'early', start: at(7, 22), name: 'morning coffee' } },
    ]),
  );
});

it('says when nothing is waiting', async () => {
  items = [];
  renderPage(<ActivitiesTab />);
  await kindsListed();
  expect(await screen.findByText('Nothing to check')).toBeInTheDocument();
  expect(screen.getByText('New clips show up here as the cameras see them.')).toBeInTheDocument();
});

it('shows twenty clips at a time, newest first, and more on request', async () => {
  items = Array.from({ length: 25 }, (_, i) => clip(`c${i}`, at(8, i), 'drinking'));
  renderPage(<ActivitiesTab />);
  await kindsListed();
  expect(await screen.findByText('25 clips to check ›')).toBeInTheDocument();
  expect(screen.getAllByText('Is this “Drinking”?')).toHaveLength(20);
  expect(screen.getByRole('listitem', { name: `Drinking at ${hm(at(8, 24))}` })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Show 5 more of 5' }));
  expect(screen.getAllByText('Is this “Drinking”?')).toHaveLength(25);
});

const asked = () => screen.getAllByText(/^Is this /).map(q => q.textContent);
/** The roster lists the kinds once every day has answered. */
const kindsListed = () => screen.findByText(/^A kind is counted in Ask/);
const kindButton = (name: RegExp) =>
  within(screen.getByRole('complementary', { name: 'Activity kinds' })).getByRole('button', { name });

it('shows only the clips of the kind picked in the roster, and all of them again from All activities', async () => {
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Squats”?');
  await userEvent.click(kindButton(/^Drinking\b.*\b1 clip to check/));
  expect(asked()).toEqual(['Is this “Drinking”?']);
  expect(screen.getByRole('heading', { name: 'Drinking: clips to check' })).toBeInTheDocument();
  expect(kindButton(/^Drinking/)).toHaveAttribute('aria-pressed', 'true');
  expect(kindButton(/^All activities/)).toHaveAttribute('aria-pressed', 'false');
  await userEvent.click(kindButton(/^All activities\b.*\b2 clips to check/));
  expect(asked()).toEqual(['Is this “Squats”?', 'Is this “Drinking”?']);
  expect(screen.getByRole('heading', { name: 'Clips to check' })).toBeInTheDocument();
});

it("shows a counted kind's clips as a 1 in 10 spot check among all activities, and all of them when it's picked", async () => {
  // walked_past is counted in Ask (the server's promotion row); of walk-0 to walk-19, walk-7 and walk-14 are sampled.
  items = [
    clip('early', at(7, 22), 'drinking'),
    ...Array.from({ length: 20 }, (_, i) => clip(`walk-${i}`, at(8, i), 'walked_past')),
  ];
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Drinking”?');
  const shownIds = () => asked();
  expect(shownIds().filter(q => /walked past/i.test(q))).toHaveLength(2);
  expect(shownIds()).toContain('Is this “Drinking”?');
  expect(
    screen.getByText('Walked past is counted in Ask already, so 1 in 10 of its clips are here to check.'),
  ).toBeInTheDocument();
  expect(kindButton(/^All activities\b.*\b3 clips to check/)).toBeInTheDocument();
  // The banner counts what the list holds, not every unanswered clip.
  expect(screen.getByText('3 clips to check ›')).toBeInTheDocument();
  await userEvent.click(kindButton(/^Walked past\b.*\b20 clips to check/));
  expect(shownIds()).toHaveLength(20);
  expect(screen.queryByText(/is counted in Ask already/)).toBeNull();
});

it('remembers the picked kind for the next visit', async () => {
  const first = renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Squats”?');
  await userEvent.click(kindButton(/^Squats/));
  first.unmount();
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Squats”?');
  expect(asked()).toEqual(['Is this “Squats”?']);
});

it('shows every kind when the remembered one is no longer listed', async () => {
  localStorage.setItem(FILTER_KEY, JSON.stringify('waved'));
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Squats”?');
  expect(asked()).toEqual(['Is this “Squats”?', 'Is this “Drinking”?']);
  expect(kindButton(/^All activities/)).toHaveAttribute('aria-pressed', 'true');
});

it('says when the picked kind has nothing left, and offers every kind', async () => {
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Squats”?');
  await userEvent.click(kindButton(/^Walked past/));
  expect(screen.queryByText(/^Is this /)).not.toBeInTheDocument();
  expect(screen.getByText('No walked past clips left to check')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Show all activities' }));
  expect(asked()).toHaveLength(2);
});

it('pages through the picked kind only', async () => {
  items = [
    ...Array.from({ length: 25 }, (_, i) => clip(`c${i}`, at(8, i), 'drinking')),
    ...Array.from({ length: 3 }, (_, i) => clip(`s${i}`, at(9, i), 'squats')),
  ];
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('28 clips to check ›');
  await userEvent.click(kindButton(/^Squats\b.*\b3 clips/));
  expect(asked()).toHaveLength(3);
  expect(screen.queryByRole('button', { name: /^Show \d+ more/ })).not.toBeInTheDocument();
  await userEvent.click(kindButton(/^Drinking\b.*\b25 clips/));
  expect(asked()).toHaveLength(20);
  await userEvent.click(screen.getByRole('button', { name: 'Show 5 more of 5' }));
  expect(asked()).toHaveLength(25);
});

/** A phone's width: PHONE_WIDTH matches, so the page lays out for a phone. */
function asPhone() {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    query =>
      ({
        matches: query.includes('max-width'),
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList,
  );
}
afterEach(() => vi.restoreAllMocks());

const chip = (name: RegExp) =>
  within(screen.getByRole('list', { name: 'Show clips of' })).getByRole('button', { name });

it('on a phone, puts the clips first with the kinds as chips over them, which filter them', async () => {
  asPhone();
  renderPage(<ActivitiesTab />);
  await screen.findByText('Is this “Squats”?');
  expect(screen.queryByRole('complementary', { name: 'Activity kinds' })).toBeNull();
  expect(chip(/^All\b.*\b2 clips to check/)).toHaveAttribute('aria-pressed', 'true');
  await userEvent.click(chip(/^Drinking\b.*\b1 clip to check/));
  expect(asked()).toEqual(['Is this “Drinking”?']);
  expect(chip(/^Drinking/)).toHaveAttribute('aria-pressed', 'true');
  await userEvent.click(chip(/^All/));
  expect(asked()).toEqual(['Is this “Squats”?', 'Is this “Drinking”?']);
});

it('names every counted kind the list thins, in a sentence', () => {
  expect(thinnedText(['walked past', 'read'])).toBe(
    'Walked past and Read are counted in Ask already, so 1 in 10 of their clips are here to check.',
  );
  expect(thinnedText(['a', 'b', 'c'])).toMatch(/^A, B, and C are counted/);
});

const roster = () => within(screen.getByRole('complementary', { name: 'Activity kinds' }));

it("lists the owner's activities by their own names, one with no clips yet and nothing to look for saying how to teach it", async () => {
  defined = [...defined, activity('zorbing', 'Zorbing')];
  renderPage(<ActivitiesTab />);
  await kindsListed();
  expect(await roster().findByText('Zorbing')).toBeInTheDocument();
  expect(roster().getByText('Nothing matches its name yet: call a clip “Zorbing” to teach it.')).toBeInTheDocument();
});

it('adds an activity from its name and a description, and says what it looks for', async () => {
  items = [];
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await userEvent.click(screen.getByRole('button', { name: 'Add an activity' }));
  const dialog = await screen.findByRole('dialog', { name: 'Add an activity' });
  await userEvent.type(within(dialog).getByRole('textbox', { name: /^Name/ }), ' Squats ');
  await userEvent.type(within(dialog).getByRole('textbox', { name: /What it looks like/ }), 'knees bent');
  await userEvent.click(within(dialog).getByRole('button', { name: 'Add' }));
  await waitFor(() =>
    expect(posts).toEqual([{ url: '/activity/kinds', body: { name: 'Squats', description: 'knees bent' } }]),
  );
  expect(await screen.findByText('Added Squats. Looks for squat down and squat.')).toBeInTheDocument();
  expect(await roster().findByText('Squats')).toBeInTheDocument();
  expect(screen.queryByRole('dialog', { name: 'Add an activity' })).toBeNull();
});

it('deletes an activity after saying its answers are kept, and Undo brings it back', async () => {
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Drinking”?');
  await userEvent.click(roster().getByRole('button', { name: 'Delete Drinking' }));
  const dialog = await screen.findByRole('dialog', { name: 'Delete “Drinking”?' });
  expect(within(dialog).getByText(/Its 6 answers are kept, so Undo/)).toBeInTheDocument();
  await userEvent.click(within(dialog).getByRole('button', { name: 'Delete' }));
  expect(await screen.findByText('Deleted Drinking')).toBeInTheDocument();
  await waitFor(() => expect(screen.queryByText('Is this “Drinking”?')).not.toBeInTheDocument());
  expect(roster().queryByRole('button', { name: 'Delete Drinking' })).toBeNull();
  await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
  expect(await roster().findByRole('button', { name: 'Delete Drinking' })).toBeInTheDocument();
  expect(posts).toEqual([
    { url: '/activity/kinds/delete', body: { kind: 'drinking' } },
    { url: '/activity/kinds/restore', body: { kind: 'drinking' } },
  ]);
});

it('keeps the activity when the delete is cancelled', async () => {
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await userEvent.click(await roster().findByRole('button', { name: 'Delete Drinking' }));
  const dialog = await screen.findByRole('dialog', { name: 'Delete “Drinking”?' });
  await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
  expect(posts).toEqual([]);
  expect(roster().getByRole('button', { name: 'Delete Drinking' })).toBeInTheDocument();
});

it('offers the owner activities as names for a clip', async () => {
  renderPage(<ActivitiesTab />);
  await kindsListed();
  await screen.findByText('Is this “Squats”?');
  await userEvent.click(within(card(/Squats at/)).getByRole('button', { name: 'Call it something else' }));
  await userEvent.type(within(card(/Squats at/)).getByRole('combobox', { name: 'Call it' }), 'dri');
  await userEvent.click(await screen.findByRole('option', { name: 'Drinking' }));
  await waitFor(() =>
    expect(posts).toEqual([{ url: '/activity/name', body: { activity: 'late', start: at(18, 40), name: 'Drinking' } }]),
  );
});
