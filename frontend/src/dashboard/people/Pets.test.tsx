import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PREFS_KEY } from '../../shared/prefs';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { navigate } from '../router';
import { Pets } from './Pets';
import classes from './people.module.css';
import type { Seen, Sighting, SightingsView } from './pets/api';

const MIA = '019c0000-0000-7000-8000-000000000001';
const REX = '019c0000-0000-7000-8000-000000000002';
const NEW_ID = '019c0000-0000-7000-8000-0000000000ff';
const SPECIES = ['dog', 'cat', 'bird']; // as /pets offers them: household, most likely first
const T = 1_800_000_000;

const sighting = (track: string, extra: Partial<Sighting> = {}): Sighting => ({
  track_id: track,
  species: 'cat',
  cam: 'cam0',
  start: T,
  end: T + 30,
  samples: 3,
  label: null,
  ...extra,
});
const seen = (track: string, end = T + 60): Seen => ({
  track_id: track,
  cam: 'cam1',
  start: end - 10,
  end,
  how: 'labeled',
  video_status: 'ready',
  video_thumb: `/saved/pet-${track}.jpg`,
});

type Pet = { id: string; name: string; species: string; sex?: string | null };
let pets: Pet[];
let view: SightingsView;
let writes: { url: string; body: unknown }[];

/** /pets and /pets/sightings as the server answers: a label takes its sightings off the list, as it does. */
function serve() {
  server.use(
    http.get('/pets', () => HttpResponse.json({ pets, species: SPECIES })),
    http.get('/pets/sightings', () => HttpResponse.json(view)),
    http.post('/pets/sightings/label', async ({ request }) => {
      const body = (await request.json()) as { track_ids: string[]; label: string | null };
      writes.push({ url: '/pets/sightings/label', body });
      if (body.label)
        view = {
          ...view,
          groups: view.groups
            .map(g => ({ ...g, sightings: g.sightings.filter(s => !body.track_ids.includes(s.track_id)) }))
            .filter(g => g.sightings.length),
        };
      return HttpResponse.json({ ok: true, labeled: body.track_ids.length });
    }),
    http.post('/pets/sightings/not', async ({ request }) => {
      const body = (await request.json()) as { track_ids: string[] };
      writes.push({ url: '/pets/sightings/not', body });
      return HttpResponse.json({ ok: true, marked: body.track_ids.length });
    }),
    http.post('/pets/sightings/block', async ({ request }) => {
      const body = (await request.json()) as { track_ids: string[]; block?: boolean };
      writes.push({ url: '/pets/sightings/block', body });
      return HttpResponse.json({ ok: true, blocked: body.track_ids.length });
    }),
    http.post('/pets/save', async ({ request }) => {
      const body = (await request.json()) as Pet & { id: string | null };
      writes.push({ url: '/pets/save', body });
      const id = body.id ?? NEW_ID;
      pets = [...pets.filter(p => p.id !== id), { ...body, id }];
      return HttpResponse.json({ ok: true, pets });
    }),
    http.post('/pets/remove', async ({ request }) => {
      const body = (await request.json()) as { id: string };
      writes.push({ url: '/pets/remove', body });
      pets = pets.filter(p => p.id !== body.id);
      return HttpResponse.json({ ok: true, pets });
    }),
  );
}

beforeEach(() => {
  history.replaceState(null, '', '/pets');
  writes = [];
  pets = [
    { id: MIA, name: 'Mia', species: 'cat', sex: 'female' },
    { id: REX, name: 'Rex', species: 'dog', sex: null },
  ];
  view = {
    label_days: 14,
    seen: { [MIA]: [seen('m1')], [REX]: [] },
    groups: [
      {
        id: 'cat:a',
        species: 'cat',
        sightings: [
          sighting('a', { recognized: { label: MIA, similarity: 0.9 } }),
          sighting('b', { recognized: { label: MIA, similarity: 0.9 } }),
          sighting('c'),
        ],
      },
    ],
    recognition: [{ species: 'cat', labeled: 8, right: 8, wrong: 0, unknown: 0, on: true, reason: null }],
    names: { [MIA]: 'Mia', [REX]: 'Rex' },
  };
});

const inbox = () => screen.findByRole('list', { name: 'Sightings to name' });
const labels = () => writes.filter(w => w.url === '/pets/sightings/label').map(w => w.body);
const blocks = () => writes.filter(w => w.url === '/pets/sightings/block').map(w => w.body);
type User = ReturnType<typeof userEvent.setup>;
/** One of a card's two buttons, by its exact words: the card is selected first, which opens one with several
 *  pictures (a folded card has no buttons). */
async function answer(user: User, row: HTMLElement, name: string) {
  await user.click(row);
  await user.click(within(row).getByRole('button', { name }));
}
/** The buttons a card shows, by their words (not its pictures, which are buttons too). */
const answers = (row: HTMLElement) =>
  within(row)
    .queryAllByRole('button')
    .filter(b => !b.hasAttribute('data-tile'))
    .map(b => b.textContent?.replace(/[A-Z]$/, ''));
/** The default group with nobody guessed for it: three cat sightings. */
const unguessed = () => {
  view.groups = [{ id: 'cat:a', species: 'cat', sightings: [sighting('a'), sighting('b'), sighting('c')] }];
};

it('polls pending pet videos while photo status stays independent and stops on ready or unavailable', async () => {
  view.seen = {
    [MIA]: [
      { ...seen('waiting', T + 120), video_status: 'pending', video_thumb: null },
      { ...seen('gone'), photo_ready: false, video_status: 'unavailable', video_thumb: null },
    ],
  };
  view.photos_pending = 0;
  view.videos_pending = 1;
  view.seen_unavailable = { [MIA]: 1 };
  let reads = 0;
  serve();
  server.use(
    http.get('/pets/sightings', () => {
      reads++;
      return HttpResponse.json(view);
    }),
  );
  history.replaceState(null, '', '/pets#pet=Mia');
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
  try {
    renderPage(<Pets />);
    const card = await screen.findByRole('article', { name: 'Card of Mia' });
    const videos = within(card).getByRole('region', { name: 'Recent video' });
    const photos = within(card).getByRole('region', { name: 'Recent sightings' });
    expect(await within(videos).findByText('Processing 1 video in the background…')).toBeVisible();
    expect(within(videos).getByText("1 video can't be shown.")).toBeVisible();
    expect(videos.querySelector('img')).toBeNull();
    expect(within(videos).queryByText('No recorded video of Mia yet.')).toBeNull();
    expect(photos.querySelector('img')).toHaveAttribute('src', '/pets/sightings/photo?track=waiting');
    expect(within(photos).getByText("1 detection can't be shown.")).toBeVisible();
    expect(within(photos).queryByText(/Processing/)).toBeNull();
    view.seen = {
      [MIA]: [
        { ...seen('waiting', T + 120), photo_ready: false, video_thumb: '/saved/full-scene.jpg' },
        { ...seen('gone'), photo_ready: false, video_status: 'unavailable', video_thumb: null },
      ],
    };
    view.videos_pending = 0;
    view.seen_unavailable = { [MIA]: 2 };
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(await within(videos).findByRole('button', { name: /^Play video:/ })).toBeVisible();
    expect(videos.querySelector('img')).toHaveAttribute('src', '/saved/full-scene.jpg');
    expect(within(videos).queryByText(/Processing/)).toBeNull();
    expect(within(videos).getByText("1 video can't be shown.")).toBeVisible();
    expect(photos.querySelector('img')).toBeNull();
    expect(within(photos).getByText("2 detections can't be shown.")).toBeVisible();
    const done = reads;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(reads).toBe(done);
  } finally {
    vi.useRealTimers();
  }
});

it('keeps pending pet crops out of image elements and polls until their pictures are saved', async () => {
  const group = view.groups[0];
  if (!group) throw new Error('fixture group missing');
  group.sightings[0] = { ...sighting('a'), photo_ready: false };
  group.sightings[1] = { ...sighting('b'), photo_ready: false };
  group.photos_pending = 1;
  group.photos_unavailable = 1;
  view.photos_pending = 1;
  view.photos_unavailable = 1;
  serve();
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
  try {
    renderPage(<Pets />);
    const row = await screen.findByRole('listitem', { name: /^3 cat sightings/ });
    expect(within(row).getByText('Processing 1 detection in the background…')).toBeVisible();
    expect(row.querySelector('img[src="/pets/sightings/photo?track=a"]')).toBeNull();
    expect(row.querySelectorAll('.who-shot img')).toHaveLength(1);
    expect(within(row).getByText("1 detection can't be shown.")).toBeVisible();
    group.sightings[0] = { ...sighting('a'), photo_ready: true };
    group.photos_pending = 0;
    view.photos_pending = 0;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await waitFor(() => expect(screen.queryByText(/Processing .* detection/)).toBeNull());
    expect(row.querySelector('img[src="/pets/sightings/photo?track=a"]')).not.toBeNull();
    expect(row.querySelector('img[src="/pets/sightings/photo?track=b"]')).toBeNull();
    expect(within(row).getByText("1 detection can't be shown.")).toBeVisible();
    expect(row).toHaveAccessibleName(/^3 cat sightings/);
  } finally {
    vi.useRealTimers();
  }
});

it('uses the newest ready pet photo while keeping later pending sightings in video and last-seen metadata', async () => {
  view.seen = { [MIA]: [{ ...seen('waiting', T + 120), photo_ready: false }, seen('ready')], [REX]: [] };
  view.photos_pending = 1;
  view.seen_pending = { [MIA]: 1 };
  serve();
  const user = userEvent.setup();
  renderPage(<Pets />);
  const roster = await screen.findByRole('list', { name: 'Named pets' });
  await user.click(within(roster).getByRole('button', { name: /^Mia/ }));
  const card = await screen.findByRole('article', { name: 'Card of Mia' });
  expect(card.querySelector('img[src="/pets/sightings/photo?track=waiting"]')).toBeNull();
  expect(card.querySelector('img[src="/pets/sightings/photo?track=ready"]')).not.toBeNull();
  expect(within(card).getByText('Processing 1 detection in the background…')).toBeVisible();
  expect(within(card).getByRole('region', { name: 'Recent video' }).querySelectorAll('li')).toHaveLength(2);
});

it('does not call unavailable pet photos processing when the queue reports no pending work', async () => {
  view.seen = {
    [MIA]: [{ ...seen('gone'), photo_ready: false, video_status: 'unavailable', video_thumb: null }],
    [REX]: [],
  };
  view.seen_pending = { [MIA]: 0 };
  view.seen_unavailable = { [MIA]: 1 };
  view.photos_pending = 0;
  view.photos_unavailable = 1;
  serve();
  const user = userEvent.setup();
  renderPage(<Pets />);
  const roster = await screen.findByRole('list', { name: 'Named pets' });
  await user.click(within(roster).getByRole('button', { name: /^Mia/ }));
  const card = await screen.findByRole('article', { name: 'Card of Mia' });
  expect(card.querySelector('img[src="/pets/sightings/photo?track=gone"]')).toBeNull();
  expect(within(card).queryByText(/Processing .* detection/)).toBeNull();
  expect(within(card).getByText("1 detection can't be shown.")).toBeVisible();
  const videos = within(card).getByRole('region', { name: 'Recent video' });
  expect(videos.querySelectorAll('li')).toHaveLength(0);
  expect(within(videos).getByText("1 video can't be shown.")).toBeVisible();
  expect(within(videos).queryByText('No recorded video of Mia yet.')).toBeNull();
});

it('counts pet video statuses only among the newest four and never requests unprepared thumbs', async () => {
  view.seen = {
    [MIA]: [
      seen('ready'),
      { ...seen('pending'), video_status: 'pending', video_thumb: '/unprepared.jpg' },
      { ...seen('unavailable'), video_status: 'unavailable', video_thumb: null },
      { ...seen('missing'), video_thumb: null },
      { ...seen('older'), video_status: 'pending', video_thumb: '/older.jpg' },
    ],
  };
  serve();
  history.replaceState(null, '', '/pets#pet=Mia');
  renderPage(<Pets />);
  const card = await screen.findByRole('article', { name: 'Card of Mia' });
  const videos = within(card).getByRole('region', { name: 'Recent video' });
  expect(videos.querySelectorAll('img')).toHaveLength(1);
  expect(videos.querySelector('img')).toHaveAttribute('src', '/saved/pet-ready.jpg');
  expect(within(videos).getByText('Processing 1 video in the background…')).toBeVisible();
  expect(within(videos).getByText("1 video can't be shown.")).toBeVisible();
});

describe("Pets in People's form", () => {
  it('lists the named pets with where they were last seen, and opens a card for each', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const list = await screen.findByRole('list', { name: 'Named pets' });
    await waitFor(() =>
      expect(within(list).getByRole('button', { name: /^Mia/ })).toHaveTextContent(/Cat · female, last seen .*cam1/),
    );
    expect(within(list).getByRole('button', { name: /^Rex/ })).toHaveTextContent('Dog, not seen yet');
    await user.type(screen.getByRole('textbox', { name: 'Search pets' }), 're');
    expect(within(list).queryByRole('button', { name: /^Mia/ })).toBeNull();
    await user.clear(screen.getByRole('textbox', { name: 'Search pets' }));
    await user.click(within(list).getByRole('button', { name: /^Mia/ }));
    const card = await screen.findByRole('article', { name: 'Card of Mia' });
    expect(within(card).getByRole('heading', { name: 'Mia' })).toBeVisible();
    expect(within(card).getByRole('region', { name: 'Recent sightings' })).toBeVisible();
    expect(within(card).getByText(/Recognized by appearance/)).toBeVisible();
    expect(within(list).getByRole('button', { name: /^Mia/ })).toHaveAttribute('aria-current', 'true');
  });

  it('opens a pet card on one click anywhere on the row', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const list = await screen.findByRole('list', { name: 'Named pets' });
    const mia = within(list).getByRole('button', { name: /^Mia/ });
    await user.click(within(mia).getByText(/^Cat/));
    expect(await screen.findByRole('article', { name: 'Card of Mia' })).toBeVisible();
    expect(mia).toHaveAttribute('aria-current', 'true');
  });

  it('opens the focused pet card with I', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const list = await screen.findByRole('list', { name: 'Named pets' });
    const mia = within(list).getByRole('button', { name: /^Mia/ });
    mia.focus();
    await user.keyboard('i');
    expect(await screen.findByRole('article', { name: 'Card of Mia' })).toBeVisible();
  });

  it("shows a pet's recent video as People's card does, from the same cached thumbnails", async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    await user.click(await screen.findByRole('button', { name: /^Mia/ }));
    const card = await screen.findByRole('article', { name: 'Card of Mia' });
    const video = within(card).getByRole('region', { name: 'Recent video' });
    const play = within(video).getByRole('button', { name: /^Play video: Seen by cam1/ });
    const thumb = play.querySelector('img');
    expect(thumb?.getAttribute('src')).toBe('/saved/pet-m1.jpg');
  });

  it('edits and removes a pet from its card, and adds one with New pet', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    await user.click(await screen.findByRole('button', { name: /^Mia/ }));
    const card = await screen.findByRole('article', { name: 'Card of Mia' });
    await user.click(within(card).getByRole('button', { name: 'Edit' }));
    const dialog = await screen.findByRole('dialog', { name: 'Edit Mia' });
    await user.clear(within(dialog).getByRole('textbox', { name: 'Pet name' }));
    await user.type(within(dialog).getByRole('textbox', { name: 'Pet name' }), 'Miso');
    await user.click(within(dialog).getByRole('button', { name: 'Save pet' }));
    expect(await screen.findByRole('article', { name: 'Card of Miso' })).toBeVisible();
    expect(writes[0]).toEqual({ url: '/pets/save', body: { id: MIA, name: 'Miso', species: 'cat', sex: 'female' } });
    await user.click(screen.getByRole('button', { name: 'More for Miso' }));
    await user.click(await screen.findByRole('menuitem', { name: 'Remove' }));
    // Asked first, with Cancel focused.
    const ask = await screen.findByRole('dialog', { name: /^Remove / });
    expect(within(ask).getByRole('button', { name: 'Cancel' })).toHaveFocus();
    await user.click(within(ask).getByRole('button', { name: 'Remove' }));
    expect(await screen.findByText("Removed Miso's name. Recorded observations stay.")).toBeVisible();
    expect(writes[1]).toEqual({ url: '/pets/remove', body: { id: MIA } });
    await user.click(screen.getByRole('button', { name: 'New pet' }));
    const add = await screen.findByRole('dialog', { name: 'New pet' });
    await user.type(within(add).getByRole('textbox', { name: 'Pet name' }), 'Nova');
    await user.selectOptions(within(add).getByRole('combobox', { name: 'Species' }), 'dog');
    await user.click(within(add).getByRole('button', { name: 'Add Nova' }));
    expect(await screen.findByRole('article', { name: 'Card of Nova' })).toBeVisible();
    expect(writes[2]).toEqual({ url: '/pets/save', body: { id: null, name: 'Nova', species: 'dog', sex: null } });
  });

  it('says what went wrong when the pets cannot load, and retries', async () => {
    let failed = true;
    server.use(
      http.get('/pets', () =>
        failed
          ? HttpResponse.json({ error: 'Broker offline' }, { status: 503 })
          : HttpResponse.json({ pets: [], species: SPECIES }),
      ),
      http.get('/pets/sightings', () => HttpResponse.json({ ...view, groups: [] })),
    );
    const user = userEvent.setup();
    renderPage(<Pets />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Broker offline');
    expect(screen.queryByText('No named pets yet.')).not.toBeInTheDocument();
    failed = false;
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await screen.findByText('No named pets yet');
  });
});

describe('Sightings to name', () => {
  it('names a group with its guess, takes it off at once, and Undo puts it back', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings: Probably Mia/ });
    expect(screen.getByRole('button', { name: /1 animal needs a name/ })).toBeVisible();
    await answer(user, row, 'Mia');
    expect(screen.queryByRole('listitem', { name: /cat sightings/ })).toBeNull();
    expect(await screen.findByText('Labeled 3 sightings as Mia.')).toBeVisible();
    expect(labels()).toEqual([{ track_ids: ['a', 'b', 'c'], label: MIA }]);
    view = {
      ...view,
      groups: [{ id: 'cat:a', species: 'cat', sightings: [sighting('a'), sighting('b'), sighting('c')] }],
    };
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    expect(await screen.findByRole('listitem', { name: /^3 cat sightings/ })).toBeVisible();
    expect(labels()[1]).toEqual({ track_ids: ['a', 'b', 'c'], label: null });
  });

  it('gives a guessed card exactly two buttons, the name and Not the name, once it is selected', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings: Probably Mia/ });
    // Three pictures: folded, with no buttons and no name picker, until it is selected.
    expect(answers(row)).toEqual([]);
    expect(within(row).queryByRole('textbox', { name: /Which pet is this\?/ })).toBeNull();
    await user.click(row);
    expect(answers(row)).toEqual(['Mia', 'Not Mia']);
    expect(within(row).queryByRole('textbox', { name: /Which pet is this\?/ })).toBeNull();
    expect(within(row).getByRole('button', { name: 'Mia' })).toHaveAttribute('aria-keyshortcuts', 'Y');
    expect(within(row).getByRole('button', { name: 'Not Mia' })).toHaveAttribute('aria-keyshortcuts', 'N');
    await user.click(within(row).getByRole('button', { name: 'Mia' }));
    expect(screen.queryByRole('listitem', { name: /cat sightings/ })).toBeNull();
    expect(await screen.findByText('Labeled 3 sightings as Mia.')).toBeVisible();
    expect(labels()).toEqual([{ track_ids: ['a', 'b', 'c'], label: MIA }]);
  });

  it('gives a card with no guess the name picker, Ignore, and Block', async () => {
    view.groups = [{ id: 'cat:z', species: 'cat', sightings: [sighting('z')] }];
    serve();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^1 cat sighting: No guess yet/ });
    expect(answers(row)).toEqual(['Ignore', 'Block']); // one picture: nothing to open, so they show at once
    expect(within(row).getByRole('textbox', { name: /Which pet is this\?/ })).toHaveAttribute(
      'placeholder',
      "Who's this?",
    );
    expect(within(row).getByRole('button', { name: 'Ignore' })).toHaveAttribute('aria-keyshortcuts', 'I');
    expect(within(row).getByRole('button', { name: 'Block' })).toHaveAttribute('aria-keyshortcuts', 'B');
  });

  it('offers the name for a clear match while recognition is still off, and shows its match score', async () => {
    view.recognition = [{ species: 'cat', labeled: 8, right: 2, wrong: 0, unknown: 6, on: false, reason: null }];
    view.groups = [
      {
        id: 'cat:z',
        species: 'cat',
        sightings: [sighting('z', { suggestion: { label: MIA, similarity: 0.86, clear: true } })],
      },
    ];
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^1 cat sighting: Probably Mia/ });
    expect(within(row).getByRole('progressbar', { name: 'Match score 86%' })).toBeInTheDocument();
    await user.click(within(row).getByRole('button', { name: 'Mia' }));
    expect(labels()).toEqual([{ track_ids: ['z'], label: MIA }]);
  });

  it('scores a lone unclear guess by how alike it looks, not as 100% of one sighting', async () => {
    view.groups = [
      {
        id: 'cat:z',
        species: 'cat',
        sightings: [sighting('z', { suggestion: { label: MIA, similarity: 0.31, clear: false } })],
      },
    ];
    serve();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^1 cat sighting: Maybe Mia/ });
    expect(within(row).getByRole('progressbar', { name: 'Match score 31%' })).toBeInTheDocument();
  });

  it.each([
    ['an unclear suggestion', { suggestion: { label: MIA, similarity: 0.7, clear: false } }],
    ['a suggestion from before clear was sent', { suggestion: { label: MIA, similarity: 0.7 } }],
  ])('offers the same two buttons for a Maybe: a group with %s', async (_, extra) => {
    view.groups = [{ id: 'cat:z', species: 'cat', sightings: [sighting('z', extra)] }];
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^1 cat sighting: Maybe Mia/ });
    expect(answers(row)).toEqual(['Mia', 'Not Mia']);
    await user.click(within(row).getByRole('button', { name: 'Mia' }));
    expect(await screen.findByText('Labeled 1 sighting as Mia.')).toBeVisible();
    expect(labels()).toEqual([{ track_ids: ['z'], label: MIA }]);
    expect(screen.getByRole('button', { name: 'Undo' })).toBeVisible(); // the same toast as a Probably's
  });

  describe('sort and picture size', () => {
    // As the server sends them, newest first: no guess, a weak Maybe, a strong Maybe, then a Probably.
    beforeEach(() => {
      const maybe = (track: string, similarity: number) =>
        sighting(track, { suggestion: { label: MIA, similarity, clear: false } });
      view.groups = [
        { id: 'cat:n', species: 'cat', sightings: [sighting('n')] },
        { id: 'cat:w', species: 'cat', sightings: [maybe('w', 0.4)] },
        { id: 'cat:s', species: 'cat', sightings: [maybe('s', 0.7)] },
        { id: 'cat:p', species: 'cat', sightings: [sighting('p', { recognized: { label: MIA, similarity: 0.9 } })] },
      ];
    });
    const order = async () =>
      within(await inbox())
        .getAllByRole('listitem')
        .map(li => li.getAttribute('aria-label')?.replace(/^1 cat sighting: /, ''));
    const flip = (user: ReturnType<typeof userEvent.setup>, now: 'highest' | 'lowest') =>
      user.click(screen.getByRole('button', { name: `Sort sightings by certainty, ${now} first` }));

    it('puts the most sure first by default, flips with the Certainty toggle, and remembers it', async () => {
      serve();
      const user = userEvent.setup();
      const page = renderPage(<Pets />);
      await within(await inbox()).findAllByRole('listitem');
      expect(await order()).toEqual(['Probably Mia', 'Maybe Mia', 'Maybe Mia', 'No guess yet']);
      expect(
        within(await inbox())
          .getAllByRole('progressbar')
          .map(b => b.getAttribute('aria-label')),
      ).toEqual(['Match score 90%', 'Match score 70%', 'Match score 40%']);
      await flip(user, 'highest');
      expect(await order()).toEqual(['No guess yet', 'Maybe Mia', 'Maybe Mia', 'Probably Mia']);
      expect(within(await inbox()).getAllByRole('progressbar')[0]).toHaveAccessibleName('Match score 40%');
      page.unmount();
      renderPage(<Pets />);
      await within(await inbox()).findAllByRole('listitem');
      expect(await order()).toEqual(['No guess yet', 'Maybe Mia', 'Maybe Mia', 'Probably Mia']);
      await flip(user, 'lowest');
      expect(await order()).toEqual(['Probably Mia', 'Maybe Mia', 'Maybe Mia', 'No guess yet']);
    });

    it('starts at highest first in a browser that remembered the old Newest order', async () => {
      localStorage.setItem(PREFS_KEY, JSON.stringify({ petSightingsSort: 'newest' }));
      serve();
      renderPage(<Pets />);
      await within(await inbox()).findAllByRole('listitem');
      expect(screen.getByRole('button', { name: 'Sort sightings by certainty, highest first' })).toBeInTheDocument();
      expect(await order()).toEqual(['Probably Mia', 'Maybe Mia', 'Maybe Mia', 'No guess yet']);
    });

    it('on a phone, folds the sort, size, and view into one View menu, and remembers a pick', async () => {
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
      serve();
      const user = userEvent.setup();
      const page = renderPage(<Pets />);
      await within(await inbox()).findAllByRole('listitem');
      expect(screen.queryByRole('button', { name: /^Sort sightings by certainty/ })).toBeNull();
      expect(screen.queryByRole('radiogroup', { name: 'Picture size' })).toBeNull();
      await user.click(screen.getByRole('button', { name: 'View' }));
      expect(await screen.findByRole('menuitem', { name: 'Highest certainty first, chosen' })).toBeInTheDocument();
      await user.click(screen.getByRole('menuitem', { name: 'Lowest certainty first' }));
      expect(await order()).toEqual(['No guess yet', 'Maybe Mia', 'Maybe Mia', 'Probably Mia']);
      expect(screen.getByRole('menuitem', { name: 'Lowest certainty first, chosen' })).toBeInTheDocument();
      page.unmount();
      vi.restoreAllMocks();
      renderPage(<Pets />);
      await within(await inbox()).findAllByRole('listitem');
      expect(screen.getByRole('button', { name: 'Sort sightings by certainty, lowest first' })).toBeInTheDocument();
    });

    it('loads the thumbnail at the small size, the full crop otherwise, in the list and By pet alike', async () => {
      serve();
      const user = userEvent.setup();
      renderPage(<Pets />);
      const list = await inbox();
      const src = (root: Element) => root.querySelector('img')?.getAttribute('src');
      await waitFor(() => expect(src(list)).toBe('/pets/sightings/photo?track=p'));
      const size = (label: string) =>
        user.click(within(screen.getByRole('radiogroup', { name: 'Picture size' })).getByLabelText(label));
      await size('S');
      await waitFor(() => expect(src(list)).toBe('/pets/sightings/photo?track=p&size=80'));
      expect(list.closest('section')).toHaveAttribute('data-photo-size', 'S');
      await user.click(screen.getByRole('radio', { name: 'By pet' }));
      const tiles = await screen.findByRole('group', { name: 'Sightings guessed as Mia' });
      await waitFor(() => expect(src(tiles)).toMatch(/&size=80$/));
      await size('M');
      await waitFor(() => expect(src(tiles)).not.toMatch(/size=/));
    });
  });

  describe('the shared naming row', () => {
    const nots = () => writes.filter(w => w.url === '/pets/sightings/not').map(w => w.body);
    beforeEach(() => {
      server.use(
        http.post('/pets/sightings/not', async ({ request }) => {
          const body = (await request.json()) as { track_ids: string[] };
          writes.push({ url: '/pets/sightings/not', body });
          return HttpResponse.json({ ok: true, marked: body.track_ids.length });
        }),
      );
    });
    const two = () => {
      view.groups = [
        { id: 'cat:a', species: 'cat', sightings: [sighting('a', { recognized: { label: MIA, similarity: 0.9 } })] },
        { id: 'cat:b', species: 'cat', sightings: [sighting('b', { recognized: { label: MIA, similarity: 0.8 } })] },
      ];
    };

    it('marks the whole group as not the guess with Not, keeps it waiting, and Undo takes it back', async () => {
      serve();
      const user = userEvent.setup();
      renderPage(<Pets />);
      const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings: Probably Mia/ });
      await answer(user, row, 'Not Mia');
      expect(await screen.findByText('Marked 3 sightings as not Mia.')).toBeVisible();
      expect(screen.getByRole('listitem', { name: /^3 cat sightings/ })).toBeVisible(); // back in the unmatched pool
      expect(nots()).toEqual([{ track_ids: ['a', 'b', 'c'], pet: MIA, not: true }]);
      expect(labels()).toEqual([]);
      await user.click(screen.getByRole('button', { name: 'Undo' }));
      await waitFor(() => expect(nots().at(-1)).toEqual({ track_ids: ['a', 'b', 'c'], pet: MIA, not: false }));
    });

    it('says yes with Y on the focused row, then moves focus to the next row without scrolling', async () => {
      two();
      serve();
      const user = userEvent.setup();
      renderPage(<Pets />);
      const rows = await within(await inbox()).findAllByRole('listitem');
      rows[0]?.focus();
      await user.keyboard('y');
      await waitFor(() => expect(labels()).toEqual([{ track_ids: ['a'], label: MIA }]));
      expect(rows[1]).toHaveFocus();
    });

    it('selects a row on a click off its answers, without answering, and Y then acts on it', async () => {
      two();
      serve();
      const user = userEvent.setup();
      renderPage(<Pets />);
      const [first, second] = (await within(await inbox()).findAllByRole('listitem')) as [HTMLElement, HTMLElement];
      await user.click(within(second).getByText(/^Probably Mia/));
      expect(second).toHaveAttribute('aria-current', 'true');
      expect(first).not.toHaveAttribute('aria-current');
      expect(second).toHaveFocus();
      expect(labels()).toEqual([]);
      await user.click(within(first).getByText(/^Probably Mia/));
      expect(first).toHaveAttribute('aria-current', 'true');
      expect(second).not.toHaveAttribute('aria-current'); // one selected row in the list
      await user.keyboard('y');
      await waitFor(() => expect(labels()).toEqual([{ track_ids: ['a'], label: MIA }]));
    });

    it('ignores with I and blocks with B on the focused row, but a name box takes a letter as typing', async () => {
      view.groups = [
        { id: 'cat:a', species: 'cat', sightings: [sighting('a')] },
        { id: 'cat:b', species: 'cat', sightings: [sighting('b', { start: T - 7200, end: T - 7100 })] },
      ];
      serve();
      const user = userEvent.setup();
      renderPage(<Pets />);
      const rows = await within(await inbox()).findAllByRole('listitem');
      const [first, second] = rows as [HTMLElement, HTMLElement];
      await user.click(picker(second));
      await user.keyboard('ib');
      expect(writes).toEqual([]);
      first.focus();
      await user.keyboard('i');
      await waitFor(() => expect(labels()).toEqual([{ track_ids: ['a'], label: 'ignored' }]));
      const left = await within(await inbox()).findByRole('listitem');
      left.focus();
      await user.keyboard('b');
      await waitFor(() => expect(blocks()).toEqual([{ track_ids: ['b'], block: true }]));
    });

    it("shows the server's guess for the group over one worked out from its sightings", async () => {
      view.groups = [
        {
          id: 'cat:z',
          species: 'cat',
          sightings: [sighting('z', { suggestion: { label: MIA, similarity: 0.4, clear: false } })],
          guess: { id: MIA, name: 'Mia', score: 91, band: 'probable', close: null },
        },
      ];
      serve();
      renderPage(<Pets />);
      const row = await within(await inbox()).findByRole('listitem', { name: /^1 cat sighting: Probably Mia/ });
      expect(within(row).getByRole('progressbar', { name: 'Match score 91%' })).toBeInTheDocument();
      expect(answers(row)).toEqual(['Mia', 'Not Mia']);
    });
  });

  it('keeps the page to the list: no introduction or per-species recognition status', async () => {
    serve();
    renderPage(<Pets />);
    await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    expect(screen.queryByText(/Animals the cameras saw/)).toBeNull();
    expect(screen.queryByText(/recognized by appearance|not recognized yet|Checked so far/)).toBeNull();
  });

  it('offers no name button for a group with no guess', async () => {
    view.groups = [{ id: 'cat:z', species: 'cat', sightings: [sighting('z')] }];
    serve();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^1 cat sighting: No guess yet/ });
    expect(picker(row)).toBeVisible();
    expect(within(row).queryByRole('button', { name: /Mia|Rex|^Yes/ })).toBeNull();
  });

  it('opens every picture of a selected card, all selected, and names only the ones still selected', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    expect(within(row).queryByRole('region')).toBeNull();
    await user.click(row);
    const grid = within(row).getByRole('region', { name: /^Every picture of/ });
    const picks = within(grid).getAllByRole('button', { name: /^cat on cam0/ });
    expect(picks.map(p => p.getAttribute('aria-pressed'))).toEqual(['true', 'true', 'true']);
    expect(within(grid).getByRole('status')).toHaveTextContent('3 of 3 pictures selected. Click one to leave it out.');
    // No separate button leaves pictures out: deselecting one is the whole gesture.
    expect(within(row).queryByRole('button', { name: /Not this pet|Ignore these|Show all|Hide/ })).toBeNull();
    await user.click(picks[2] as HTMLElement);
    expect(picks[2]).toHaveAttribute('aria-pressed', 'false');
    expect(within(grid).getByRole('status')).toHaveTextContent('2 of 3 pictures selected. The rest stay unnamed.');
    await user.click(within(row).getByRole('button', { name: 'Mia' }));
    await waitFor(() => expect(labels()).toEqual([{ track_ids: ['a', 'b'], label: MIA }]));
    // The one left out stays unnamed, waiting on its own.
    expect(await screen.findByRole('listitem', { name: /^1 cat sighting: No guess yet/ })).toBeVisible();
  });

  it('says only the selected sightings are not the guess', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await user.click(row);
    await user.click(within(row).getAllByRole('button', { name: /^cat on cam0/ })[0] as HTMLElement);
    await user.click(within(row).getByRole('button', { name: 'Not Mia' }));
    await waitFor(() =>
      expect(writes).toEqual([{ url: '/pets/sightings/not', body: { track_ids: ['b', 'c'], pet: MIA, not: true } }]),
    );
  });

  it('turns both buttons off when every picture is deselected', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await user.click(row);
    for (const tile of within(row).getAllByRole('button', { name: /^cat on cam0/ })) await user.click(tile);
    expect(within(row).getByRole('button', { name: 'Mia' })).toBeDisabled();
    expect(within(row).getByRole('button', { name: 'Not Mia' })).toBeDisabled();
    await user.keyboard('y');
    expect(writes).toEqual([]);
  });

  const picker = (row: HTMLElement) => within(row).getByRole('textbox', { name: /Which pet is this\?/ });

  it('names a group as a pet of another species when the detector got it wrong', async () => {
    unguessed();
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await user.click(row);
    await user.click(picker(row));
    const species = await within(await screen.findByRole('listbox')).findAllByRole('option');
    expect(species.map(o => o.textContent)).toEqual(['Cat (detected)', 'Dog', 'Bird']); // the detected one first
    await user.click(screen.getByRole('option', { name: 'Dog' }));
    expect(await screen.findByRole('option', { name: 'Rex' })).toBeVisible();
    expect(screen.queryByRole('option', { name: 'Mia' })).toBeNull(); // a dog's pets, not a cat's
    await user.click(screen.getByRole('option', { name: 'Rex' }));
    await waitFor(() => expect(labels()).toEqual([{ track_ids: ['a', 'b', 'c'], label: REX }]));
  });

  it('names a group as a new pet with Other…, its species filled in from the column', async () => {
    view.groups = [{ id: 'cat:z', species: 'cat', sightings: [sighting('z')] }];
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^1 cat sighting: No guess yet/ });
    expect(within(row).queryByRole('button', { name: 'Another animal' })).toBeNull();
    await user.click(picker(row));
    await user.click(await screen.findByRole('option', { name: 'Dog' }));
    await user.click(await screen.findByRole('option', { name: 'Other…' }));
    const dialog = await screen.findByRole('dialog', { name: 'New pet' });
    expect(within(dialog).getByRole('combobox', { name: 'Species' })).toHaveValue('dog');
    await user.type(within(dialog).getByRole('textbox', { name: 'Pet name' }), 'Pip');
    await user.click(within(dialog).getByRole('button', { name: 'Add Pip' }));
    await waitFor(() => expect(labels()).toEqual([{ track_ids: ['z'], label: NEW_ID }]));
    expect(writes[0]).toEqual({ url: '/pets/save', body: { id: null, name: 'Pip', species: 'dog', sex: null } });
  });

  it('ignores a group: it leaves the list, with Undo', async () => {
    unguessed();
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await answer(user, row, 'Ignore');
    expect(screen.queryByRole('listitem', { name: /cat sightings/ })).toBeNull();
    expect(await screen.findByText("Ignored 3 sightings. They won't be asked about again.")).toBeVisible();
    expect(labels()).toEqual([{ track_ids: ['a', 'b', 'c'], label: 'ignored' }]);
    view = {
      ...view,
      groups: [{ id: 'cat:a', species: 'cat', sightings: [sighting('a'), sighting('b'), sighting('c')] }],
    };
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    expect(await screen.findByRole('listitem', { name: /^3 cat sightings/ })).toBeVisible();
    expect(labels()[1]).toEqual({ track_ids: ['a', 'b', 'c'], label: null });
  });

  it('ignores only the selected pictures, and the one left out still waits', async () => {
    unguessed();
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await user.click(row);
    await user.click(within(row).getAllByRole('button', { name: /^cat on cam0/ })[2] as HTMLElement);
    await user.click(within(row).getByRole('button', { name: 'Ignore' }));
    expect(await screen.findByRole('listitem', { name: /^1 cat sighting/ })).toBeVisible();
    expect(labels()).toEqual([{ track_ids: ['a', 'b'], label: 'ignored' }]);
  });

  it('blocks a group: it leaves the list, and Undo unblocks it', async () => {
    unguessed();
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await answer(user, row, 'Block');
    expect(screen.queryByRole('listitem', { name: /cat sightings/ })).toBeNull();
    expect(await screen.findByText("Blocked 3 sightings. Ones very like them won't be listed.")).toBeVisible();
    expect(blocks()).toEqual([{ track_ids: ['a', 'b', 'c'], block: true }]);
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    await waitFor(() => expect(blocks().at(-1)).toEqual({ track_ids: ['a', 'b', 'c'], block: false }));
    expect(await screen.findByRole('listitem', { name: /^3 cat sightings/ })).toBeVisible();
  });

  describe('Show ignored and blocked', () => {
    let reads: number;
    beforeEach(() => {
      reads = 0;
      server.use(
        http.get('/pets/sightings/hidden', () => {
          reads++;
          return HttpResponse.json({
            ignored: [{ track_id: 'i1', species: 'cat', cam: 'cam0', start: T, end: T + 30, at: T + 60 }],
            ignored_count: 1,
            blocked: [{ track_id: 'b1', species: 'dog', cam: 'cam1', start: T, end: T + 30, at: T + 90 }],
            blocked_count: 1,
          });
        }),
      );
    });
    const toggle = () => screen.findByRole('switch', { name: 'Show ignored and blocked' });

    it('is off by default, and reads nothing until it is turned on', async () => {
      serve();
      renderPage(<Pets />);
      expect(await toggle()).not.toBeChecked();
      expect(screen.queryByRole('region', { name: 'Ignored' })).toBeNull();
      expect(reads).toBe(0);
    });

    it('lists the ignored and the blocked sightings, and un-ignores or unblocks each', async () => {
      serve();
      const user = userEvent.setup();
      renderPage(<Pets />);
      await user.click(await toggle());
      const ignored = await screen.findByRole('region', { name: 'Ignored' });
      const blocked = screen.getByRole('region', { name: 'Blocked' });
      expect(within(ignored).getAllByRole('listitem')).toHaveLength(1);
      expect(within(blocked).getByRole('listitem', { name: /^Dog on / })).toBeVisible();
      await user.click(within(ignored).getByRole('button', { name: /^Un-ignore Cat on / }));
      await waitFor(() => expect(labels()).toEqual([{ track_ids: ['i1'], label: null }]));
      await user.click(within(blocked).getByRole('button', { name: /^Unblock Dog on / }));
      await waitFor(() => expect(blocks()).toEqual([{ track_ids: ['b1'], block: false }]));
      expect(await screen.findByText('Unblocked. It waits for a name again.')).toBeVisible();
    });

    it('says why when one cannot be brought back', async () => {
      serve();
      server.use(
        http.post('/pets/sightings/label', () =>
          HttpResponse.json({ ok: false, error: '1 sightings are no longer recorded' }, { status: 404 }),
        ),
      );
      const user = userEvent.setup();
      renderPage(<Pets />);
      await user.click(await toggle());
      const ignored = await screen.findByRole('region', { name: 'Ignored' });
      await user.click(within(ignored).getByRole('button', { name: /^Un-ignore/ }));
      expect(await screen.findByRole('alert')).toHaveTextContent('no longer recorded');
    });
  });

  it('offers the household species for a sighting the detector called something else', async () => {
    view.groups = [{ id: 'animal:e', species: 'animal', sightings: [sighting('e', { species: 'animal' })] }];
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^1 animal sighting/ });
    await user.click(picker(row));
    const species = await within(await screen.findByRole('listbox')).findAllByRole('option');
    expect(species.map(o => o.textContent)).toEqual(['Dog', 'Cat', 'Bird']); // no zebras, nothing marked detected
  });

  it('puts a group back and says why when the label fails', async () => {
    serve();
    server.use(
      http.post('/pets/sightings/label', () =>
        HttpResponse.json({ ok: false, error: '3 sightings are no longer recorded' }, { status: 404 }),
      ),
    );
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await answer(user, row, 'Mia');
    expect(
      await screen.findByText(/Could not label those sightings: 3 sightings are no longer recorded/),
    ).toBeVisible();
    expect(screen.getByRole('listitem', { name: /^3 cat sightings/ })).toBeVisible();
  });
});

describe('By pet', () => {
  beforeEach(() => {
    view.groups.push(
      { id: 'cat:d', species: 'cat', sightings: [sighting('d', { suggestion: { label: MIA, similarity: 0.95 } })] },
      {
        id: 'dog:r',
        species: 'dog',
        sightings: [sighting('r', { species: 'dog', recognized: { label: REX, similarity: 0.8 } })],
      },
    );
  });
  const tiles = () => within(screen.getByRole('group', { name: /^Sightings guessed as/ })).getAllByRole('button');

  it("labels one pet's picked sightings in one call, marks the unpicked ones not that pet, and Undo takes back both", async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await user.click(screen.getByRole('radio', { name: 'By pet' }));
    expect(screen.getByRole('radio', { name: 'By pet' })).toBeChecked();
    expect(screen.getByRole('radio', { name: 'Mia (3)' })).toBeChecked(); // the pet with the most waiting
    expect(tiles().map(t => t.getAttribute('aria-label')?.match(/match \d+%/)?.[0])).toEqual([
      'match 95%',
      'match 90%',
      'match 90%',
    ]); // best match first; c has no guess and isn't here
    expect(tiles().every(t => t.getAttribute('aria-pressed') === 'true')).toBe(true);
    await user.click(tiles()[1] as HTMLElement);
    expect(tiles()[1]).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByText(/those 1 are saved as not Mia\./)).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Label 2 as Mia' }));
    expect(labels()).toEqual([{ track_ids: ['d', 'b'], label: MIA }]);
    expect(await screen.findByText('Labeled 2 sightings as Mia. Marked 1 as not Mia.')).toBeVisible();
    const nots = () => writes.filter(w => w.url === '/pets/sightings/not').map(w => w.body);
    expect(nots()).toEqual([{ track_ids: ['a'], pet: MIA, not: true }]); // a, left out, still waits, not as Mia
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    await waitFor(() => expect(labels()[1]).toEqual({ track_ids: ['d', 'b'], label: null }));
    await waitFor(() => expect(nots()[1]).toEqual({ track_ids: ['a'], pet: MIA, not: false }));
    await user.click(screen.getByRole('radio', { name: 'List' }));
    expect(await screen.findByRole('listitem', { name: /^2 cat sightings/ })).toBeVisible(); // back to the list: a and c
  });

  it('switches pets with the chips and remembers the last one', async () => {
    serve();
    const user = userEvent.setup();
    const { unmount } = renderPage(<Pets />);
    await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await user.click(screen.getByRole('radio', { name: 'By pet' }));
    await user.click(screen.getByRole('radio', { name: 'Rex (1)' }));
    expect(tiles()).toHaveLength(1);
    expect(screen.getByRole('button', { name: 'Label 1 as Rex' })).toBeEnabled();
    unmount();
    renderPage(<Pets />);
    await user.click(await screen.findByRole('radio', { name: 'By pet' }));
    expect(screen.getByRole('radio', { name: 'Rex (1)' })).toBeChecked();
  });

  it("offers at most the label route's 500 at a time, best first, and says so", async () => {
    view.groups = [
      {
        id: 'cat:many',
        species: 'cat',
        sightings: Array.from({ length: 501 }, (_, i) =>
          sighting(`t${i}`, { suggestion: { label: MIA, similarity: 0.9 - i / 10_000 } }),
        ),
      },
    ];
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    await user.click(await screen.findByRole('radio', { name: 'By pet' }));
    expect(screen.getByRole('radio', { name: 'Mia (501)' })).toBeChecked();
    expect(screen.getByText(/Showing the best 500 of 501/)).toBeVisible();
    expect(tiles()).toHaveLength(500);
    await user.click(screen.getByRole('button', { name: 'Label 500 as Mia' }));
    expect(labels()[0]).toMatchObject({ label: MIA });
    expect((labels()[0] as { track_ids: string[] }).track_ids).not.toContain('t500'); // the weakest waits
  });

  it('says the sightings failed to load instead of showing an empty pet', async () => {
    serve();
    server.use(http.get('/pets/sightings', () => HttpResponse.json({ error: 'broker down' }, { status: 500 })));
    const user = userEvent.setup();
    renderPage(<Pets />);
    await user.click(await screen.findByRole('radio', { name: 'By pet' }));
    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't load sightings");
    expect(screen.queryByText(/No waiting sightings are guessed/)).toBeNull();
    expect(screen.queryByRole('radiogroup', { name: 'Which pet' })).toBeNull();
  });

  it('moves between pictures with the arrows and toggles one with Space', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await user.click(screen.getByRole('radio', { name: 'By pet' }));
    (tiles()[0] as HTMLElement).focus();
    await user.keyboard('{ArrowRight}');
    expect(tiles()[1]).toHaveFocus();
    await user.keyboard(' ');
    expect(tiles()[1]).toHaveAttribute('aria-pressed', 'false');
    await user.keyboard('{ArrowLeft} ');
    expect(tiles()[0]).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', { name: 'Label 1 as Mia' })).toBeEnabled();
  });
});

describe('Pets from a link', () => {
  it('lists only the linked species, until Show all', async () => {
    serve();
    history.replaceState(null, '', '/pets#species=dog');
    const user = userEvent.setup();
    renderPage(<Pets />);
    expect(await screen.findByText('Showing dog only.')).toBeInTheDocument();
    const list = await screen.findByRole('list', { name: 'Named pets' });
    expect(within(list).getByRole('button', { name: /^Rex/ })).toBeVisible();
    expect(within(list).queryByRole('button', { name: /^Mia/ })).toBeNull();
    expect(screen.queryByRole('listitem', { name: /cat sightings/ })).toBeNull(); // a cat group isn't a dog's
    await user.click(screen.getByRole('button', { name: 'Show all' }));
    expect(within(list).getByRole('button', { name: /^Mia/ })).toBeVisible();
    expect(location.hash).toBe('');
  });

  it("opens the linked pet's card, whatever the case", async () => {
    serve();
    history.replaceState(null, '', '/pets#pet=mia');
    renderPage(<Pets />);
    expect(await screen.findByRole('article', { name: 'Card of Mia' })).toBeVisible();
    expect(screen.getByRole('button', { name: /^Mia/ })).toHaveAttribute('aria-current', 'true');
  });
});

describe('the waiting count', () => {
  beforeEach(() => {
    view.groups.push(
      { id: 'cat:d', species: 'cat', sightings: [sighting('d', { suggestion: { label: MIA, similarity: 0.95 } })] },
      { id: 'dog:r', species: 'dog', sightings: [sighting('r', { species: 'dog' })] },
    );
  });
  /** Label writes that wait until the test lets them through, so the count is checked before any reply. */
  function holdLabels() {
    let open = () => {};
    const gate = new Promise<void>(resolve => {
      open = resolve;
    });
    server.use(
      http.post('/pets/sightings/label', async ({ request }) => {
        const body = (await request.json()) as { track_ids: string[]; label: string | null };
        writes.push({ url: '/pets/sightings/label', body });
        await gate;
        return HttpResponse.json({ ok: true, labeled: body.track_ids.length });
      }),
    );
    return open;
  }

  it('drops as soon as the name is pressed, before the label is saved', async () => {
    serve();
    const open = holdLabels();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const row = await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    expect(screen.getByRole('button', { name: /3 animals need a name/ })).toBeVisible();
    await answer(user, row, 'Mia');
    await waitFor(() => expect(labels()).toHaveLength(1));
    expect(screen.getByRole('button', { name: /2 animals need a name/ })).toBeVisible();
    open();
    expect(await screen.findByText('Labeled 3 sightings as Mia.')).toBeVisible();
  });

  it('drops as soon as By pet labels a whole group, before the label is saved', async () => {
    serve();
    const open = holdLabels();
    const user = userEvent.setup();
    renderPage(<Pets />);
    await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await user.click(screen.getByRole('radio', { name: 'By pet' }));
    const tiles = within(screen.getByRole('group', { name: /^Sightings guessed as/ })).getAllByRole('button');
    await user.click(tiles[1] as HTMLElement); // leave b out: cat:a still waits (a, c), cat:d (d) is done
    await user.click(tiles[2] as HTMLElement);
    await user.click(screen.getByRole('button', { name: 'Label 1 as Mia' }));
    await waitFor(() => expect(labels()).toEqual([{ track_ids: ['d'], label: MIA }]));
    expect(screen.getByRole('button', { name: /2 animals need a name/ })).toBeVisible();
    open();
    expect(await screen.findByText('Labeled 1 sighting as Mia. Marked 2 as not Mia.')).toBeVisible();
  });

  it('marks every one not the pet when none are left checked', async () => {
    serve();
    const user = userEvent.setup();
    renderPage(<Pets />);
    await within(await inbox()).findByRole('listitem', { name: /^3 cat sightings/ });
    await user.click(screen.getByRole('radio', { name: 'By pet' }));
    const tiles = () => within(screen.getByRole('group', { name: /^Sightings guessed as/ })).getAllByRole('button');
    for (const tile of tiles()) await user.click(tile);
    await user.click(screen.getByRole('button', { name: 'Mark 3 as not Mia' }));
    expect(await screen.findByText('Marked 3 as not Mia.')).toBeVisible();
    expect(labels()).toEqual([]);
  });
});

it('shows each naming row its newest picture large while avatars stay small', async () => {
  serve();
  const user = userEvent.setup();
  renderPage(<Pets />);
  const naming = await inbox();
  await waitFor(() => expect(naming.querySelector('img')).toHaveAttribute('src', '/pets/sightings/photo?track=a'));
  const roster = await screen.findByRole('list', { name: 'Named pets' });
  expect(roster.querySelector('img')).toHaveAttribute('src', '/pets/sightings/photo?track=m1&size=80');
  await user.click(within(roster).getByRole('button', { name: /^Mia/ }));
  const card = await screen.findByRole('article', { name: 'Card of Mia' });
  expect(within(card).getByRole('img', { name: 'Mia' })).toHaveAttribute('src', '/pets/sightings/photo?track=m1');
});

it('keeps a ready main pet photo visible while its compact avatar and naming tile are pending', async () => {
  view.seen = { [MIA]: [{ ...seen('m1'), thumb_ready: false }], [REX]: [] };
  view.groups = [
    {
      id: 'cat:a',
      species: 'cat',
      sightings: [sighting('a', { photo_ready: true, thumb_ready: false }), sighting('b'), sighting('c')],
    },
  ];
  serve();
  const user = userEvent.setup();
  renderPage(<Pets />);
  await waitFor(() => expect(screen.getByRole('listitem', { name: /^3 cat sightings/ })).toBeInTheDocument());
  const naming = await inbox();
  expect(naming.querySelector('img[src*="track=a&"]')).toBeNull();
  expect(naming.querySelector('img[src="/pets/sightings/photo?track=a"]')).not.toBeNull();
  const roster = await screen.findByRole('list', { name: 'Named pets' });
  expect(roster.querySelector('img')).toBeNull();
  await user.click(within(roster).getByRole('button', { name: /^Mia/ }));
  const card = await screen.findByRole('article', { name: 'Card of Mia' });
  expect(within(card).getByRole('img', { name: 'Mia' })).toHaveAttribute('src', '/pets/sightings/photo?track=m1');
});

describe("a pet card's photo menu", () => {
  function serveCard() {
    serve();
    server.use(
      http.post('/pets/photo', async ({ request }) => {
        const body = (await request.json()) as { id: string; action: string; track: string | null };
        writes.push({ url: '/pets/photo', body });
        pets = pets.map(p =>
          p.id !== body.id
            ? p
            : body.action === 'main'
              ? { ...p, photo: body.track }
              : body.action === 'hide'
                ? { ...p, hidden: [body.track as string] }
                : { ...p, photo: null, hidden: [] },
        );
        return HttpResponse.json({ ok: true, pets });
      }),
      http.post('/pets/sightings/not', async ({ request }) => {
        writes.push({ url: '/pets/sightings/not', body: await request.json() });
        return HttpResponse.json({ ok: true, marked: 1 });
      }),
    );
  }

  async function openCard() {
    view.seen = { [MIA]: [seen('m2', T + 120), seen('m1')], [REX]: [] };
    serveCard();
    const user = userEvent.setup();
    renderPage(<Pets />);
    const roster = await screen.findByRole('list', { name: 'Named pets' });
    await user.click(within(roster).getByRole('button', { name: /^Mia/ }));
    const card = await screen.findByRole('article', { name: 'Card of Mia' });
    return { user, card };
  }

  const more = (card: HTMLElement, track: string) =>
    within(card)
      .getAllByRole('button', { name: /^More for the photo of Mia/ })
      .find(b => b.closest('li')?.querySelector(`img[src$="track=${track}"]`)) as HTMLElement;

  it('makes a sighting the main photo, and the card and its avatar follow', async () => {
    const { user, card } = await openCard();
    expect(within(card).getByRole('img', { name: 'Mia' })).toHaveAttribute('src', '/pets/sightings/photo?track=m2');
    await user.click(more(card, 'm1'));
    await user.click(await screen.findByRole('menuitem', { name: 'Use as main photo' }));
    await waitFor(() =>
      expect(within(card).getByRole('img', { name: 'Mia' })).toHaveAttribute('src', '/pets/sightings/photo?track=m1'),
    );
    expect(writes.at(-1)).toEqual({ url: '/pets/photo', body: { id: MIA, action: 'main', track: 'm1' } });
  });

  it('hides a sighting from the card', async () => {
    const { user, card } = await openCard();
    await user.click(more(card, 'm1'));
    await user.click(await screen.findByRole('menuitem', { name: 'Hide from the card' }));
    await waitFor(() => expect(within(card).queryByRole('img', { name: /^Mia on .*cam1/ })).not.toBeNull());
    await waitFor(() => expect(card.querySelectorAll('img[src$="track=m1"]')).toHaveLength(0));
  });

  it('says a sighting is not this pet', async () => {
    const { user, card } = await openCard();
    await user.click(more(card, 'm1'));
    await user.click(await screen.findByRole('menuitem', { name: 'Not Mia' }));
    await waitFor(() =>
      expect(writes.at(-1)).toEqual({ url: '/pets/sightings/not', body: { track_ids: ['m1'], pet: MIA, not: true } }),
    );
  });
});

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

describe('Pets on a phone', () => {
  afterEach(() => vi.restoreAllMocks());

  it('opens on the sightings waiting for a name, with the list one tap back', async () => {
    asPhone();
    serve();
    renderPage(<Pets />);
    expect(await inbox()).toBeInTheDocument();
    expect(document.querySelector(`.${classes.picked}`)).not.toBeNull();
    await userEvent.click(screen.getByRole('button', { name: 'Back to pets' }));
    expect(document.querySelector(`.${classes.picked}`)).toBeNull();
  });

  it("opens a pet's card on one tap", async () => {
    asPhone();
    serve();
    renderPage(<Pets />);
    await userEvent.click(await screen.findByRole('button', { name: 'Back to pets' }));
    const roster = await screen.findByRole('list', { name: 'Named pets' });
    await userEvent.click(within(roster).getByRole('button', { name: /^Mia/ }));
    expect(await screen.findByRole('article', { name: 'Card of Mia' })).toBeInTheDocument();
    expect(document.querySelector(`.${classes.picked}`)).not.toBeNull();
  });
});

describe('links in, as the command palette jumps (the review)', () => {
  it('opens the linked pet over the one picked before', async () => {
    serve();
    history.replaceState(null, '', '/pets');
    renderPage(<Pets />);
    const roster = await screen.findByRole('list', { name: 'Named pets' });
    await userEvent.click(within(roster).getByRole('button', { name: /^Mia/ }));
    await screen.findByRole('article', { name: 'Card of Mia' });
    act(() => navigate('/pets#pet=Rex'));
    expect(await screen.findByRole('article', { name: 'Card of Rex' })).toBeInTheDocument();
  });

  it('opens what waits for a name from #sightings, and tidies the address', async () => {
    serve();
    history.replaceState(null, '', '/pets');
    renderPage(<Pets />);
    const roster = await screen.findByRole('list', { name: 'Named pets' });
    await userEvent.click(within(roster).getByRole('button', { name: /^Mia/ }));
    await screen.findByRole('article', { name: 'Card of Mia' });
    act(() => navigate('/pets#sightings'));
    expect(await inbox()).toBeInTheDocument();
    expect(screen.queryByRole('article', { name: 'Card of Mia' })).not.toBeInTheDocument();
    expect(location.hash).toBe('');
  });
});
