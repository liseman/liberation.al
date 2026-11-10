import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { request, TIMEOUT_MS } from '../../shared/http';
import { newQueryClient, Providers } from '../../shared/Providers';
import { RETRY_MS } from '../../shared/SafeImage';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { SLOW_MS } from '../lib/loading';
import { PAGE_SLOW_MS } from '../lib/page';
import { getUnnamed } from '../people/api';
import { navigate } from '../router';
import {
  UNKNOWN_GROUPS_KEY,
  type UnknownGroup,
  type UnknownVoice,
  unknownGroupsQuery,
  unknownVoicesQuery,
} from './api';
import { ShotPicker } from './ShotPicker';
import { guessText, seenAt, shotSrc, togetherText, WhosThis } from './WhosThis';

const NOW = new Date(2026, 9, 28, 20, 0).getTime() / 1000;
const at = (h: number, m: number, day = 28) => new Date(2026, 9, day, h, m).getTime() / 1000;

const group = (over: Partial<UnknownGroup>): UnknownGroup => ({
  id: 1,
  label: 'Face 1',
  count: 42,
  last_seen: at(18, 40),
  shots: [{ id: over.id ?? 1, ts: at(18, 40), url: `/faces/shot/${over.id ?? 1}.jpg` }],
  guess: null,
  score: 0,
  band: 'none',
  reason: 'A clear face, unlike anyone named',
  ...over,
});

const KARA = group({
  id: 5,
  label: 'Face 5',
  shots: [1, 2, 3, 4, 5, 6].map(id => ({ id, ts: at(18, 40), url: `/faces/shot/${id}.jpg` })),
  guess: { name: 'Kara', score: 86, band: 'probable', close: null },
  score: 86,
  band: 'probable',
  reason: "A clear face, close to Kara's photos",
});
const LUCA = group({
  id: 6,
  label: 'Face 6',
  count: 7,
  guess: { name: 'Luca', score: 58, band: 'maybe', close: null },
  score: 58,
  band: 'maybe',
  reason: 'Side-on and dim',
});
const NOBODY = group({ id: 9, label: 'Face 9', count: 3, score: 20 });

let posts: { url: string; body: unknown }[];
/** The server's list; a tag or an ignore takes the face off it, as the real one does. */
function serve(list: UnknownGroup[], skipped: number[] = [], voices: UnknownVoice[] = []) {
  let groups = list;
  const drop = (id: number) => {
    groups = groups.filter(g => g.id !== id);
  };
  server.use(
    http.get('/faces/unknown/groups', () => HttpResponse.json({ groups, count: groups.length, skipped })),
    http.get('/voices/unknown/groups', () => HttpResponse.json({ groups: voices, count: voices.length, skipped: [] })),
    http.post('/people/claim', async ({ request }) => {
      const body = (await request.json()) as { id: number; name: string };
      posts.push({ url: '/people/claim', body });
      drop(body.id);
      return HttpResponse.json({ ok: true, name: body.name });
    }),
    http.post('/inbox/skip', async ({ request }) => {
      const body = (await request.json()) as { id: number };
      posts.push({ url: '/inbox/skip', body });
      drop(body.id);
      return HttpResponse.json({ ok: true, skipped: 1 });
    }),
    http.get('/speakers', () => HttpResponse.json({ speakers: [], merged: {} })),
    http.get('/people', () => HttpResponse.json({ people: [] })),
    http.get('/transcript', () => HttpResponse.json({ entries: [] })),
  );
}

beforeEach(() => {
  posts = [];
  history.replaceState(null, '', '/live');
});

const show = () => renderPage(<WhosThis last={{}} live={[]} />);
const row = (name: RegExp) => screen.findByRole('listitem', { name });
const nameBox = (card: HTMLElement) => within(card).getByRole('combobox', { name: "This person's name" });
async function choice(card: HTMLElement, name: string) {
  await userEvent.click(nameBox(card));
  return screen.findByRole('option', { name: new RegExp(`^${name} \\(guess,`) });
}
async function pick(card: HTMLElement, name: string) {
  await userEvent.click(await choice(card, name));
}
/** An answer from the row's icon actions: Not <name>, a voice's own answers, or Ignore. */
async function more(card: HTMLElement, name: string) {
  await userEvent.click(within(card).getByRole('button', { name: new RegExp(`^${name}`) }));
}

it('omits face groups without a ready picture instead of drawing blank naming rows', async () => {
  serve([
    KARA,
    group({ id: 77, label: 'Face 77', shots: [], shot_count: 0 }),
    group({ id: 78, label: 'Face 78', shots: [{ id: 78, ts: 0, url: '' }], shot_count: 1 }),
  ]);
  show();
  const kara = await row(/^Face 5/);
  expect(kara.querySelectorAll('img')).toHaveLength(1); // its newest picture
  expect(screen.queryByRole('listitem', { name: /^Face 77/ })).toBeNull();
  expect(screen.queryByRole('listitem', { name: /^Face 78/ })).toBeNull();
  expect(screen.queryByText(/no showable picture/)).toBeNull(); // a group nobody can see isn't counted either
  expect(document.querySelector('img[src="/faces/unknown/77.jpg"]')).toBeNull();
});

it('keeps the crop processing status without rendering a pending-only face group', async () => {
  serve([]);
  server.use(
    http.get('/faces/unknown/groups', () =>
      HttpResponse.json({
        groups: [group({ id: 77, label: 'Face 77', shots: [], shot_count: 0, shots_pending: 3 })],
        count: 1,
        shots_pending: 3,
      }),
    ),
  );
  show();
  expect(await screen.findByText('Processing 3 detections in the background…')).toBeVisible();
  expect(screen.queryByRole('listitem', { name: /^Face 77/ })).toBeNull();
  expect(screen.queryByText('Nobody is waiting for a name.')).toBeNull();
});

it('shows only saved face crops and polls a processing count without requesting a group fallback image', async () => {
  serve([]);
  let ready = false;
  let reads = 0;
  server.use(
    http.get('/faces/unknown/groups', () => {
      reads++;
      return HttpResponse.json({
        groups: [
          {
            ...KARA,
            shots: ready ? KARA.shots.slice(0, 1) : [],
            shot_count: ready ? 1 : 0,
            shots_pending: ready ? 0 : 124,
          },
        ],
        shots_pending: ready ? 0 : 124,
        shots_unavailable: 2556,
      });
    }),
  );
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
  try {
    show();
    expect(await screen.findByText('Processing 124 detections in the background…')).toBeVisible();
    expect(screen.getByText("2,556 detections can't be shown.")).toBeVisible();
    expect(screen.queryByRole('listitem', { name: /^Face 5/ })).toBeNull();
    ready = true;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await waitFor(() => expect(screen.queryByText(/Processing .* detections/)).toBeNull());
    const kara = await row(/^Face 5/);
    expect(kara.querySelector('.who-shot img')).toHaveAttribute('src', '/faces/shot/1.jpg');
    expect(within(kara).getByRole('button', { name: 'Show all (1)' })).toBeVisible();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(reads).toBe(2); // unavailable detections alone do not keep the processing poll running
  } finally {
    vi.useRealTimers();
  }
});

it.each(['Kara', 'Ignore'])(
  'hides immediately for %s, blocks duplicates, survives stale refresh, and restores on rejection',
  async action => {
    serve([KARA, NOBODY]);
    let release!: (value: Response) => void;
    let calls = 0;
    server.use(
      http.post(action === 'Ignore' ? '/inbox/skip' : '/people/claim', () => {
        calls++;
        return new Promise<Response>(resolve => {
          release = resolve;
        });
      }),
    );
    const { client } = show();
    const card = await row(/^Face 5/);
    if (action === 'Ignore') {
      // The I key, twice: the menu closes after one pick, so the key is how a double answer can happen.
      fireEvent.keyDown(card, { key: 'i' });
      fireEvent.keyDown(card, { key: 'i' });
    } else {
      const button = await choice(card, action);
      fireEvent.click(button);
      fireEvent.click(button);
    }
    expect(screen.queryByRole('listitem', { name: /^Face 5/ })).not.toBeInTheDocument();
    await waitFor(() => expect(calls).toBe(1));
    await act(async () => {
      await client.invalidateQueries({ queryKey: ['faces', 'unknown', 'groups'] });
    });
    expect(screen.queryByRole('listitem', { name: /^Face 5/ })).not.toBeInTheDocument();
    await act(async () => release(HttpResponse.json({ ok: false, error: 'Synthetic refusal' })));
    expect(within(await row(/^Face 5/)).getByRole('alert')).toHaveTextContent('Synthetic refusal');
    expect(await row(/^Face 9/)).toBeInTheDocument();
    if (action === 'Ignore') await more(await row(/^Face 5/), 'Ignore');
    else fireEvent.click(await choice(await row(/^Face 5/), action));
    await waitFor(() => expect(calls).toBe(2));
    await act(async () => release(HttpResponse.json({ ok: true })));
    await act(async () => {
      await client.invalidateQueries({ queryKey: ['faces', 'unknown', 'groups'] });
    });
    expect(screen.queryByRole('listitem', { name: /^Face 5/ })).not.toBeInTheDocument();
  },
);

it('words each confidence band', () => {
  expect(guessText(KARA)).toBe('Probably Kara 86%');
  expect(guessText(LUCA)).toBe('Maybe Luca 58%');
  expect(guessText(NOBODY)).toBe('No good guess');
});

it('says when a face was last seen', () => {
  expect(seenAt(at(18, 40), NOW)).toBe('today 18:40'); // 24-hour, as the rest of the app
  expect(seenAt(at(9, 5, 27), NOW)).toMatch(/^yesterday /);
  expect(seenAt(at(9, 5, 26), NOW)).toMatch(/^Mon 26 Oct /);
});

describe("Who's this?", () => {
  it("counts every face and voice that waits, not just the ones listed, as People's line does", async () => {
    // The server lists the first face groups but counts them all: 3 listed of 30, plus no voices.
    serve([]);
    server.use(
      http.get('/faces/unknown/groups', () =>
        HttpResponse.json({ groups: [KARA, LUCA, NOBODY], count: 30, skipped: [] }),
      ),
    );
    renderPage(<WhosThis last={{}} live={[]} wide />);
    expect(await screen.findByText('30 waiting · 3 shown')).toBeInTheDocument();
  });

  it('counts the listed rows when the server lists them all', async () => {
    serve([KARA, LUCA, NOBODY]);
    renderPage(<WhosThis last={{}} live={[]} wide />);
    expect(await screen.findByText('3 waiting')).toBeInTheDocument();
  });

  it('shows each face group with its pictures, how often it was seen, the guess, a bar, and the reason', async () => {
    serve([KARA, LUCA, NOBODY]);
    show();
    const kara = await row(/^Face 5/);
    expect(kara).toHaveAccessibleName('Face 5: Probably Kara 86%');
    // One picture, the newest, with Show all for the rest: decorative, the row's name says who it is
    expect(kara.querySelectorAll('.who-shot img')).toHaveLength(1);
    expect(within(kara).getByText(/^Seen 42 times, /)).toBeInTheDocument();
    expect(within(kara).getByText('Probably Kara 86%')).toHaveAttribute('data-band', 'probable');
    expect(within(kara).getByRole('progressbar', { name: 'Match score 86%' })).toBeInTheDocument();
    expect(within(kara).getByText("A clear face, close to Kara's photos")).toBeInTheDocument();
    const luca = await row(/^Face 6/);
    expect(within(luca).getByText('Maybe Luca 58%')).toHaveAttribute('data-band', 'maybe');
    expect(within(luca).getByText('Side-on and dim')).toBeInTheDocument();
    const nobody = await row(/^Face 9/);
    expect(within(nobody).getByText('No good guess')).toHaveAttribute('data-band', 'none');
    expect(within(nobody).queryByRole('button', { name: /^Yes/ })).not.toBeInTheDocument();
    expect(screen.getAllByRole('listitem').map(li => li.getAttribute('aria-label'))).toEqual([
      'Face 5: Probably Kara 86%',
      'Face 6: Maybe Luca 58%',
      'Face 9: No good guess',
    ]); // as the server sorts them: best guess first
  });

  it('shows every name selector immediately without stealing focus, and Enter accepts the highlighted guess', async () => {
    serve([KARA, LUCA, NOBODY]);
    show();
    await row(/^Face 9/);
    expect(screen.getAllByRole('combobox', { name: "This person's name" })).toHaveLength(3);
    expect(screen.queryByRole('button', { name: /^(Name them|Someone else|Tag$)/ })).toBeNull();
    // A guess gets a one-click Yes, a Probably's a Default button and a Maybe's a Subtle one; no guess, none.
    expect(screen.getAllByRole('button', { name: /^Yes,/ }).map(b => [b.textContent, b.dataset.variant])).toEqual([
      ['Yes, Kara', 'default'],
      ['Yes, Luca', 'subtle'],
    ]);
    expect(document.activeElement?.tagName).not.toBe('INPUT');
    await userEvent.click(nameBox(await row(/^Face 5/)));
    expect(await screen.findByRole('option', { name: 'Kara (guess, 86%)' })).toHaveAttribute('data-combobox-selected');
    await userEvent.keyboard('{Enter}');
    await waitFor(() => expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'face', id: 5, name: 'Kara' } }]));
  });

  it('tags the face as the selected guess', async () => {
    serve([KARA]);
    show();
    await pick(await row(/^Face 5/), 'Kara');
    await waitFor(() => expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'face', id: 5, name: 'Kara' } }]));
    expect(
      await screen.findByText('Face 5 tagged as Kara. Liberational will know them from now on.'),
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole('listitem')).not.toBeInTheDocument()); // the list is read again
    expect(screen.getByRole('status', { name: 'Decisions' })).toHaveTextContent('Face 5 tagged as Kara.'); // and the confirmation stays
  });

  it('keeps the page where it was after a pick, with focus on the next row and nothing scrolled', async () => {
    const scrolled: Element[] = [];
    const saved = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = function (this: Element) {
      scrolled.push(this);
    };
    const focus = vi.spyOn(HTMLElement.prototype, 'focus');
    onTestFinished(() => {
      Element.prototype.scrollIntoView = saved;
      focus.mockRestore();
    });
    serve([KARA, LUCA, NOBODY]);
    show();
    await row(/^Face 9/);
    focus.mockClear();
    await pick(await row(/^Face 6/), 'Luca');
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(await row(/^Face 9/)).toHaveFocus(); // the next one down, not the heading at the top
    // focusing without preventScroll scrolls the element into view: the heading's did, to the top of the page
    const scrolling = focus.mock.contexts.filter((_, i) => !(focus.mock.calls[i]?.[0] as FocusOptions)?.preventScroll);
    expect(scrolling).not.toContain(screen.getByRole('heading', { name: "Who's this?" }));
    expect(scrolling).not.toContain(await row(/^Face 9/));
    expect(scrolled.filter(e => e.getAttribute('role') !== 'option')).toEqual([]); // the dropdown scrolls its own options
  });

  it('moves focus to the row above when the last one is picked', async () => {
    serve([KARA, LUCA]);
    show();
    await pick(await row(/^Face 6/), 'Luca');
    expect(await row(/^Face 5/)).toHaveFocus();
  });

  it('tags the face as someone else, typed in', async () => {
    serve([LUCA]);
    show();
    const luca = await row(/^Face 6/);
    await userEvent.type(within(luca).getByRole('combobox', { name: "This person's name" }), 'Sam{Enter}');
    await waitFor(() => expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'face', id: 6, name: 'Sam' } }]));
    expect(await screen.findByText(/^Face 6 tagged as Sam\./)).toBeInTheDocument();
  });

  it('adds a new name immediately when Add is clicked', async () => {
    serve([LUCA]);
    show();
    const luca = await row(/^Face 6/);
    await userEvent.type(within(luca).getByRole('combobox', { name: "This person's name" }), 'Sam');
    await userEvent.click(await screen.findByRole('option', { name: 'Add “Sam”' }));
    await waitFor(() => expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'face', id: 6, name: 'Sam' } }]));
  });

  it('keeps a typed name when Tab moves focus off the box', async () => {
    serve([LUCA]);
    show();
    const luca = await row(/^Face 6/);
    const input = within(luca).getByRole('combobox', { name: "This person's name" });
    await userEvent.type(input, 'Sam');
    await userEvent.tab();
    expect(input).toHaveValue('Sam');
    expect(posts).toEqual([]);
  });

  it('picks a known name from the list, and tags it once', async () => {
    serve([LUCA]);
    server.use(http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }, { name: 'Sam' }] })));
    show();
    const face = await row(/^Face 6/);
    await userEvent.type(within(face).getByRole('combobox', { name: "This person's name" }), 'Da');
    // what was typed comes first, as "Add", so Enter never swaps it for a longer name
    const options = await screen.findAllByRole('option');
    expect(options.map(o => o.textContent)).toEqual(['Add “Da”', 'Dana']);
    await userEvent.click(screen.getByRole('option', { name: 'Dana' }));
    await waitFor(() => expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'face', id: 6, name: 'Dana' } }]));
    expect(await screen.findByText(/^Face 6 tagged as Dana\./)).toBeInTheDocument();
  });

  it('takes a known name typed in any case, with Enter', async () => {
    serve([LUCA]);
    server.use(http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }] })));
    show();
    const face = await row(/^Face 6/);
    const input = within(face).getByRole('combobox', { name: "This person's name" });
    await userEvent.type(input, 'dana');
    expect((await screen.findAllByRole('option')).map(o => o.textContent)).toEqual(['Dana']);
    await userEvent.type(input, '{Enter}');
    await waitFor(() => expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'face', id: 6, name: 'Dana' } }]));
  });

  it('keeps a chosen name editable when saving is rejected', async () => {
    serve([LUCA]);
    server.use(
      http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }] })),
      http.post('/people/claim', () => HttpResponse.json({ ok: false, error: 'Please try again' })),
    );
    show();
    const face = await row(/^Face 6/);
    await userEvent.type(within(face).getByRole('combobox', { name: "This person's name" }), 'Dana{Enter}');
    expect(await screen.findByRole('alert')).toHaveTextContent('Please try again');
    expect(screen.getByRole('combobox', { name: "This person's name" })).toHaveValue('Dana');
    expect(screen.getByRole('combobox', { name: "This person's name" })).toBeEnabled();
  });

  it('clears the draft with Escape without tagging', async () => {
    serve([LUCA]);
    show();
    const luca = await row(/^Face 6/);
    await userEvent.type(nameBox(luca), 'Sam{Escape}');
    expect(nameBox(luca)).toHaveValue('');
    expect(posts).toEqual([]);
  });

  it('puts a face aside with Ignore', async () => {
    serve([NOBODY]);
    show();
    await more(await row(/^Face 9/), 'Ignore');
    await waitFor(() => expect(posts).toEqual([{ url: '/inbox/skip', body: { kind: 'face', id: 9 } }]));
    expect(await screen.findByText("Face 9 ignored. It won't be asked about again.")).toBeInTheDocument();
  });

  it('selects a row on a click off its answers, and I then ignores the selected one', async () => {
    serve([KARA, NOBODY]);
    show();
    const nobody = await row(/^Face 9/);
    await userEvent.click(within(nobody).getByText('No good guess'));
    expect(nobody).toHaveAttribute('aria-current', 'true');
    expect(nobody).toHaveFocus();
    expect(posts).toEqual([]);
    await userEvent.keyboard('i');
    await waitFor(() => expect(posts).toEqual([{ url: '/inbox/skip', body: { kind: 'face', id: 9 } }]));
  });

  it('marks a face in view while showing only its saved pictures', async () => {
    serve([KARA]);
    renderPage(
      <WhosThis
        last={{ cam0: { cam: 'cam0', ts: NOW, faces: [{ name: 'Face 5', bbox: [10, 10, 80, 80] }] } }}
        live={['cam0']}
      />,
    );
    const kara = await row(/^Face 5/);
    expect(within(kara).getByText(/in view now/)).toBeInTheDocument();
    expect(kara.querySelectorAll('.who-shot img')).toHaveLength(1); // its newest saved picture
    expect(kara.querySelector('img[src^="/faces/unknown/"]')).toBeNull();
    expect(kara.querySelector('canvas')).toBeNull();
  });

  it('does not invent naming rows for live faces without a saved picture', async () => {
    serve([KARA], [13]);
    const face = (name: string) => ({
      name,
      bbox: [10, 10, 80, 80] as [number, number, number, number],
      bearing_deg: 90,
    });
    renderPage(
      <WhosThis last={{ cam0: { cam: 'cam0', ts: NOW, faces: [face('Face 12'), face('Face 13')] } }} live={['cam0']} />,
    );
    const rows = await screen.findAllByRole('listitem');
    expect(rows.map(r => r.getAttribute('aria-label'))).toEqual(['Face 5: Probably Kara 86%']);
    expect(document.querySelector('img[src^="/faces/unknown/"]')).toBeNull();
  });

  it('omits a group whose last ready URL fails instead of leaving a blank or placeholder row', async () => {
    serve([{ ...LUCA, shots: [{ id: 1, ts: 0, url: '/faces/shot/missing.jpg' }] }]);
    const { client } = show();
    const luca = await row(/^Face 6/);
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      for (const wait of RETRY_MS) {
        fireEvent.error(luca.querySelector('.who-shot img') as HTMLImageElement);
        expect(luca.querySelector('[data-safe-image][data-visible="true"]')).not.toBeNull(); // still loading
        act(() => void vi.advanceTimersByTime(wait));
      }
      fireEvent.error(luca.querySelector('.who-shot img') as HTMLImageElement);
      expect(screen.queryByRole('listitem', { name: /^Face 6/ })).toBeNull();
    } finally {
      vi.useRealTimers();
    }
    vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 1000);
    await act(async () => {
      await client.refetchQueries({ queryKey: UNKNOWN_GROUPS_KEY });
    });
    const restored = await row(/^Face 6/);
    fireEvent.load(restored.querySelector('.who-shot img') as HTMLImageElement);
  });

  it('scrolls to itself when a link to #whos-this is followed, on opening and while Live stays open', async () => {
    const scrolled: string[] = [];
    const saved = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = function (this: Element) {
      if (this.id === 'whos-this') scrolled.push(this.id);
    };
    onTestFinished(() => {
      Element.prototype.scrollIntoView = saved;
    });
    serve([KARA]);
    history.replaceState(null, '', '/#whos-this');
    show();
    await waitFor(() => expect(scrolled).toEqual(['whos-this']));
    navigate('/');
    navigate('/#whos-this'); // People's link, with Live already open
    expect(scrolled).toEqual(['whos-this', 'whos-this']);
  });

  it('says when no face is waiting, and when the list could not load', async () => {
    serve([]);
    const page = show();
    expect(await screen.findByText('Nobody is waiting for a name.')).toBeInTheDocument();
    page.unmount();
    server.use(http.get('/faces/unknown/groups', () => HttpResponse.json({ error: 'busy' }, { status: 500 })));
    show();
    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't load the faces waiting for a name");
    expect(screen.getByText('busy')).toBeInTheDocument(); // the server's reason, under Show details
  });
});

// Voices

const VOICE: UnknownVoice = {
  id: 12,
  label: 'Speaker 12',
  count: 14,
  last_seen: at(19, 5),
  clips: [
    {
      ts: at(19, 5),
      line: `kitchen@${at(19, 5).toFixed(3)}`, // its key: another microphone may have a line in the same millisecond
      text: 'Shall we put the kettle on?',
      dur: 2.4,
      url: `/clip?line=kitchen%40${at(19, 5).toFixed(3)}`,
    },
  ],
  guess: { name: 'Dana', score: 90, band: 'probable', close: null },
  score: 90,
  band: 'probable',
  reason: "A clear voice, close to Dana's",
};

function serveVoices(voices: UnknownVoice[], faces: UnknownGroup[] = []) {
  serve(faces, [], voices);
  server.use(
    http.post('/speakers/refine', async ({ request }) => {
      posts.push({ url: '/speakers/refine', body: await request.json() });
      return HttpResponse.json({ ok: true, entry: { speaker: 'multiple speakers' } });
    }),
    http.post('/speakers/refine/undo', async ({ request }) => {
      const body = (await request.json()) as { lines: string[] };
      posts.push({ url: '/speakers/refine/undo', body });
      return HttpResponse.json({ undone: body.lines.map(line => ({ ts_start: 0, line })) });
    }),
  );
}

it.each([
  ['/faces/shot/7.jpg?size=80', true, '/faces/shot/7.jpg?size=80'],
  ['/faces/shot/7.jpg?size=80', false, '/faces/shot/7.jpg'],
  ['/faces/shot/7.jpg', true, '/faces/shot/7.jpg?size=80'],
  ['/faces/shot/7.jpg', false, '/faces/shot/7.jpg'],
])('sizes shot %s (small: %s) as %s, never doubling the size', (url, small, want) => {
  expect(shotSrc(url, small)).toBe(want);
});

it('shows a server shot URL that already asks for the thumbnail as the saved crop at the default size', async () => {
  serve([{ ...KARA, shots: [{ id: 1, ts: at(18, 40), url: '/faces/shot/1.jpg?size=80' }] }]);
  renderPage(<WhosThis last={{}} live={[]} wide />);
  const kara = await row(/^Face 5/);
  expect(kara.querySelector('.who-shot img')).toHaveAttribute('src', '/faces/shot/1.jpg');
});

describe('on the shared naming inbox', () => {
  it('names a face with Yes, as picking the guess does', async () => {
    serve([KARA, LUCA]);
    show();
    await userEvent.click(within(await row(/^Face 5/)).getByRole('button', { name: 'Yes, Kara' }));
    await waitFor(() => expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'face', id: 5, name: 'Kara' } }]));
  });

  it('says a face is not the guess with Not, keeps it waiting, and Undo takes it back', async () => {
    serve([KARA]);
    server.use(
      http.post('/faces/not', async ({ request }) => {
        posts.push({ url: '/faces/not', body: await request.json() });
        return HttpResponse.json({ ok: true, marked: 1, undo: [] });
      }),
    );
    show();
    const kara = await row(/^Face 5/);
    await more(kara, 'Not Kara');
    expect(await screen.findByText('Face 5 is not Kara.')).toBeVisible();
    expect(posts).toEqual([{ url: '/faces/not', body: { name: 'Kara', groups: [5], shots: [], not: true } }]);
    expect(await row(/^Face 5/)).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    await waitFor(() =>
      expect(posts.at(-1)).toEqual({ url: '/faces/not', body: { name: 'Kara', groups: [5], shots: [], not: false } }),
    );
  });

  it("shows a face's newest picture, its thumbnail at the small size and the saved crop otherwise", async () => {
    serve([KARA]);
    renderPage(<WhosThis last={{}} live={[]} wide />);
    const kara = await row(/^Face 5/);
    expect(kara.querySelectorAll('.who-shot img')).toHaveLength(1);
    expect(kara.querySelector('.who-shot img')).toHaveAttribute('src', KARA.shots[0]?.url); // M, the default
    await userEvent.click(within(screen.getByRole('radiogroup', { name: 'Picture size' })).getByLabelText('S'));
    await waitFor(() =>
      expect(kara.querySelector('.who-shot img')).toHaveAttribute('src', `${KARA.shots[0]?.url}?size=80`),
    );
  });

  it('names the checked faces in By person, marks the unchecked ones not that person, and Undo takes back both', async () => {
    const SECOND = {
      ...KARA,
      id: 7,
      label: 'Face 7',
      score: 70,
      shots: [{ id: 70, ts: at(18, 40), url: '/faces/shot/70.jpg' }],
    };
    serve([KARA, SECOND, LUCA]);
    server.use(
      http.post('/people/claim', async ({ request }) => {
        const body = (await request.json()) as { id: number; name: string };
        posts.push({ url: '/people/claim', body });
        return HttpResponse.json({ ok: true, name: body.name, undo: `undo-${body.id}` });
      }),
      http.post('/faces/not', async ({ request }) => {
        posts.push({ url: '/faces/not', body: await request.json() });
        return HttpResponse.json({ ok: true, marked: 1, undo: [] });
      }),
      http.post('/people/undo', async ({ request }) => {
        posts.push({ url: '/people/undo', body: await request.json() });
        return HttpResponse.json({ ok: true });
      }),
    );
    renderPage(<WhosThis last={{}} live={[]} wide />);
    await row(/^Face 5/);
    await userEvent.click(screen.getByRole('radio', { name: 'By person' }));
    expect(screen.getByRole('radio', { name: 'Kara (2)' })).toBeChecked(); // the person with the most guessed
    const tiles = within(screen.getByRole('group', { name: 'Faces guessed as Kara' })).getAllByRole('button');
    expect(tiles.map(t => t.getAttribute('aria-label'))).toEqual([
      'Face 5, seen 42 times, match 86%',
      'Face 7, seen 42 times, match 70%',
    ]);
    await userEvent.click(tiles[1] as HTMLElement);
    await userEvent.click(screen.getByRole('button', { name: 'Label 1 as Kara' }));
    expect(await screen.findByText('Named 1 face group as Kara. Marked 1 as not Kara.')).toBeVisible();
    expect(posts).toEqual([
      { url: '/people/claim', body: { kind: 'face', id: 5, name: 'Kara' } },
      { url: '/faces/not', body: { name: 'Kara', groups: [7], shots: [], not: true } },
    ]);
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    await waitFor(() =>
      expect(posts.slice(2)).toEqual([
        { url: '/people/undo', body: expect.objectContaining({ undo: 'undo-5' }) },
        { url: '/faces/not', body: { name: 'Kara', groups: [7], shots: [], not: false } },
      ]),
    );
  });

  /** Face groups guessed as Kara (ids from 100), and the claim, not, and undo calls, with `claimFails` and `undoMissing`
   *  ids answered as failures. */
  function kates(n: number, { claimFails = [] as number[], undoMissing = [] as string[] } = {}) {
    const list = Array.from({ length: n }, (_, i) => ({
      ...KARA,
      id: 100 + i,
      label: `Face ${100 + i}`,
      score: 90 - i,
      shots: [{ id: 100 + i, ts: at(18, 40), url: `/faces/shot/${100 + i}.jpg` }],
    }));
    serve(list);
    server.use(
      http.post('/people/claim', async ({ request }) => {
        const body = (await request.json()) as { id: number; name: string };
        posts.push({ url: '/people/claim', body });
        return claimFails.includes(body.id)
          ? HttpResponse.json({ ok: false, error: 'busy' })
          : HttpResponse.json({ ok: true, name: body.name, undo: `undo-${body.id}` });
      }),
      http.post('/faces/not', async ({ request }) => {
        posts.push({ url: '/faces/not', body: await request.json() });
        return HttpResponse.json({ ok: true, marked: 1, undo: [] });
      }),
      http.post('/people/undo', async ({ request }) => {
        const body = (await request.json()) as { undo: string };
        posts.push({ url: '/people/undo', body });
        return undoMissing.includes(body.undo)
          ? HttpResponse.json({ ok: false, error: 'that change is too old to undo' }, { status: 404 })
          : HttpResponse.json({ ok: true });
      }),
    );
    renderPage(<WhosThis last={{}} live={[]} wide />);
  }
  const byPerson = async () => {
    await row(/^Face 100/);
    await userEvent.click(screen.getByRole('radio', { name: 'By person' }));
    return within(screen.getByRole('group', { name: 'Faces guessed as Kara' })).getAllByRole('button');
  };

  it('brings a face whose claim failed back to the list, and keeps Undo for the ones that were named', async () => {
    kates(2, { claimFails: [101] });
    await byPerson();
    await userEvent.click(screen.getByRole('button', { name: 'Label 2 as Kara' }));
    expect(await screen.findByText(/^Named 1 face group as Kara\. busy$/)).toBeVisible();
    expect(screen.getByRole('button', { name: 'Undo' })).toBeEnabled();
    expect(within(screen.getByRole('group', { name: 'Faces guessed as Kara' })).getAllByRole('button')).toHaveLength(1);
    await userEvent.click(screen.getByRole('radio', { name: 'List' }));
    expect(await row(/^Face 101/)).toBeVisible();
  });

  it('names at most 50 faces in one save, so one Undo covers them all', async () => {
    kates(55);
    const tiles = await byPerson();
    expect(tiles).toHaveLength(50);
    expect(screen.getByText(/Showing the best 50 of 55/)).toBeVisible();
    expect(screen.getByRole('button', { name: 'Label 50 as Kara' })).toBeEnabled();
  });

  it('undoes newest first and carries on past a name the server no longer keeps', async () => {
    kates(3, { undoMissing: ['undo-102'] });
    await byPerson();
    await userEvent.click(screen.getByRole('button', { name: 'Label 3 as Kara' }));
    await screen.findByText('Named 3 face groups as Kara.');
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(await screen.findByText(/^Undid the rest, but that change is too old to undo\.$/)).toBeVisible();
    expect(posts.filter(p => p.url === '/people/undo').map(p => (p.body as { undo: string }).undo)).toEqual([
      'undo-102',
      'undo-101',
      'undo-100',
    ]);
  });

  it("keeps Live's rail to the list: no sort, size, or count", async () => {
    serve([KARA]);
    show();
    await row(/^Face 5/);
    expect(screen.queryByRole('button', { name: /^Sort by certainty/ })).toBeNull();
    expect(screen.queryByRole('radiogroup', { name: 'Picture size' })).toBeNull();
  });

  it("gives People's wide form the sort and size", async () => {
    serve([KARA]);
    renderPage(<WhosThis last={{}} live={[]} wide />);
    await row(/^Face 5/);
    expect(screen.getByRole('button', { name: 'Sort by certainty, highest first' })).toBeInTheDocument();
    expect(screen.getByRole('radiogroup', { name: 'Picture size' })).toBeInTheDocument();
  });

  it('sorts by certainty, highest first, and the Certainty toggle flips it to lowest first', async () => {
    serve([LUCA, KARA]);
    renderPage(<WhosThis last={{}} live={[]} wide />);
    await row(/^Face 5/);
    const order = () => screen.getAllByRole('listitem').map(li => li.getAttribute('aria-label'));
    expect(order()).toEqual(['Face 5: Probably Kara 86%', 'Face 6: Maybe Luca 58%']);
    await userEvent.click(screen.getByRole('button', { name: 'Sort by certainty, highest first' }));
    expect(order()).toEqual(['Face 6: Maybe Luca 58%', 'Face 5: Probably Kara 86%']);
    await userEvent.click(screen.getByRole('button', { name: 'Sort by certainty, lowest first' }));
    expect(order()).toEqual(['Face 5: Probably Kara 86%', 'Face 6: Maybe Luca 58%']);
  });
});

describe("voices in Who's this?", () => {
  it('ranks voices and faces together, best guess first, a voice with its clip and lines heard', async () => {
    serveVoices([VOICE], [KARA, NOBODY]);
    show();
    await row(/^Speaker 12/);
    expect(screen.getAllByRole('listitem').map(li => li.getAttribute('aria-label'))).toEqual([
      'Speaker 12: Probably Dana 90%',
      'Face 5: Probably Kara 86%',
      'Face 9: No good guess',
    ]);
    const voice = await row(/^Speaker 12/);
    expect(within(voice).getByText(/^Heard 14 lines, /)).toBeInTheDocument();
    expect(within(voice).getByText('“Shall we put the kettle on?”')).toBeInTheDocument();
    expect(
      within(voice).getByRole('button', { name: 'Play Speaker 12: Shall we put the kettle on?' }),
    ).toBeInTheDocument();
  });

  it('plays and stops the clip with Space on the selected voice card, but not from its name box', async () => {
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(() => Promise.resolve());
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    serveVoices([VOICE]);
    show();
    const voice = await row(/^Speaker 12/);
    await userEvent.click(nameBox(voice));
    await userEvent.keyboard(' ');
    expect(within(voice).getByRole('button', { name: /^Play Speaker 12/ })).toBeInTheDocument();
    voice.focus();
    await userEvent.keyboard(' ');
    expect(await within(voice).findByRole('button', { name: /^Stop Speaker 12/ })).toBeInTheDocument();
    await userEvent.keyboard(' ');
    expect(await within(voice).findByRole('button', { name: /^Play Speaker 12/ })).toBeInTheDocument();
  });

  it('names a voice by selecting the guess', async () => {
    serveVoices([VOICE]);
    show();
    await pick(await row(/^Speaker 12/), 'Dana');
    await waitFor(() =>
      expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'voice', id: 12, name: 'Dana' } }]),
    );
  });

  it('takes a labeled row off the list at once, even while the server still lists it', async () => {
    serveVoices([VOICE]); // the voice list here never drops it, like the server's cached rows
    show();
    await pick(await row(/^Speaker 12/), 'Dana');
    await waitFor(() => expect(screen.queryByRole('listitem')).not.toBeInTheDocument());
    expect(screen.getByRole('status', { name: 'Decisions' })).toHaveTextContent(/tagged as Dana/);
  });

  it('labels the clip as several people, and Undo puts it back', async () => {
    serveVoices([VOICE]);
    show();
    const voice = await row(/^Speaker 12/);
    voice.focus();
    await userEvent.keyboard('m');
    await waitFor(() =>
      expect(posts).toEqual([{ url: '/speakers/refine', body: { action: 'multi', line: VOICE.clips[0]?.line } }]),
    );
    const said = screen.getByRole('status', { name: 'Decisions' });
    expect(said).toHaveTextContent('Labeled “Shall we put the kettle on?” as Multiple speakers.');
    const restored = await row(/^Speaker 12/);
    expect(nameBox(restored)).toBeEnabled(); // the voice still waits
    await userEvent.click(within(said).getByRole('button', { name: 'Undo' }));
    await waitFor(() =>
      expect(posts.at(-1)).toEqual({ url: '/speakers/refine/undo', body: { lines: [VOICE.clips[0]?.line] } }),
    );
    // A new message is a new toast in the shared queue (lib/Toast)
    expect(await screen.findByRole('status', { name: 'Decisions' })).toHaveTextContent('Undid 1 label.');
  });

  it('names a clip from before line keys by its time', async () => {
    const t = at(18, 2);
    const old = { ts: t, text: 'Bye now.', dur: 1.8, url: `/clip?ts_start=${t}` };
    serveVoices([{ ...VOICE, clips: [old] }]);
    show();
    await more(await row(/^Speaker 12/), 'Multiple speakers');
    await waitFor(() => expect(posts).toEqual([{ url: '/speakers/refine', body: { action: 'multi', ts_start: t } }]));
    await userEvent.click(
      within(screen.getByRole('status', { name: 'Decisions' })).getByRole('button', { name: 'Undo' }),
    );
    await waitFor(() =>
      expect(posts.at(-1)).toEqual({ url: '/speakers/refine/undo', body: { lines: [t.toFixed(3)] } }),
    );
  });

  it("shows a voice's answers as Multi, Noise, TV, and Ignore, each with its key and what it does", async () => {
    serveVoices([VOICE]);
    show();
    const voice = await row(/^Speaker 12/);
    const answers = ['Multiple speakers', 'Noise: not a voice, drop just this clip', 'TV or radio', 'Ignore'].map(
      name => within(voice).getByRole('button', { name }),
    );
    expect(answers.map(b => b.textContent)).toEqual(['MultiM', 'NoiseN', 'TVT', 'IgnoreI']);
    expect(within(voice).queryByRole('button', { name: /Nobody/ })).toBeNull();
    const hints = [
      'Multi (M): Several people talk over each other in this clip. The clip is labeled Multiple speakers and leaves the questions; the voice stays.',
      'Noise (N): Not a voice at all (a door, a kettle). Drops just this clip from the questions; the voice stays.',
      "TV (T): This voice is a TV or radio. All its lines stay in the transcript as nobody's, and the voice leaves the list. Undo brings it back.",
      "Ignore (I): Set this whole voice aside. It leaves the list and you won't be asked about it again.",
    ];
    for (const [i, button] of answers.entries()) {
      await userEvent.hover(button);
      expect(await screen.findByText(hints[i] ?? '')).toHaveAttribute('role', 'tooltip');
      await userEvent.unhover(button);
    }
  });

  it('ignores a voice for good', async () => {
    serveVoices([VOICE]);
    show();
    await more(await row(/^Speaker 12/), 'Ignore');
    await waitFor(() => expect(posts).toEqual([{ url: '/inbox/skip', body: { kind: 'voice', id: 12 } }]));
  });

  it("shows People's wide form with the recent decisions under it, and no Live anchor", async () => {
    serveVoices([VOICE]);
    renderPage(<WhosThis wide />);
    await row(/^Speaker 12/);
    expect(screen.getByText(/Faces and voices nobody has named yet/)).toBeInTheDocument();
    expect(screen.getByText(/^Recent decisions/)).toBeInTheDocument();
    expect(document.getElementById('whos-this')).toBeNull(); // Live's rail keeps that id; both stay mounted
  });
});

describe("answering Who's this? quickly", () => {
  it('drops just one clip with Noise (N), and keeps the voice', async () => {
    serveVoices([VOICE]);
    show();
    const voice = await row(/^Speaker 12/);
    voice.focus();
    await userEvent.keyboard('d'); // Nobody's old key does nothing now
    expect(posts).toEqual([]);
    await userEvent.keyboard('n');
    await waitFor(() =>
      expect(posts).toEqual([{ url: '/inbox/skip', body: { kind: 'line', id: VOICE.clips[0]?.line } }]),
    );
    expect(screen.getByRole('status', { name: 'Decisions' })).toHaveTextContent(
      'Removed that clip from the questions.',
    );
    expect(nameBox(await row(/^Speaker 12/))).toBeEnabled();
  });

  it('marks a voice as a TV or radio, and Undo brings it back', async () => {
    serveVoices([VOICE]);
    const sent: { url: string; body: unknown }[] = [];
    server.use(
      http.post('/voices/unknown/media', async ({ request }) => {
        sent.push({ url: '/voices/unknown/media', body: await request.json() });
        return HttpResponse.json({ ok: true, lines: [VOICE.clips[0]?.line], undo: 'tv-voice' });
      }),
      http.post('/transcript/undo', async ({ request }) => {
        sent.push({ url: '/transcript/undo', body: await request.json() });
        return HttpResponse.json({ ok: true, lines: [VOICE.clips[0]?.line] });
      }),
    );
    show();
    const voice = await row(/^Speaker 12/);
    voice.focus();
    await userEvent.keyboard('t');
    await waitFor(() => expect(sent).toEqual([{ url: '/voices/unknown/media', body: { group: 12 } }]));
    const toast = screen.getByRole('status', { name: 'Decisions' });
    expect(toast).toHaveTextContent('marked as a TV or radio');
    await userEvent.click(within(toast).getByRole('button', { name: 'Undo' }));
    await waitFor(() => expect(sent.at(-1)).toEqual({ url: '/transcript/undo', body: { undo: 'tv-voice' } }));
    expect(await screen.findByRole('status', { name: 'Decisions' })).toHaveTextContent('waiting for a name again');
    expect(await row(/^Speaker 12/)).toBeInTheDocument();
  });

  it('has a way back to the list of people on a narrow window', async () => {
    serve([KARA]);
    let back = 0;
    renderPage(<WhosThis wide onBack={() => back++} />);
    await userEvent.click(await screen.findByRole('button', { name: 'Back to people' }));
    expect(back).toBe(1);
  });
});

describe('seen or heard together', () => {
  const together = (name: string, moments = 3) => ({
    name,
    moments,
    last: at(18, 40),
    times: [at(18, 40), at(12, 5), at(9, 30)].slice(0, moments),
    others: {},
  });

  it('words a suggestion for a face and for a voice', () => {
    expect(togetherText('face', together('Dana'), NOW)).toBe(
      `Dana was heard while this face was in view: 3 times, last ${seenAt(at(18, 40), NOW)}.`,
    );
    expect(togetherText('voice', together('Kara', 2), NOW)).toMatch(
      /^Kara was in view while this voice spoke: 2 times/,
    );
    expect(togetherText('face', null, NOW)).toBe('');
    expect(guessText({ ...NOBODY, together: together('Dana') })).toBe('Maybe Dana, together 3 times');
  });

  it('suggests a name for a face with no other guess, and selecting it tags the face', async () => {
    serve([{ ...NOBODY, together: together('Dana') }]);
    show();
    const face = await row(/^Face 9/);
    expect(within(face).getByText(/^Dana was heard while this face was in view: 3 times/)).toBeInTheDocument();
    await pick(face, 'Dana');
    await waitFor(() => expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'face', id: 9, name: 'Dana' } }]));
  });

  it('sorts a face with a suggestion among the maybes, ahead of one with no guess seen later', async () => {
    const older = group({ id: 11, label: 'Face 11', last_seen: at(9, 0), together: together('Dana') });
    serve([NOBODY, older]); // NOBODY was seen at 18:40, after Face 11
    show();
    await row(/^Face 11/);
    const labels = screen.getAllByRole('listitem').map(li => li.getAttribute('aria-label'));
    expect(labels).toEqual(['Face 11: Maybe Dana, together 3 times', 'Face 9: No good guess']);
  });

  it('offers one guess when the suggestion is the guess in other capitals', async () => {
    serve([{ ...KARA, together: together('kara') }]);
    show();
    await userEvent.click(nameBox(await row(/^Face 5/)));
    expect(screen.getAllByRole('option')).toHaveLength(1);
  });

  it('offers it beside a guess that names someone else, and not twice when they agree', async () => {
    serve([
      { ...LUCA, together: together('Sam') },
      { ...KARA, together: together('Kara') },
    ]);
    show();
    const luca = await row(/^Face 6/);
    await userEvent.click(nameBox(luca));
    expect(screen.getAllByRole('option').map(o => o.textContent)).toEqual([
      'Luca (guess, 58%)',
      'Sam (guess, together 3 times)',
    ]);
    await userEvent.click(screen.getByRole('option', { name: 'Sam (guess, together 3 times)' }));
    await waitFor(() => expect(posts).toEqual([{ url: '/people/claim', body: { kind: 'face', id: 6, name: 'Sam' } }]));
  });
});

describe('optimistic action variants', () => {
  const voice: UnknownVoice = {
    ...KARA,
    id: 5,
    label: 'Voice 5',
    clips: [{ ts: 100, line: 'synthetic-line', text: 'Synthetic clip', dur: 2, url: '/synthetic.wav' }],
  };
  it.each(['Kara', 'Ignore', 'Robin', 'Dana'])(
    'handles delayed voice %s without hiding a face with the same id',
    async action => {
      serve([KARA], [], [{ ...voice, together: { name: 'Dana', moments: 2, last: NOW, times: [], others: {} } }]);
      let release!: (value: Response) => void;
      server.use(
        http.post(
          action === 'Ignore' ? '/inbox/skip' : '/people/claim',
          () =>
            new Promise<Response>(resolve => {
              release = resolve;
            }),
        ),
      );
      const { client } = show();
      const card = await row(/^Voice 5/);
      if (action === 'Robin') {
        await userEvent.type(nameBox(card), 'Robin{Enter}');
      } else if (action === 'Ignore') {
        await more(card, 'Ignore');
      } else {
        await pick(card, action);
      }
      expect(card).not.toBeVisible();
      expect(await row(/^Face 5/)).toBeVisible();
      await waitFor(() => expect(release).toBeTypeOf('function'));
      // Even a poll omitting the pending voice must not unmount its draft/error state.
      await act(async () => {
        client.setQueryData(['voices', 'unknown', 'groups'], { groups: [], skipped: [], count: 0 });
      });
      await act(async () => release(HttpResponse.error()));
      expect(within(await row(/^Voice 5/)).getByRole('alert')).toBeVisible();
      if (action === 'Robin') expect(within(card).getByRole('combobox')).toHaveValue('Robin');
    },
  );
});

it('keeps loaded faces and voices visible when a face refresh fails', async () => {
  serve([KARA], [], [VOICE]);
  const { client } = show();
  await row(/^Face 5/);
  await row(/^Speaker 12/);
  server.use(
    http.get('/faces/unknown/groups', () => HttpResponse.json({ error: 'face history busy' }, { status: 503 })),
  );
  await act(async () => {
    await client.refetchQueries({ queryKey: UNKNOWN_GROUPS_KEY });
  });
  expect(await screen.findByRole('alert')).toHaveTextContent('face history busy');
  expect(screen.getByRole('listitem', { name: /^Face 5/ })).toBeInTheDocument();
  expect(screen.getByRole('listitem', { name: /^Speaker 12/ })).toBeInTheDocument();
  expect(within(screen.getByRole('alert')).getByRole('button', { name: 'Retry' })).toBeEnabled();
});

it('shows partial results while voices load slowly, time out, and recover on Retry', async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  const client = newQueryClient(); // production retry defaults must not hide the first deadline
  let waiting = true;
  const fetcher = vi.fn<typeof fetch>((url, init) => {
    if (url === '/faces/unknown/groups') return Promise.resolve(Response.json({ groups: [KARA], skipped: [] }));
    if (url === '/people') return Promise.resolve(Response.json({ people: [] }));
    if (url === '/voices/unknown/groups') {
      if (!waiting) return Promise.resolve(Response.json({ groups: [VOICE], skipped: [] }));
      return new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), {
          once: true,
        });
      });
    }
    return Promise.reject(new Error(`Unexpected fetch ${url}`));
  });
  vi.stubGlobal('fetch', fetcher);
  onTestFinished(() => {
    client.clear();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });
  render(
    <Providers client={client} test>
      <WhosThis />
    </Providers>,
  );
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  // Faces wait for voices under row-sized placeholders, so voices sorted in later can't move them (layout shift).
  expect(screen.queryByRole('listitem', { name: /^Face 5/ })).toBeNull();
  expect(screen.getByTestId('row-placeholders')).toBeInTheDocument();
  expect(screen.getByText('Loading voices…')).toBeInTheDocument();
  await act(async () => {
    await vi.advanceTimersByTimeAsync(SLOW_MS);
  });
  expect(screen.getByText(/Still loading voices/)).toBeInTheDocument();
  // Voices still slow at PAGE_SLOW_MS: the faces show on their own.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(PAGE_SLOW_MS);
  });
  expect(screen.getByRole('listitem', { name: /^Face 5/ })).toBeInTheDocument();
  expect(screen.queryByTestId('row-placeholders')).toBeNull();
  await act(async () => {
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS - PAGE_SLOW_MS);
  });
  expect(screen.getByRole('alert')).toHaveTextContent("Couldn't load the voices waiting for a name");
  expect(screen.getByRole('alert')).toHaveTextContent(/The server took too long to answer/);
  expect(screen.getByRole('listitem', { name: /^Face 5/ })).toBeInTheDocument();
  expect(screen.queryByText('Nobody is waiting for a name.')).not.toBeInTheDocument();
  expect(fetcher.mock.calls.filter(([url]) => url === '/voices/unknown/groups')).toHaveLength(1);
  waiting = false;
  await act(async () => {
    fireEvent.click(within(screen.getByRole('alert')).getByRole('button', { name: 'Retry' }));
    await vi.advanceTimersByTimeAsync(0);
  });
  expect(screen.getByRole('listitem', { name: /^Speaker 12/ })).toBeInTheDocument();
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it('keeps group lists and profile picks in the page lane while status polls wait', async () => {
  const replies: ((response: Response) => void)[] = [];
  const fetcher = vi.fn<typeof fetch>(() => new Promise<Response>(resolve => replies.push(resolve)));
  vi.stubGlobal('fetch', fetcher);
  const client = newQueryClient();
  onTestFinished(() => {
    client.clear();
    vi.unstubAllGlobals();
  });
  const ambient = request('/ambient', undefined, undefined, 'background');
  const services = request('/services', undefined, undefined, 'background');
  const faces = client.fetchQuery(unknownGroupsQuery());
  await Promise.resolve();
  expect(fetcher.mock.calls.map(([url]) => url)).toEqual(['/ambient', '/faces/unknown/groups']);
  replies[1]?.(Response.json({ groups: [] }));
  await faces;
  const voices = client.fetchQuery(unknownVoicesQuery());
  await Promise.resolve();
  expect(fetcher).toHaveBeenLastCalledWith('/voices/unknown/groups', expect.anything());
  replies[2]?.(Response.json({ groups: [] }));
  await voices;
  const picks = getUnnamed();
  await Promise.resolve();
  expect(fetcher).toHaveBeenLastCalledWith('/inbox/unnamed', expect.anything());
  replies[3]?.(Response.json({ faces: [], videos: [], voices: [] }));
  await picks;
  replies[0]?.(Response.json({}));
  await ambient;
  expect(fetcher).toHaveBeenLastCalledWith('/services', expect.anything());
  replies[4]?.(Response.json({}));
  await services;
});

it('retries failed faces without restarting a failed voice request', async () => {
  serve([]);
  let facesFail = true;
  const reads = { faces: 0, voices: 0 };
  server.use(
    http.get('/faces/unknown/groups', () => {
      reads.faces++;
      return facesFail
        ? HttpResponse.json({ error: 'faces busy' }, { status: 503 })
        : HttpResponse.json({ groups: [KARA] });
    }),
    http.get('/voices/unknown/groups', () => {
      reads.voices++;
      return HttpResponse.json({ error: 'voices busy' }, { status: 503 });
    }),
  );
  show();
  await waitFor(() => expect(screen.getAllByRole('alert')).toHaveLength(2));
  const faceError = screen.getAllByRole('alert').find(alert => alert.textContent?.includes('faces busy'));
  expect(faceError).toBeDefined();
  facesFail = false;
  await userEvent.click(within(faceError as HTMLElement).getByRole('button', { name: 'Retry' }));
  expect(await row(/^Face 5/)).toBeInTheDocument();
  expect(screen.getByRole('alert')).toHaveTextContent('voices busy');
  expect(reads).toEqual({ faces: 2, voices: 1 });
});

it('keeps ready voices usable while pending faces hide unknown skips and the empty state', async () => {
  serve([], [], [VOICE]);
  let pending = true;
  server.use(
    http.get('/faces/unknown/groups', () =>
      HttpResponse.json(
        pending ? { pending: true, groups: [], count: null, skipped: [] } : { groups: [], count: 0, skipped: [12] },
        { status: pending ? 202 : 200 },
      ),
    ),
  );
  const { client } = renderPage(
    <WhosThis
      last={{ cam0: { cam: 'cam0', ts: NOW, faces: [{ name: 'Face 12', bbox: [10, 10, 80, 80] }] } }}
      live={['cam0']}
    />,
  );
  expect(await row(/^Speaker 12/)).toBeInTheDocument();
  await waitFor(() => expect(client.getQueryState(UNKNOWN_GROUPS_KEY)?.status).toBe('success'));
  expect(screen.queryByRole('listitem', { name: /^Face 12/ })).not.toBeInTheDocument();
  expect(screen.getByRole('status', { name: 'Loading unnamed lists' })).toHaveTextContent('Loading faces');
  expect(screen.queryByText('Nobody is waiting for a name.')).not.toBeInTheDocument();
  pending = false;
  await act(async () => {
    await client.refetchQueries({ queryKey: UNKNOWN_GROUPS_KEY });
  });
  await waitFor(() => expect(screen.queryByRole('status', { name: 'Loading unnamed lists' })).not.toBeInTheDocument());
  expect(screen.queryByRole('listitem', { name: /^Face 12/ })).not.toBeInTheDocument();
});

it('never calls a pending empty list nobody waiting and exposes a subsequent failure with Retry', async () => {
  serve([]);
  server.use(
    http.get('/voices/unknown/groups', () =>
      HttpResponse.json({ pending: true, groups: [], count: null, skipped: [] }, { status: 202 }),
    ),
  );
  const { client } = show();
  await waitFor(() => expect(client.getQueryState(['voices', 'unknown', 'groups'])?.status).toBe('success'));
  expect(screen.getByRole('status', { name: 'Loading unnamed lists' })).toHaveTextContent('Loading voices');
  expect(screen.queryByText('Nobody is waiting for a name.')).not.toBeInTheDocument();
  server.use(
    http.get('/voices/unknown/groups', () =>
      HttpResponse.json({ error: 'voice bootstrap unavailable' }, { status: 500 }),
    ),
  );
  await act(async () => {
    await client.refetchQueries({ queryKey: ['voices', 'unknown', 'groups'] });
  });
  expect(await screen.findByRole('alert')).toHaveTextContent('voice bootstrap unavailable');
  expect(within(screen.getByRole('alert')).getByRole('button', { name: 'Retry' })).toBeEnabled();
  expect(screen.queryByText('Nobody is waiting for a name.')).not.toBeInTheDocument();
  serve([]);
  await userEvent.click(within(screen.getByRole('alert')).getByRole('button', { name: 'Retry' }));
  expect(await screen.findByText('Nobody is waiting for a name.')).toBeInTheDocument();
});

it('keeps saying nobody is waiting through a background refresh, with no loading line', async () => {
  serve([]);
  const { client } = show();
  expect(await screen.findByText('Nobody is waiting for a name.')).toBeInTheDocument();
  let release!: () => void;
  server.use(
    http.get('/faces/unknown/groups', async () => {
      await new Promise<void>(resolve => {
        release = resolve;
      });
      return HttpResponse.json({ groups: [], count: 0, skipped: [] });
    }),
  );
  void client.refetchQueries({ queryKey: UNKNOWN_GROUPS_KEY });
  await waitFor(() => expect(client.getQueryState(UNKNOWN_GROUPS_KEY)?.fetchStatus).toBe('fetching'));
  expect(screen.getByText('Nobody is waiting for a name.')).toBeInTheDocument();
  expect(screen.queryByText(/Loading/)).not.toBeInTheDocument();
  release();
});

describe('Show all and Not this person', () => {
  const shot = (id: string) => ({ id, ts: at(18, 40), url: `/faces/shot/${id}.jpg` });
  function pictures(ids: string[], ignored = false) {
    let left = ids;
    server.use(
      http.get('/faces/unknown/5/sightings', ({ request }) => {
        const q = new URL(request.url).searchParams;
        const offset = Number(q.get('offset'));
        const page = left.slice(offset, offset + Number(q.get('limit')));
        return HttpResponse.json({ sightings: page.map(shot), total: left.length });
      }),
      http.post('/faces/unknown/reject', async ({ request }) => {
        const body = (await request.json()) as { group: number; ids: string[] };
        posts.push({ url: '/faces/unknown/reject', body });
        left = left.filter(id => !body.ids.includes(id));
        return HttpResponse.json({ ok: true, removed: body.ids.length, left: left.length, ignored });
      }),
    );
  }

  it('opens every picture with A on the focused row', async () => {
    serve([{ ...KARA, shot_count: 3 }]);
    pictures(['a', 'b', 'c']);
    show();
    const card = await row(/^Face 5/);
    // No Show all photos button of its own: the Show all link, A, and Space open them
    expect(within(card).queryByRole('button', { name: /^Show all photos/ })).toBeNull();
    expect(within(card).getByRole('button', { name: 'Show all (3)' })).toHaveAttribute('aria-keyshortcuts', 'A');
    card.focus();
    await userEvent.keyboard('a');
    expect(await within(card).findByRole('region', { name: 'Every picture of Face 5' })).toBeVisible();
  });

  it('opens every picture with Space on the selected row, but not from its name box', async () => {
    serve([{ ...KARA, shot_count: 3 }]);
    pictures(['a', 'b', 'c']);
    show();
    const card = await row(/^Face 5/);
    await userEvent.click(nameBox(card));
    await userEvent.keyboard(' ');
    expect(within(card).queryByRole('region', { name: 'Every picture of Face 5' })).toBeNull();
    card.focus();
    await userEvent.keyboard(' ');
    expect(await within(card).findByRole('region', { name: 'Every picture of Face 5' })).toBeVisible();
  });

  it('refreshes an open empty grid as pending crops land and stops polling when none remain', async () => {
    let ready = false;
    let reads = 0;
    server.use(
      http.get('/faces/unknown/5/sightings', () => {
        reads++;
        return HttpResponse.json({
          sightings: ready ? [shot('a'), shot('b')] : [],
          total: ready ? 2 : 0,
          shots_pending: ready ? 0 : 2,
          shots_unavailable: 1,
        });
      }),
    );
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    try {
      renderPage(<ShotPicker id={5} label="Face 5" total={0} onClose={vi.fn()} onRejected={vi.fn()} />);
      const grid = await screen.findByRole('region', { name: 'Every picture of Face 5' });
      expect(await within(grid).findByText('Processing 2 detections in the background…')).toBeVisible();
      expect(grid.querySelector('img')).toBeNull();
      ready = true;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
      await waitFor(() => expect(within(grid).getAllByRole('button', { name: /^Picture from/ })).toHaveLength(2));
      expect(within(grid).queryByText(/Processing .* detections/)).toBeNull();
      expect(within(grid).getByText("1 detection can't be shown.")).toBeVisible();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
      expect(reads).toBe(2);
      expect(grid.querySelector('img')).toHaveAttribute('loading', 'lazy');
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows every picture, takes the selected ones out, and keeps the rest', async () => {
    serve([{ ...KARA, shot_count: 3 }]);
    pictures(['a', 'b', 'c']);
    show();
    const card = await row(/^Face 5/);
    fireEvent.click(within(card).getByRole('button', { name: 'Show all (3)' }));
    const grid = await within(card).findByRole('region', { name: 'Every picture of Face 5' });
    const picks = await within(grid).findAllByRole('button', { name: /^Picture from/ });
    expect(picks).toHaveLength(3);
    const remove = within(grid).getByRole('button', { name: 'Not this person' });
    expect(remove).toBeDisabled();
    fireEvent.click(picks[0] as HTMLElement);
    fireEvent.click(picks[2] as HTMLElement);
    fireEvent.click(picks[2] as HTMLElement); // a second click puts it back
    fireEvent.click(picks[1] as HTMLElement);
    expect(picks.map(p => p.getAttribute('aria-pressed'))).toEqual(['true', 'true', 'false']);
    fireEvent.click(remove);
    expect(await screen.findByText('Took 2 pictures out of Face 5.')).toBeInTheDocument();
    expect(posts).toEqual([{ url: '/faces/unknown/reject', body: { group: 5, ids: ['a', 'b'] } }]);
    expect(within(grid).getAllByRole('button', { name: /^Picture from/ })).toHaveLength(1);
    expect(screen.getByRole('listitem', { name: /^Face 5/ })).toBeInTheDocument();
  });

  it('keeps ready picture pagination separate from all remaining detections after a rejection', async () => {
    serve([{ ...KARA, shot_count: 2 }]);
    server.use(
      http.get('/faces/unknown/5/sightings', () => HttpResponse.json({ sightings: [shot('a'), shot('b')], total: 2 })),
      http.post('/faces/unknown/reject', () => HttpResponse.json({ ok: true, removed: 1, left: 6, ignored: false })),
    );
    show();
    const card = await row(/^Face 5/);
    fireEvent.click(within(card).getByRole('button', { name: 'Show all (2)' }));
    const grid = await within(card).findByRole('region', { name: 'Every picture of Face 5' });
    const picks = await within(grid).findAllByRole('button', { name: /^Picture from/ });
    fireEvent.click(picks[0] as HTMLElement);
    fireEvent.click(within(grid).getByRole('button', { name: 'Not this person' }));
    await waitFor(() => expect(within(grid).getAllByRole('button', { name: /^Picture from/ })).toHaveLength(1));
    expect(within(grid).queryByRole('button', { name: /Load .* more/ })).toBeNull();
    expect(within(grid).getByText(/^1 picture\./)).toBeVisible();
  });

  it('takes a group with nothing left off the list', async () => {
    serve([{ ...KARA, shot_count: 1 }]);
    pictures(['a'], true);
    show();
    const card = await row(/^Face 5/);
    fireEvent.click(within(card).getByRole('button', { name: 'Show all (1)' }));
    fireEvent.click(await within(card).findByRole('button', { name: /^Picture from/ }));
    fireEvent.click(within(card).getByRole('button', { name: 'Not this person' }));
    expect(await screen.findByText(/Nothing was left, so it was put aside\./)).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole('listitem', { name: /^Face 5/ })).not.toBeInTheDocument());
  });

  it('loads the rest a page at a time', async () => {
    const ids = Array.from({ length: 70 }, (_, i) => `s${i}`);
    serve([{ ...KARA, shot_count: 70 }]);
    pictures(ids);
    show();
    const card = await row(/^Face 5/);
    fireEvent.click(within(card).getByRole('button', { name: 'Show all (70)' }));
    const more = await within(card).findByRole('button', { name: 'Load 10 more' });
    expect(within(card).getAllByRole('button', { name: /^Picture from/ })).toHaveLength(60);
    fireEvent.click(more);
    await waitFor(() => expect(within(card).getAllByRole('button', { name: /^Picture from/ })).toHaveLength(70));
    expect(within(card).queryByRole('button', { name: /more$/ })).not.toBeInTheDocument();
  });
});

// The server's claim token retracts one naming decision, independently of clip-label Undo.
describe('claim Undo', () => {
  it.each(['face', 'voice'] as const)('puts an unnamed %s group back once', async kind => {
    let claimed = false;
    let release!: (response: Response) => void;
    let undos = 0;
    serve(kind === 'face' ? [KARA] : [], [], kind === 'voice' ? [VOICE] : []);
    server.use(
      http.get(kind === 'face' ? '/faces/unknown/groups' : '/voices/unknown/groups', () =>
        HttpResponse.json({
          groups: claimed ? [] : [kind === 'face' ? KARA : VOICE],
          skipped: [],
          count: claimed ? 0 : 1,
        }),
      ),
      http.post('/people/claim', () => {
        claimed = true;
        return HttpResponse.json({ ok: true, name: kind === 'face' ? 'Kara' : 'Dana', undo: 'claim-token' });
      }),
      http.post('/people/undo', async ({ request }) => {
        expect(await request.json()).toEqual({ undo: 'claim-token' });
        undos++;
        return new Promise<Response>(resolve => {
          release = resolve;
        });
      }),
    );
    show();
    const pattern = kind === 'face' ? /^Face 5/ : /^Speaker 12/;
    await pick(await row(pattern), kind === 'face' ? 'Kara' : 'Dana');
    const toast = await screen.findByRole('status', { name: 'Decisions' });
    const undo = within(toast).getByRole('button', { name: 'Undo' });
    undo.focus(); // as a click or a Tab to it would
    fireEvent.click(undo);
    fireEvent.click(undo);
    await waitFor(() => expect(undos).toBe(1));
    expect(undo).toBeDisabled();
    claimed = false;
    await act(async () => release(HttpResponse.json({ ok: true, people: [] })));
    expect(nameBox(await row(pattern))).toBeEnabled();
    // The toast's Undo went with its toast: focus lands on the row it brought back, not on the page
    await waitFor(async () => expect(await row(pattern)).toHaveFocus());
    expect(screen.getByRole('status', { name: 'Decisions' })).toHaveTextContent('Undid the name');
    expect(
      within(screen.getByRole('status', { name: 'Decisions' })).queryByRole('button', { name: 'Undo' }),
    ).toBeNull();
  });

  it('keeps Undo available for retry when the request fails before a response', async () => {
    serve([KARA]);
    let undos = 0;
    server.use(
      http.post('/people/claim', () => HttpResponse.json({ ok: true, name: 'Kara', undo: 'claim-token' })),
      http.post('/people/undo', () => {
        undos++;
        return undos === 1
          ? HttpResponse.json({ error: 'Temporarily unavailable' }, { status: 503 })
          : HttpResponse.json({ ok: true, people: [] });
      }),
    );
    show();
    await pick(await row(/^Face 5/), 'Kara');
    await userEvent.click(
      within(await screen.findByRole('status', { name: 'Decisions' })).getByRole('button', { name: 'Undo' }),
    );
    expect(await screen.findByRole('alert', { name: 'Decisions' })).toHaveTextContent('Temporarily unavailable');
    await userEvent.click(
      within(screen.getByRole('alert', { name: 'Decisions' })).getByRole('button', { name: 'Undo' }),
    );
    await waitFor(() => expect(undos).toBe(2));
    expect(nameBox(await row(/^Face 5/))).toBeEnabled();
  });
});

it('reports a consumed claim Undo failure without offering that token again', async () => {
  serve([KARA]);
  server.use(
    http.post('/people/claim', () => HttpResponse.json({ ok: true, name: 'Kara', undo: 'claim-token' })),
    http.post('/people/undo', () =>
      HttpResponse.json({ ok: false, errors: ['Face was restored, but its gallery row was missing'] }),
    ),
  );
  show();
  await pick(await row(/^Face 5/), 'Kara');
  await userEvent.click(
    within(await screen.findByRole('status', { name: 'Decisions' })).getByRole('button', { name: 'Undo' }),
  );
  const error = await screen.findByRole('alert', { name: 'Decisions' });
  expect(error).toHaveTextContent('gallery row was missing');
  expect(within(error).queryByRole('button', { name: 'Undo' })).toBeNull();
});

describe('Yes to all, and the ones with no good guess (Known design pass)', () => {
  const KARA_TOO = group({ ...KARA, id: 8, label: 'Face 8' });
  const ANN = group({
    ...KARA,
    id: 10,
    label: 'Face 10',
    guess: { name: 'Ann', score: 91, band: 'probable', close: null },
    score: 91,
  });
  const wide = () => renderPage(<WhosThis last={{}} live={[]} wide />);

  it('names every probably-right row in one call, by person, and one Undo brings them all back', async () => {
    serve([KARA, KARA_TOO, ANN, LUCA, NOBODY]);
    let undone: unknown = null;
    server.use(
      http.post('/people/assign-many', async ({ request }) => {
        posts.push({ url: '/people/assign-many', body: await request.json() });
        return HttpResponse.json({
          ok: true,
          assigned: [
            { name: 'Kara', faces: 2, voices: 0, videos: 0 },
            { name: 'Ann', faces: 1, voices: 0, videos: 0 },
          ],
          failed: [],
          undo: 'tok-all',
          people: [],
        });
      }),
      http.post('/people/undo', async ({ request }) => {
        undone = await request.json();
        return HttpResponse.json({ ok: true, errors: [], people: [], card: null });
      }),
    );
    wide();
    await userEvent.click(await screen.findByRole('button', { name: 'Say yes to all 3' }));
    await waitFor(() => expect(posts.find(p => p.url === '/people/assign-many')).toBeDefined());
    const sent = posts.find(p => p.url === '/people/assign-many')?.body as { items: { name: string }[] } | undefined;
    // One item per person, in the list's order (the surest first): compared by name here.
    expect([...(sent?.items ?? [])].sort((a, b) => a.name.localeCompare(b.name))).toEqual([
      { name: 'Ann', faces: [10], voices: [], videos: [] },
      { name: 'Kara', faces: [5, 8], voices: [], videos: [] },
    ]);
    await waitFor(() => expect(screen.queryByRole('listitem', { name: /^Face 5:/ })).not.toBeInTheDocument());
    expect(screen.getByRole('listitem', { name: /^Face 6:/ })).toBeInTheDocument(); // a Maybe stays
    expect(await screen.findByText('Named 3 faces: 2 as Kara, 1 as Ann.')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    await waitFor(() => expect(undone).toEqual({ undo: 'tok-all' }));
    expect(await row(/^Face 5:/)).toBeInTheDocument();
    expect(await row(/^Face 10:/)).toBeInTheDocument();
  });

  it('brings a row back at once when it could not be named', async () => {
    serve([KARA, ANN]);
    server.use(
      http.post('/people/assign-many', () =>
        HttpResponse.json({
          ok: true,
          assigned: [{ name: 'Kara', faces: 1, voices: 0, videos: 0 }],
          failed: [{ name: 'Ann', kind: 'face', id: 10, error: 'gone' }],
          undo: 'tok',
          people: [],
        }),
      ),
    );
    wide();
    await userEvent.click(await screen.findByRole('button', { name: 'Say yes to all 2' }));
    expect(await screen.findByText(/1 couldn't be named\.$/)).toBeInTheDocument();
    expect(await row(/^Face 10:/)).toBeInTheDocument();
    expect(screen.queryByRole('listitem', { name: /^Face 5:/ })).not.toBeInTheDocument();
  });

  it('offers no Yes to all for fewer than two sure rows', async () => {
    serve([KARA, LUCA, NOBODY]);
    wide();
    await row(/^Face 5:/);
    expect(screen.queryByRole('button', { name: /^Say yes to all/ })).not.toBeInTheDocument();
  });

  it('folds the rows with no good guess under a count, and still counts them as waiting', async () => {
    serve([KARA, LUCA, NOBODY]);
    wide();
    await row(/^Face 5:/);
    expect(screen.getByText('3 waiting')).toBeInTheDocument();
    expect(screen.queryByRole('listitem', { name: /^Face 9:/ })).toBeNull(); // out of sight
    await userEvent.click(screen.getByRole('button', { name: '1 more with no good guess' }));
    expect(screen.getByRole('listitem', { name: /^Face 9:/ })).toBeVisible();
  });
});
