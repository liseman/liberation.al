import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { PAGE_SLOW_MS } from '../lib/page';
import { navigate } from '../router';
import type { PeopleView, Person } from './api';
import { useClones } from './clones/store';
import { PeopleTab } from './PeopleTab';
import classes from './people.module.css';

const person = (name: string, extra: Partial<Person> = {}): Person => ({ name, voices: [], ...extra });

const VIEW: PeopleView = {
  people: [
    person('Alice', { owner: true, voices: [1], lines: 12, last_heard: 100 }),
    person('Dana', { voices: [3], role: 'CEO', organization: 'Acme', lines: 2, last_heard: 50 }),
    person('multiple speakers', { lines: 4 }),
  ],
  unknown: {
    voices: [],
    faces: [{ id: 5, label: 'Face 5', photo: '/faces/unknown/5.jpg', count: 3 }],
    voice_clips: [{ id: 2, label: 'Speaker 2', ts: 300.5, text: 'hello', dur: 2 }],
    voice_clip_total: 1,
  },
};

/** The servers the whole tab reads, answering with nothing much but VIEW. */
function serve(view: PeopleView = VIEW, faces = 3, voices = 2) {
  const posts: { url: string; body: unknown }[] = [];
  server.use(
    http.get('/people', () => HttpResponse.json(view)),
    http.get('/faces/unknown/groups', () => HttpResponse.json({ groups: [], count: faces, skipped: [] })),
    http.get('/voices/unknown/groups', () => HttpResponse.json({ groups: [], count: voices, skipped: [] })),
    http.get('/inbox/unnamed', () => HttpResponse.json({ faces: [], videos: [], voices: [] })),
    http.get('/speakers/refine', () => HttpResponse.json({ items: [], remaining: 0 })),
    http.get('/calls', () => HttpResponse.json([])),
    http.get('/clones', () =>
      HttpResponse.json({
        clones: [
          { id: 1, label: 'Alice', ready: false, built: false },
          { id: 3, label: 'Dana', ready: true, built: false, clean_s: 14 },
        ],
      }),
    ),
    http.get('/people/:name', ({ params }) =>
      HttpResponse.json({ name: params.name, voices: [], lines: 0, recent: [], memory: [] }),
    ),
    http.get('/people/:name/card', ({ params }) =>
      HttpResponse.json({
        name: params.name,
        photo: null,
        shots: [],
        videos: [],
        voices: [],
        last_seen: null,
        here_now: null,
      }),
    ),
    http.get('/speakers', () => HttpResponse.json({ speakers: [], merged: {} })),
    http.post('/people/add', async ({ request }) => {
      posts.push({ url: '/people/add', body: await request.json() });
      return HttpResponse.json({ name: 'Erin' });
    }),
    http.post('/clones/build', async ({ request }) => {
      posts.push({ url: '/clones/build', body: await request.json() });
      return HttpResponse.json({ ok: true, utterances: 4, seconds: 14 });
    }),
  );
  return posts;
}

beforeEach(() => {
  server.use(http.get('/pets', () => HttpResponse.json({ pets: [], species: ['cat', 'dog', 'bird'] })));
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Alice', cams: ['cam0'], hotkeys: false });
  useClones.setState({ sel: null, busy: null });
});
afterEach(() => {
  vi.unstubAllGlobals();
  history.replaceState(null, '', '/'); // a card picked writes #card= to the address
});

const roster = () => within(screen.getByRole('region', { name: 'People' }));

function requiredClass(name: string | undefined): string {
  if (name == null) throw new Error('missing people layout class');
  return name;
}

const PAGE = requiredClass(classes.page);
const PICKED = requiredClass(classes.picked);

/** The People page element whose `picked` class hides the roster at 820px and under. */
function peoplePage(container: HTMLElement): HTMLElement {
  const node = container.querySelector(`.${CSS.escape(PAGE)}`);
  if (!(node instanceof HTMLElement)) throw new Error('missing people page');
  return node;
}

/** Unix seconds for a local time yesterday. */
const yesterday = (h: number, m: number) => {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  d.setHours(h, m, 0, 0);
  return d.getTime() / 1000;
};
const NOTHING_WAITS: PeopleView['unknown'] = { voices: [], faces: [], voice_clips: [], voice_clip_total: 0 };

describe('the list', () => {
  it('shares one roster request with the naming suggestions on first load', async () => {
    serve();
    let reads = 0;
    server.use(
      http.get('/people', () => {
        reads++;
        return HttpResponse.json(VIEW);
      }),
    );
    renderPage(<PeopleTab />);
    await roster().findByRole('button', { name: /^Dana/ });
    expect(reads).toBe(1);
  });

  it('lists everyone but the several-speakers label, with a count and what is known of them', async () => {
    serve();
    renderPage(<PeopleTab />);
    expect(await roster().findByRole('button', { name: /^Dana/ })).toHaveTextContent('CEO at Acme');
    expect(roster().getByRole('button', { name: /^Alice/ })).toHaveTextContent('You');
    expect(roster().queryByRole('button', { name: /multiple speakers/i })).not.toBeInTheDocument();
    expect(roster().getByRole('heading', { name: 'People' }).nextElementSibling).toHaveTextContent('2');
  });

  it('leaves the voice clones list to Settings: People is the roster and the detail beside it', async () => {
    serve();
    renderPage(<PeopleTab />);
    await roster().findByRole('button', { name: /^Dana/ });
    expect(screen.queryByRole('radiogroup', { name: 'Voice to clone' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Record a new voice' })).toBeNull();
  });

  it('leaves Calls to its own tab instead of rendering the call form under People', async () => {
    serve();
    renderPage(<PeopleTab />);
    await roster().findByRole('button', { name: /^Dana/ });
    expect(screen.queryByRole('heading', { name: 'Calls' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Call' })).not.toBeInTheDocument();
  });

  it('puts who is here now first, with a dot and the camera, and everyone else with when they were last seen', async () => {
    serve({
      ...VIEW,
      people: [
        person('Alice', { owner: true, voices: [1], here_now: { cam: 'cam0' }, last_seen: yesterday(9, 0) }),
        person('Dana', { role: 'CEO', organization: 'Acme', last_seen: yesterday(20, 12), last_seen_cam: 'cam1' }),
        person('Eli', { voices: [4], lines: 2 }),
      ],
    });
    renderPage(<PeopleTab />);
    const here = within(await roster().findByRole('list', { name: 'Here now' }));
    const [alice] = here.getAllByRole('button');
    expect(here.getAllByRole('button')).toHaveLength(1);
    expect(alice).toHaveAccessibleName(/^Alice/);
    expect(within(alice as HTMLElement).getByText('Here now, seen by cam0')).toBeInTheDocument();
    const everyone = within(roster().getByRole('list', { name: 'Everyone' }));
    const [dana, eli] = everyone.getAllByRole('button') as HTMLElement[];
    expect(everyone.getAllByRole('button')).toHaveLength(2);
    expect(
      within(dana as HTMLElement).getByText('CEO at Acme, Last seen yesterday at 20:12 by cam1'),
    ).toBeInTheDocument();
    expect(within(eli as HTMLElement).getByText('2 lines heard')).toBeInTheDocument();
  });

  it('leaves out the Here now heading when nobody is here', async () => {
    serve();
    renderPage(<PeopleTab />);
    await roster().findByRole('list', { name: 'Everyone' });
    expect(roster().queryByRole('list', { name: 'Here now' })).not.toBeInTheDocument();
  });

  it('searches by name, role, or organization, and says when nobody matches', async () => {
    serve();
    renderPage(<PeopleTab />);
    const search = await roster().findByRole('textbox', { name: 'Search people' });
    await roster().findByRole('button', { name: /^Dana/ });
    await userEvent.type(search, 'acme');
    expect(roster().getByRole('button', { name: /^Dana/ })).toBeInTheDocument();
    expect(roster().queryByRole('button', { name: /^Alice/ })).not.toBeInTheDocument();
    await userEvent.clear(search);
    await userEvent.type(search, 'ALI');
    expect(roster().getByRole('button', { name: /^Alice/ })).toBeInTheDocument();
    expect(roster().queryByRole('button', { name: /^Dana/ })).not.toBeInTheDocument();
    await userEvent.clear(search);
    await userEvent.type(search, 'zed');
    expect(roster().getByText('Nobody matches “zed”')).toBeInTheDocument();
  });

  it('says when nobody is known yet', async () => {
    serve({ people: [], unknown: NOTHING_WAITS });
    renderPage(<PeopleTab />);
    expect(await roster().findByText('No one yet')).toBeInTheDocument();
  });

  it('says nobody is known yet, not that nobody matches, when the list is empty and a search is typed', async () => {
    serve({ people: [], unknown: NOTHING_WAITS });
    renderPage(<PeopleTab />);
    await roster().findByText('No one yet');
    await userEvent.type(roster().getByRole('textbox', { name: 'Search people' }), 'zed');
    expect(roster().getByText('No one yet')).toBeInTheDocument();
    expect(roster().queryByText('Nobody matches “zed”')).not.toBeInTheDocument();
  });

  it("opens a person's card on one click anywhere on the row", async () => {
    serve();
    renderPage(<PeopleTab />);
    const dana = await roster().findByRole('button', { name: /^Dana/ });
    await userEvent.click(within(dana).getByText(/CEO at Acme/));
    expect(await screen.findByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
    expect(dana).toHaveAttribute('aria-current', 'true');
  });

  it("opens a person's card when their name is clicked", async () => {
    serve();
    renderPage(<PeopleTab />);
    const dana = await roster().findByRole('button', { name: /^Dana/ });
    await userEvent.click(within(dana).getByText('Dana'));
    expect(await screen.findByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
    expect(dana).toHaveAttribute('aria-current', 'true');
  });

  it("opens the focused person's card with I", async () => {
    serve();
    renderPage(<PeopleTab />);
    const dana = await roster().findByRole('button', { name: /^Dana/ });
    dana.focus();
    await userEvent.keyboard('i');
    expect(await screen.findByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
  });
});

describe("what waits in Who's this?", () => {
  // The lists are read once; later reads wait until the test opens the gate, as the server's face groups can take
  // half a minute after a name. The roster's count must drop on the server's OK alone. The fake server drops
  // what was named or put aside, as the real one does.
  const waiting = (id: number, label: string) => ({
    id,
    label,
    count: 4,
    last_seen: 100,
    guess: null,
    score: 0,
    band: 'none',
    reason: 'Unlike anyone named',
  });
  function serveSlowLists() {
    const posts = serve();
    let open = () => {};
    const gate = new Promise<void>(resolve => {
      open = resolve;
    });
    let faces = [{ ...waiting(5, 'Face 5'), shots: [{ id: 1, ts: 100, url: '/faces/shot/1.jpg' }] }];
    let voices = [
      { ...waiting(7, 'Speaker 7'), clips: [{ ts: 100, line: 'l1', text: 'hello', dur: 2, url: '/a.wav' }] },
    ];
    const drop = ({ kind, id }: { kind: string; id: number }) => {
      if (kind === 'face') faces = faces.filter(g => g.id !== id);
      else voices = voices.filter(g => g.id !== id);
    };
    let reads = 0;
    const list = (body: () => object) => async () => {
      if (reads++ >= 2) await gate;
      return HttpResponse.json(body());
    };
    server.use(
      http.get(
        '/faces/unknown/groups',
        list(() => ({ groups: faces, count: faces.length + 2, skipped: [] })),
      ),
      http.get(
        '/voices/unknown/groups',
        list(() => ({ groups: voices, count: voices.length + 1, skipped: [] })),
      ),
      http.post('/people/claim', async ({ request }) => {
        const body = (await request.json()) as { kind: string; id: number };
        posts.push({ url: '/people/claim', body });
        drop(body);
        return HttpResponse.json({ ok: true, name: 'Dana' });
      }),
      http.post('/inbox/skip', async ({ request }) => {
        const body = (await request.json()) as { kind: string; id: number };
        posts.push({ url: '/inbox/skip', body });
        drop(body);
        return HttpResponse.json({ ok: true, skipped: 1 });
      }),
      // Ignore on a face's card: its pictures are hidden, and with none left the group goes.
      http.post('/faces/unknown/hide', async ({ request }) => {
        const body = (await request.json()) as { group: number; action: string };
        posts.push({ url: '/faces/unknown/hide', body });
        drop({ kind: 'face', id: body.group });
        return HttpResponse.json({ ok: true, kind: 'ignored', keys: ['1'], hidden: 1, left: 0 });
      }),
      http.get('/transcript', () => {
        tails++;
        return HttpResponse.json({ entries: [] });
      }),
    );
    let tails = 0;
    /** Let the reads through, and wait for the naming's own follow-up read so it can't land in the next test. */
    const finish = async () => {
      open();
      await waitFor(() => expect(tails).toBeGreaterThan(0));
    };
    return { posts, finish };
  }
  const answer = async (row: string, how: 'name' | 'ignore') => {
    // These have no good guess, so they wait folded under the list until shown (Known design pass).
    await userEvent.click(await screen.findByRole('button', { name: /more with no good guess$/ }));
    const item = await screen.findByRole('listitem', { name: new RegExp(`^${row}:`) });
    if (how === 'ignore') {
      await userEvent.click(within(item).getByRole('button', { name: /^Ignore/ }));
    } else await userEvent.type(within(item).getByRole('combobox', { name: "This person's name" }), 'Dana{Enter}');
  };

  it.each([
    ['naming a face', 'Face 5', 'name', '2 faces and 2 voices need a name'],
    ['ignoring a face', 'Face 5', 'ignore', '2 faces and 2 voices need a name'],
    ['naming a voice', 'Speaker 7', 'name', '3 faces and 1 voice need a name'],
    ['ignoring a voice', 'Speaker 7', 'ignore', '3 faces and 1 voice need a name'],
  ] as const)('lowers the count on %s at once, before the lists are read again', async (_, row, how, after) => {
    const { posts, finish } = serveSlowLists();
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: '3 faces and 2 voices need a name' }));
    await answer(row, how);
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(await roster().findByRole('button', { name: after })).toBeInTheDocument(); // the reads still wait
    await finish();
    expect(await roster().findByRole('button', { name: after })).toBeInTheDocument(); // and agree once they're in
  });

  it('keeps the count when the server refuses the name', async () => {
    const { posts, finish } = serveSlowLists();
    server.use(
      http.post('/people/claim', async ({ request }) => {
        posts.push({ url: '/people/claim', body: await request.json() });
        return HttpResponse.json({ ok: false, error: 'That face group has changed.' }, { status: 409 });
      }),
    );
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: '3 faces and 2 voices need a name' }));
    await answer('Face 5', 'name');
    expect(await screen.findByText('That face group has changed.')).toBeInTheDocument();
    expect(roster().getByRole('button', { name: '3 faces and 2 voices need a name' })).toBeInTheDocument();
    await finish();
  });

  it.each([
    ['faces', '2 voices need a name · Faces unavailable'],
    ['voices', '3 faces need a name · Voices unavailable'],
  ])('preserves the available count when %s fail', async (kind, label) => {
    serve();
    server.use(
      http.get(`/${kind}/unknown/groups`, () => HttpResponse.json({ error: 'source unavailable' }, { status: 503 })),
    );
    renderPage(<PeopleTab />);
    expect(await roster().findByRole('button', { name: label })).toBeInTheDocument();
  });

  it('keeps pending counts unknown until a completed group answer arrives', async () => {
    serve(VIEW, 0, 2);
    let pending = true;
    server.use(
      http.get('/faces/unknown/groups', () =>
        HttpResponse.json(
          pending ? { pending: true, groups: [], count: null, skipped: [] } : { groups: [], count: 0, skipped: [] },
          { status: pending ? 202 : 200 },
        ),
      ),
    );
    const { client } = renderPage(<PeopleTab />);
    // A count still being worked out holds the line as a skeleton, not "2 voices" and then a jump to the total.
    expect(await roster().findByRole('button', { name: "Who's this? · Loading the count" })).toBeInTheDocument();
    expect(roster().queryByRole('button', { name: '2 voices need a name' })).not.toBeInTheDocument();
    pending = false;
    await act(async () => {
      await client.invalidateQueries({ queryKey: ['faces', 'unknown', 'groups'] });
    });
    expect(await roster().findByRole('button', { name: '2 voices need a name' })).toBeInTheDocument();
  });

  it("shows Who's this? inside People, not on Live, with how many faces and voices wait", async () => {
    serve();
    renderPage(<PeopleTab />);
    const own = await roster().findByRole('button', { name: '3 faces and 2 voices need a name' });
    expect(own).toHaveAttribute('aria-current', 'true'); // shown until someone is picked
    await userEvent.click(own);
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    expect(location.pathname).toBe('/'); // no jump to Live
    expect(location.hash).toBe('');
  });

  it("opens Who's this? from Live's link, even over a card that was showing, and tidies the address", async () => {
    serve();
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: /^Dana/ }));
    await screen.findByRole('article', { name: 'Profile of Dana' });
    navigate('/people#whos-this'); // People stays mounted between tabs
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    expect(screen.queryByRole('article')).not.toBeInTheDocument();
    expect(location.hash).toBe('');
  });

  it("leaves Live's own #whos-this alone", async () => {
    serve();
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: /^Dana/ }));
    await screen.findByRole('article', { name: 'Profile of Dana' });
    navigate('/#whos-this'); // Live's rail section, with People still mounted behind it
    expect(location.hash).toBe('#whos-this');
    expect(screen.getByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
    history.replaceState(null, '', '/');
  });

  it('counts one in the singular, and says just "Who\'s this?" when nothing waits', async () => {
    serve(VIEW, 1, 0);
    const { unmount } = renderPage(<PeopleTab />);
    expect(await roster().findByRole('button', { name: '1 face needs a name' })).toBeInTheDocument();
    unmount();
    serve(VIEW, 0, 0);
    renderPage(<PeopleTab />);
    expect(await roster().findByRole('button', { name: "Who's this?" })).toBeInTheDocument();
  });

  it("goes back to Who's this? from a card", async () => {
    serve();
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: /^Dana/ }));
    await screen.findByRole('article', { name: 'Profile of Dana' });
    await userEvent.click(roster().getByRole('button', { name: '3 faces and 2 voices need a name' }));
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
  });

  it("keeps Who's this? one click away, and shows it by default, when nothing waits", async () => {
    serve({ ...VIEW, unknown: NOTHING_WAITS }, 0, 0);
    renderPage(<PeopleTab />);
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    await userEvent.click(await roster().findByRole('button', { name: /^Alice/ }));
    await screen.findByRole('article', { name: 'Profile of Alice' });
    await userEvent.click(roster().getByRole('button', { name: "Who's this?" }));
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
  });

  it("keeps showing Who's this? when the last face is named, rather than switching to a card", async () => {
    let view = VIEW;
    serve();
    server.use(http.get('/people', () => HttpResponse.json(view)));
    const { client } = renderPage(<PeopleTab />);
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    view = { ...VIEW, unknown: NOTHING_WAITS };
    await client.refetchQueries({ queryKey: ['people'], exact: true });
    expect(screen.getByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    expect(screen.queryByRole('article')).not.toBeInTheDocument();
  });
});

describe('deleting someone', () => {
  it("says what was forgotten, even when nothing waits in Who's this?", async () => {
    serve({ ...VIEW, unknown: NOTHING_WAITS });
    server.use(
      http.post('/people/delete', () =>
        HttpResponse.json({ ok: true, name: 'Dana', removed: { voices: 1 }, people: VIEW.people }),
      ),
    );
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: /^Dana/ }));
    await screen.findByRole('article', { name: 'Profile of Dana' });
    await userEvent.click(screen.getByRole('button', { name: 'More for Dana' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }));
    await userEvent.click(
      within(screen.getByRole('dialog', { name: 'Delete Dana' })).getByRole('button', { name: 'Delete' }),
    );
    expect(await screen.findByText('Forgot Dana: 1 voice.')).toBeInTheDocument();
  });
});

describe('New person', () => {
  it("opens the new person's card on picking what's theirs, and only theirs", async () => {
    serve();
    let added = false;
    server.use(
      http.get('/people', () =>
        HttpResponse.json(added ? { ...VIEW, people: [...VIEW.people, person('Erin')] } : VIEW),
      ),
      http.post('/people/add', () => {
        added = true;
        return HttpResponse.json({ name: 'Erin' });
      }),
    );
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: 'New person' }));
    const dialog = await screen.findByRole('dialog', { name: 'New person' });
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name' }), 'Erin');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Add Erin' }));
    expect(
      await screen.findByText("New. Pick the photos, video, and voice that are Erin's, and the card fills in."),
    ).toBeInTheDocument();
    await userEvent.click(await roster().findByRole('button', { name: /^Dana/ }));
    expect(await screen.findByRole('region', { name: 'Photos' })).toBeInTheDocument();
    expect(screen.queryByText(/^New\. Pick/)).not.toBeInTheDocument();
    // Back to Erin later: no longer new, so the plain card.
    await userEvent.click(roster().getByRole('button', { name: /^Erin/ }));
    expect(await screen.findByRole('article', { name: 'Profile of Erin' })).toBeInTheDocument();
    expect(await screen.findByRole('region', { name: 'Photos' })).toBeInTheDocument();
    expect(screen.queryByText(/Pick the photos/)).not.toBeInTheDocument();
  });

  it('adds a person by name and opens their card', async () => {
    const posts = serve();
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: 'New person' }));
    const dialog = await screen.findByRole('dialog', { name: 'New person' });
    expect(within(dialog).getByText(/Add manually at any time/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Add' })).toBeDisabled();
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name' }), ' Erin ');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Add Erin' }));
    expect(await screen.findByRole('article', { name: 'Profile of Erin' })).toBeInTheDocument();
    expect(posts).toContainEqual({ url: '/people/add', body: { name: 'Erin', number: '' } });
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'New person' })).not.toBeInTheDocument());
  });

  it('closes on Cancel without adding anyone, and opens empty next time', async () => {
    const posts = serve();
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: 'New person' }));
    const dialog = await screen.findByRole('dialog', { name: 'New person' });
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name' }), 'Erin');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'New person' })).not.toBeInTheDocument());
    expect(posts).toEqual([]);
    await userEvent.click(roster().getByRole('button', { name: 'New person' }));
    expect(
      within(await screen.findByRole('dialog', { name: 'New person' })).getByRole('textbox', { name: 'Name' }),
    ).toHaveValue('');
  });

  it("shows the server's refusal and stays open", async () => {
    serve();
    server.use(
      http.post('/people/add', () =>
        HttpResponse.json({ error: "Names start with a letter and use letters, spaces, . ' or -" }, { status: 400 }),
      ),
    );
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: 'New person' }));
    const dialog = await screen.findByRole('dialog', { name: 'New person' });
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name' }), '9lives');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Add 9lives' }));
    expect(await within(dialog).findByRole('alert')).toHaveTextContent(
      "Names start with a letter and use letters, spaces, . ' or -",
    );
    expect(screen.queryByRole('article', { name: 'Profile of 9lives' })).not.toBeInTheDocument();
  });
});

describe('loading and errors', () => {
  it('shows placeholder rows, then a loader once the wait runs long, then the list', async () => {
    let answer: () => void = () => {};
    const ready = new Promise<void>(r => {
      answer = r;
    });
    serve();
    server.use(
      http.get('/people', async () => {
        await ready;
        return HttpResponse.json(VIEW);
      }),
    );
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    renderPage(<PeopleTab />);
    expect(roster().getByLabelText('Loading people')).toHaveAttribute('aria-busy', 'true');
    expect(roster().getByRole('status')).toBeEmptyDOMElement(); // a quick load says nothing
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAGE_SLOW_MS);
    });
    expect(roster().getByRole('status')).toHaveTextContent('Still loading people.');
    vi.useRealTimers(); // before the answer, so the reads that follow it run on real time
    answer();
    expect(await roster().findByRole('button', { name: /^Dana/ })).toBeInTheDocument();
    expect(roster().queryByLabelText('Loading people')).not.toBeInTheDocument();
  });

  it('does not ask who is waiting until the roster is on screen', async () => {
    let answer: () => void = () => {};
    const ready = new Promise<void>(resolve => {
      answer = resolve;
    });
    const asked: string[] = [];
    serve();
    server.use(
      http.get('/people', async () => {
        await ready;
        return HttpResponse.json(VIEW);
      }),
      http.get('/faces/unknown/groups', () => {
        asked.push('faces');
        return HttpResponse.json({ groups: [], count: 0, skipped: [] });
      }),
      http.get('/voices/unknown/groups', () => {
        asked.push('voices');
        return HttpResponse.json({ groups: [], count: 0, skipped: [] });
      }),
    );
    renderPage(<PeopleTab />);
    expect(roster().getByLabelText('Loading people')).toBeInTheDocument();
    await act(async () => {
      await Promise.resolve();
    });
    expect(asked).toEqual([]);
    expect(roster().getByRole('button', { name: "Who's this? · Loading the count" })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: "Who's this?" })).not.toBeInTheDocument();
    answer();
    expect(await roster().findByRole('button', { name: /^Dana/ })).toBeInTheDocument();
    await waitFor(() => expect(asked).toEqual(['faces', 'voices']));
  });

  it('shows the error with a retry, never an endless placeholder, and loads on retry', async () => {
    const asked: string[] = [];
    serve();
    let fail = true;
    server.use(
      http.get('/people', () =>
        fail ? HttpResponse.json({ error: 'the registry is busy' }, { status: 503 }) : HttpResponse.json(VIEW),
      ),
      http.get('/faces/unknown/groups', () => {
        asked.push('faces');
        return HttpResponse.json({ groups: [], count: 0, skipped: [] });
      }),
      http.get('/voices/unknown/groups', () => {
        asked.push('voices');
        return HttpResponse.json({ groups: [], count: 0, skipped: [] });
      }),
    );
    renderPage(<PeopleTab />);
    const alert = await roster().findByRole('alert');
    expect(alert).toHaveTextContent("Couldn't load people");
    expect(alert).toHaveTextContent('the registry is busy');
    expect(roster().getByRole('button', { name: "Who's this? · People unavailable" })).toBeInTheDocument();
    expect(roster().queryByLabelText('Loading people')).not.toBeInTheDocument();
    expect(asked).toEqual([]);
    fail = false;
    await userEvent.click(within(alert).getByRole('button', { name: 'Retry' }));
    expect(await roster().findByRole('button', { name: /^Dana/ })).toBeInTheDocument();
  });
});

describe("Who's this before the roster view exists", () => {
  it('keeps a #whos-this link on the roster loader until people arrive, then opens it', async () => {
    let answer: () => void = () => {};
    const ready = new Promise<void>(resolve => {
      answer = resolve;
    });
    serve();
    server.use(
      http.get('/people', async () => {
        await ready;
        return HttpResponse.json(VIEW);
      }),
    );
    history.replaceState(null, '', '/people#whos-this');
    const { container } = renderPage(<PeopleTab />);
    const page = peoplePage(container);
    expect(roster().getByLabelText('Loading people')).toBeInTheDocument();
    expect(page).not.toHaveClass(PICKED);
    expect(screen.queryByRole('heading', { name: "Who's this?" })).not.toBeInTheDocument();
    answer();
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    expect(page).toHaveClass(PICKED);
  });

  it("keeps Retry on the roster when that link's load fails, and opens Who's this after retry", async () => {
    serve();
    let fail = true;
    server.use(
      http.get('/people', () =>
        fail ? HttpResponse.json({ error: 'the registry is busy' }, { status: 503 }) : HttpResponse.json(VIEW),
      ),
    );
    history.replaceState(null, '', '/people#whos-this');
    const { container } = renderPage(<PeopleTab />);
    const page = peoplePage(container);
    const alert = await roster().findByRole('alert');
    expect(alert).toHaveTextContent("Couldn't load people");
    expect(alert).toHaveTextContent('the registry is busy');
    expect(page).not.toHaveClass(PICKED);
    expect(screen.queryByRole('heading', { name: "Who's this?" })).not.toBeInTheDocument();
    fail = false;
    await userEvent.click(within(alert).getByRole('button', { name: 'Retry' }));
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    expect(page).toHaveClass(PICKED);
  });

  it("keeps the loader on the roster when Who's this is tapped while people are loading", async () => {
    let answer: () => void = () => {};
    const ready = new Promise<void>(resolve => {
      answer = resolve;
    });
    serve();
    server.use(
      http.get('/people', async () => {
        await ready;
        return HttpResponse.json(VIEW);
      }),
    );
    history.replaceState(null, '', '/people');
    const { container } = renderPage(<PeopleTab />);
    const page = peoplePage(container);
    await userEvent.click(roster().getByRole('button', { name: "Who's this? · Loading the count" }));
    expect(roster().getByLabelText('Loading people')).toBeInTheDocument();
    expect(page).not.toHaveClass(PICKED);
    expect(screen.queryByRole('heading', { name: "Who's this?" })).not.toBeInTheDocument();
    answer();
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    expect(page).toHaveClass(PICKED);
  });
});

describe('a link to a card (/people#card=<name>)', () => {
  it('leaves Live’s full address unchanged when a refresh resolves a retained People link', async () => {
    serve();
    history.replaceState(null, '', '/people#card=Erin');
    const { client } = renderPage(<PeopleTab />);
    await screen.findByRole('heading', { name: "Who's this?" });
    await waitFor(() => expect(location.hash).toBe(''));
    act(() => navigate('/?camera=cam0#whos-this'));
    server.use(http.get('/people', () => HttpResponse.json({ ...VIEW, people: [...VIEW.people, person('Erin')] })));
    await act(() => client.refetchQueries({ queryKey: ['people'], exact: true }));
    await screen.findByRole('article', { name: 'Profile of Erin' });
    expect(location.pathname + location.search + location.hash).toBe('/?camera=cam0#whos-this');
  });

  it('does not follow a card hash belonging to Live while People remains mounted', async () => {
    serve();
    history.replaceState(null, '', '/people#card=Dana');
    renderPage(<PeopleTab />);
    await screen.findByRole('article', { name: 'Profile of Dana' });
    act(() => navigate('/?camera=cam0#card=Alice'));
    expect(screen.getByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
    expect(location.pathname + location.search + location.hash).toBe('/?camera=cam0#card=Alice');
  });

  it('opens the card it names when People loads, matching the name whatever its case', async () => {
    serve();
    history.replaceState(null, '', '/people#card=dana');
    renderPage(<PeopleTab />);
    expect(await screen.findByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
    await waitFor(() => expect(location.hash).toBe('#card=Dana'));
  });

  it('follows a link while People is already showing (it stays mounted between tabs)', async () => {
    serve();
    history.replaceState(null, '', '/people');
    renderPage(<PeopleTab />);
    await screen.findByRole('heading', { name: "Who's this?" });
    act(() => {
      history.pushState(null, '', '/people#card=Alice');
      window.dispatchEvent(new HashChangeEvent('hashchange'));
    });
    expect(await screen.findByRole('article', { name: 'Profile of Alice' })).toBeInTheDocument();
  });

  it("keeps the address in step with the card picked, and clears it for Who's this?", async () => {
    serve();
    history.replaceState(null, '', '/people');
    const before = history.length;
    renderPage(<PeopleTab />);
    await userEvent.click(await roster().findByRole('button', { name: /^Dana/ }));
    await waitFor(() => expect(location.hash).toBe('#card=Dana'));
    await userEvent.click(roster().getByRole('button', { name: /^Alice/ }));
    await waitFor(() => expect(location.hash).toBe('#card=Alice'));
    expect(history.length).toBe(before); // replaced, not pushed: Back still leaves People
    await userEvent.click(roster().getByRole('button', { name: /need a name|Who's this\?/ }));
    await waitFor(() => expect(location.hash).toBe(''));
  });

  it("shows Who's this? for a name nobody has, and clears the address", async () => {
    serve();
    history.replaceState(null, '', `/people#card=${encodeURIComponent('Nobody Here')}`);
    renderPage(<PeopleTab />);
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    expect(screen.queryByRole('article')).not.toBeInTheDocument();
    await waitFor(() => expect(location.hash).toBe(''));
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

describe('on a phone', () => {
  afterEach(() => vi.restoreAllMocks());
  const picked = () => document.querySelector(`.${classes.picked}`);

  it("opens on Who's this?, with the list one tap back", async () => {
    asPhone();
    serve();
    renderPage(<PeopleTab />);
    expect(await screen.findByRole('heading', { name: "Who's this?" })).toBeInTheDocument();
    await waitFor(() => expect(picked()).not.toBeNull()); // the roster is hidden behind it
    await userEvent.click(screen.getByRole('button', { name: 'Back to people' }));
    expect(picked()).toBeNull();
  });

  it('opens a linked card rather than the queue', async () => {
    asPhone();
    location.hash = '#card=Dana';
    serve();
    renderPage(<PeopleTab />);
    expect(await screen.findByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
  });

  it("opens a person's card on one tap", async () => {
    asPhone();
    serve();
    renderPage(<PeopleTab />);
    await userEvent.click(await screen.findByRole('button', { name: 'Back to people' }));
    await userEvent.click(await roster().findByRole('button', { name: /^Dana/ }));
    expect(await screen.findByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
    expect(picked()).not.toBeNull();
  });

  it('keeps the list beside the queue on a wide window', async () => {
    serve();
    renderPage(<PeopleTab />);
    await screen.findByRole('heading', { name: "Who's this?" });
    expect(picked()).toBeNull();
  });
});
