import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http, type JsonBodyType } from 'msw';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { TIMEOUT_MS } from '../../../shared/http';
import { newQueryClient, Providers } from '../../../shared/Providers';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { isoDay } from '../../lib/format';
import { SLOW_MS } from '../../lib/loading';
import { stop, usePlayer } from '../../lib/player';
import { type Card, type Clone, cardKey, type Person, type Profile as ProfileData, profileKey } from '../api';
import { Profile } from './Profile';

vi.mock('../../review/gatewaySegments', async original => ({
  ...(await original<typeof import('../../review/gatewaySegments')>()),
  daySegments: vi.fn(async () => []),
}));

const DANA: ProfileData = { name: 'Dana', voices: [3], lines: 0, role: 'CEO', organization: 'Acme' };
const PEOPLE: Person[] = [
  { name: 'Dana', voices: [3], role: 'CEO', organization: 'Acme' },
  { name: 'Eli', voices: [4] },
];
const EMPTY: Card = { name: 'Dana', photo: null, shots: [], videos: [], voices: [], last_seen: null, here_now: null };

/** Unix seconds for a local time today. */
const today = (h: number, m: number) => {
  const d = new Date();
  d.setHours(h, m, 0, 0);
  return d.getTime() / 1000;
};

/** The reads a card makes, with what each one answers. */
function serve({
  profile = DANA,
  card = EMPTY,
  clones = [],
}: {
  profile?: ProfileData;
  card?: Card;
  clones?: Clone[];
} = {}) {
  server.use(
    http.get('/people/:name', () => HttpResponse.json(profile)),
    http.get('/people/:name/card', () => HttpResponse.json(card)),
    http.get('/clones', () => HttpResponse.json({ clones })),
  );
}

/** Record each JSON body posted to path, and answer with reply. */
function posts(path: string, reply: JsonBodyType) {
  const bodies: unknown[] = [];
  server.use(
    http.post(path, async ({ request }) => {
      bodies.push(await request.json());
      return HttpResponse.json(reply);
    }),
  );
  return bodies;
}

function show(over: { name?: string; person?: Person; people?: Person[] } = {}) {
  const people = over.people ?? PEOPLE;
  const name = over.name ?? 'Dana';
  const props = {
    name,
    person: over.person ?? people.find(p => p.name === name),
    people,
    onChanged: vi.fn(),
    onDeleted: vi.fn(),
    onBack: vi.fn(),
  };
  const { client } = renderPage(<Profile {...props} />);
  return { ...props, client };
}

const profile = (name = 'Dana') => screen.findByRole('article', { name: `Profile of ${name}` });
const button = (name: string | RegExp) => screen.getByRole('button', { name });
const region = (name: string) => screen.findByRole('region', { name });

it('polls pending card crops until ready photos replace the processing line', async () => {
  let card: Card = { ...EMPTY, shots_pending: 2 };
  serve();
  server.use(http.get('/people/:name/card', () => HttpResponse.json(card)));
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
  try {
    show();
    const photos = await region('Photos');
    expect(await within(photos).findByText('Processing 2 detections in the background…')).toBeVisible();
    expect(photos.querySelector('img')).toBeNull();
    card = {
      ...EMPTY,
      shots_pending: 0,
      shots_unavailable: 1,
      shots: [{ id: 'ready', url: '/faces/shot/ready.jpg', ts: 100, cam: 'cam0', score: 0.9 }],
    };
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await waitFor(() => expect(within(photos).queryByText(/Processing .* detections/)).toBeNull());
    expect(photos.querySelector('img')).toHaveAttribute('src', '/faces/shot/ready.jpg');
    expect(within(photos).getByText("1 detection can't be shown.")).toBeVisible();
  } finally {
    vi.useRealTimers();
  }
});

it('asks again for a card the server answered from an older read, until the fresh one comes', async () => {
  const old: Card = { ...EMPTY, refreshing: true };
  const fresh: Card = {
    ...EMPTY,
    shots: [{ id: 'new', url: '/faces/shot/new.jpg', ts: 100, cam: 'cam0', score: 0.9 }],
  };
  let card = old;
  let reads = 0;
  serve();
  server.use(
    http.get('/people/:name/card', () => {
      reads += 1;
      return HttpResponse.json(card);
    }),
  );
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
  try {
    show();
    const photos = await region('Photos');
    await waitFor(() => expect(reads).toBe(1));
    card = fresh;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await waitFor(() => expect(photos.querySelector('img')).toHaveAttribute('src', '/faces/shot/new.jpg'));
    const settled = reads;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(reads).toBe(settled); // the fresh card isn't marked: no more polling
  } finally {
    vi.useRealTimers();
  }
});

/** Exercise the real request deadline and production retry policy without a real 20-second wait. */
function waitingCard(cached?: Card) {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  const client = newQueryClient();
  if (cached) client.setQueryData(cardKey('Dana'), cached, { updatedAt: Date.now() - 60_000 });
  let waiting = true;
  const fetcher = vi.fn<typeof fetch>((url, init) => {
    if (url === '/people/Dana') return Promise.resolve(Response.json(DANA));
    if (url === '/clones') return Promise.resolve(Response.json({ clones: [] }));
    if (url === '/people/references?person=Dana') return Promise.resolve(Response.json({ references: [] }));
    if (url === '/people/Dana/card') {
      if (!waiting) return Promise.resolve(Response.json(EMPTY));
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
      <Profile
        name="Dana"
        person={PEOPLE[0]}
        people={PEOPLE}
        onChanged={vi.fn()}
        onDeleted={vi.fn()}
        onBack={vi.fn()}
      />
    </Providers>,
  );
  return {
    fetcher,
    recover: () => {
      waiting = false;
    },
  };
}
/** Open the card's More menu and pick an item. */
async function menu(item: string, who = 'Dana') {
  await userEvent.click(button(`More for ${who}`));
  await userEvent.click(await screen.findByRole('menuitem', { name: item }));
}
const menuItems = async (who = 'Dana') => {
  await userEvent.click(button(`More for ${who}`));
  return (await screen.findAllByRole('menuitem')).map(i => i.textContent);
};

beforeEach(() => {
  server.use(http.get('/ambient', () => HttpResponse.json({ devices: {} })));
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
});
afterEach(() => {
  stop();
  history.replaceState(null, '', '/');
});

describe('loading a card', () => {
  it('shows slow media loading and the first deadline, then recovers on Retry', async () => {
    const { fetcher, recover } = waitingCard();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
    expect(screen.getByText("Loading Dana's photos, video, and voice…")).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(SLOW_MS);
    });
    expect(screen.getByText(/Still loading Dana's photos, video, and voice/)).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(TIMEOUT_MS);
    });
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('The server took too long to answer');
    expect(screen.getByRole('region', { name: 'About' })).toBeInTheDocument();
    expect(fetcher.mock.calls.filter(([url]) => url === '/people/Dana/card')).toHaveLength(1);
    recover();
    await act(async () => {
      fireEvent.click(within(alert).getByRole('button', { name: 'Retry' }));
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByRole('region', { name: 'Recent video' })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(fetcher.mock.calls.filter(([url]) => url === '/people/Dana/card')).toHaveLength(2);
  });

  it('keeps cached media usable when its refresh times out and offers Retry', async () => {
    const { fetcher } = waitingCard({
      ...EMPTY,
      shots: [{ id: 'kept', url: '/shot.jpg', ts: 1, cam: 'cam0', score: 0.9 }],
      videos: [{ cam: 'cam0', start: 10, end: 20, duration_s: 10, thumb: '/video-thumb.jpg', what: 'Saved video' }],
      voices: [{ line: 'mic@10', ts_start: 10, text: 'A saved conversation', mic: 'mic' }],
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByText('A saved conversation')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Play video: Saved video/ })).toBeEnabled();
    expect(screen.getByRole('region', { name: 'Photos' }).querySelector('img')).toHaveAttribute('src', '/shot.jpg');
    await act(async () => {
      await vi.advanceTimersByTimeAsync(TIMEOUT_MS);
    });
    expect(screen.getByRole('alert')).toHaveTextContent('The server took too long to answer');
    expect(within(screen.getByRole('alert')).getByRole('button', { name: 'Retry' })).toBeEnabled();
    expect(screen.getByText('A saved conversation')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Play.*A saved conversation/ })).toBeEnabled();
    expect(fetcher.mock.calls.filter(([url]) => url === '/people/Dana/card')).toHaveLength(1);
  });

  it('shows a placeholder under the name of someone not in the list while it loads, then the person', async () => {
    serve();
    show({ people: [] });
    expect(screen.getByLabelText('Loading Dana')).toHaveAttribute('aria-busy', 'true');
    const page = await profile();
    expect(within(page).getByRole('heading', { level: 2, name: 'Dana' })).toBeInTheDocument();
    expect(
      await within(await region('Photos')).findByText(
        'No clear photo of Dana yet. The best shots show up here once a camera has seen their face.',
      ),
    ).toBeInTheDocument();
    expect(within(await region('Recent video')).getByText('No recorded video of Dana yet.')).toBeInTheDocument();
    expect(within(await region('Recent voice')).getByText('Nothing heard from Dana yet.')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /in Review/ })).not.toBeInTheDocument();
    expect(within(page).getByRole('img', { name: 'Dana, no photo yet' })).toHaveTextContent('D');
  });

  it('shows someone in the list at once from their row, and adds what only their profile has when it comes', async () => {
    serve({ profile: { ...DANA, lines: 5, room_lines: 3, call_lines: 2, memory: ['Likes tea'] } });
    let answer = () => {};
    const held = new Promise<void>(resolve => {
      answer = resolve;
    });
    server.use(
      http.get('/people/:name', async () => {
        await held;
        return HttpResponse.json({ ...DANA, lines: 5, room_lines: 3, call_lines: 2, memory: ['Likes tea'] });
      }),
    );
    show({ person: { name: 'Dana', voices: [3], role: 'CEO', organization: 'Acme', lines: 5, room: 3, calls: 2 } });
    // no skeleton: the heading, the chips, and the line counts come from the list's row
    expect(screen.queryByLabelText('Loading Dana')).not.toBeInTheDocument();
    const page = await profile();
    expect(within(page).getByRole('heading', { level: 2, name: 'Dana' })).toBeInTheDocument();
    expect(within(page).getByText('Voice known')).toBeInTheDocument();
    const about = await region('About');
    expect(within(about).getByText('CEO at Acme')).toBeInTheDocument();
    expect(within(about).getByText('5 lines heard, 3 in the room and 2 on calls')).toBeInTheDocument();
    expect(within(about).queryByText('Likes tea')).not.toBeInTheDocument();
    answer();
    expect(await within(about).findByText('Likes tea')).toBeInTheDocument();
  });

  it("shows the server's error as is", async () => {
    serve();
    server.use(http.get('/people/:name', () => HttpResponse.json({ error: 'no such person' }, { status: 404 })));
    show();
    expect(await screen.findByText('no such person')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Dana' })).toBeInTheDocument();
  });

  it('says it could not load when the request fails, and loads on retry', async () => {
    serve();
    let fail = true;
    server.use(
      http.get('/people/:name', ({ params }) =>
        fail
          ? new HttpResponse(null, { status: 500 })
          : HttpResponse.json({ name: params.name, voices: [1], lines: 0 }),
      ),
    );
    show();
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Could not load this profile.');
    fail = false;
    await userEvent.click(within(alert).getByRole('button', { name: 'Retry' }));
    expect(await profile()).toBeInTheDocument();
  });

  it('keeps the card up when a later refresh fails', async () => {
    serve();
    const { client } = show();
    await profile();
    server.use(http.get('/people/:name', () => new HttpResponse(null, { status: 500 })));
    await client.refetchQueries({ queryKey: profileKey('Dana') });
    expect(client.getQueryState(profileKey('Dana'))?.status).toBe('error');
    // Let the page draw the failed refresh before looking: TanStack tells its observers on a setTimeout(0) queued
    // during the refetch, so one more timer, inside act, runs after it and flushes the render it starts. Waiting
    // 300 ms for the error not to appear was a race under load, both ways.
    await act(() => new Promise(resolve => setTimeout(resolve, 0)));
    expect(screen.queryByText('Could not load this profile.')).not.toBeInTheDocument();
    expect(screen.getByRole('article', { name: 'Profile of Dana' })).toBeInTheDocument();
  });

  it('keeps the header and About when the photos, video, and voice fail, and loads them on retry', async () => {
    serve();
    let fail = true;
    server.use(
      http.get('/people/:name/card', () =>
        fail ? HttpResponse.json({ error: 'sightings unavailable' }, { status: 500 }) : HttpResponse.json(EMPTY),
      ),
    );
    show();
    const page = await profile();
    const alert = await within(page).findByRole('alert');
    expect(alert).toHaveTextContent('sightings unavailable');
    expect(within(page).getByText('CEO at Acme', { selector: 'p' })).toBeInTheDocument();
    expect(await region('About')).toBeInTheDocument();
    fail = false;
    await userEvent.click(within(alert).getByRole('button', { name: 'Retry' }));
    expect(await region('Photos')).toBeInTheDocument();
  });
});

describe('the header', () => {
  it('shows the main photo from the card over the list photo', async () => {
    serve({ card: { ...EMPTY, photo: { id: 'd9', url: '/faces/shot/9.jpg', ts: 1, cam: 'cam0', score: 0.9 } } });
    show({ person: { name: 'Dana', voices: [3], photo: '/people/Dana/photos/1.jpg' } });
    const page = await profile();
    await waitFor(() =>
      expect(within(page).getByRole('img', { name: 'Dana' })).toHaveAttribute('src', '/faces/shot/9.jpg'),
    );
  });

  it('replaces an unavailable retained photo without a broken image', async () => {
    serve({
      card: { ...EMPTY, photo: { id: 'missing', url: '/faces/shot/missing.jpg', ts: 1, cam: 'cam0', score: 0.9 } },
    });
    show({ person: { name: 'Dana', voices: [] } });
    const page = await profile();
    const image = await within(page).findByRole('img', { name: 'Dana' });
    fireEvent.error(image);
    expect(within(page).getByRole('img', { name: 'Dana: image unavailable' })).not.toHaveAttribute('src');
  });

  it('shows the list photo until the card has one', async () => {
    serve();
    show({ person: { name: 'Dana', voices: [3], photo: '/people/Dana/photos/1.jpg' } });
    const page = await profile();
    expect(within(page).getByRole('img', { name: 'Dana' })).toHaveAttribute('src', '/people/Dana/photos/1.jpg');
  });

  it('says they are here now, with the camera, from the card when they have no list row', async () => {
    serve({ card: { ...EMPTY, here_now: { cam: 'cam0' }, last_seen: { ts: today(9, 0), cam: 'cam1' } } });
    show({ people: [PEOPLE[1] as Person] }); // not in the list yet, so the card says where
    expect(await screen.findByText('Here now, seen by cam0')).toBeInTheDocument();
    expect(screen.queryByText(/Last seen/)).not.toBeInTheDocument();
  });

  it("follows the list's here now, which refreshes, over the card's", async () => {
    serve({ card: { ...EMPTY, here_now: { cam: 'cam0' }, last_seen: { ts: today(0, 5), cam: 'cam0' } } });
    show({ person: { name: 'Dana', voices: [3], here_now: null, last_seen: today(0, 6), last_seen_cam: 'cam1' } });
    expect(await screen.findByText(/^Last seen today at 00:06 by cam1$/)).toBeInTheDocument();
    expect(screen.queryByText(/Here now/)).not.toBeInTheDocument();
  });

  it('says when and where they were last seen, from the card when they have no list row', async () => {
    serve({ card: { ...EMPTY, last_seen: { ts: today(0, 5), cam: 'cam1' } } });
    show({ people: [PEOPLE[1] as Person] }); // not in the list yet, so the card says where
    expect(await screen.findByText(/^Last seen today at 00:05 by cam1$/)).toBeInTheDocument();
  });

  it('says face history is updating rather than an old time while faces are rematched', async () => {
    serve({
      profile: { ...DANA, last_seen_pending: true },
      card: { ...EMPTY, last_seen: { ts: today(0, 5), cam: 'cam1' } },
    });
    show();
    expect(await screen.findByText('Face history updating')).toBeInTheDocument();
    expect(screen.queryByText(/Last seen/)).not.toBeInTheDocument();
  });

  it('has chips for what is known', async () => {
    serve({ profile: { ...DANA, faces: 2, number: '+14155550123' } });
    show();
    const page = await profile();
    for (const chip of ['Voice known', 'Face known', 'Can be called'])
      expect(await within(page).findByText(chip)).toBeInTheDocument();
  });

  it('has no chips for someone with nothing known', async () => {
    serve({ profile: { name: 'Dana', voices: [] } });
    show({ person: { name: 'Dana', voices: [] } });
    const page = await profile();
    expect(within(page).queryByText('Voice known')).not.toBeInTheDocument();
    expect(within(page).queryByText('Face known')).not.toBeInTheDocument();
  });
});

describe('photos', () => {
  it('shows up to six of the clearest shots, best first', async () => {
    const shots = [1, 2, 3, 4, 5, 6, 7].map(i => ({
      id: `d${i}`,
      url: `/faces/shot/${i}.jpg`,
      ts: today(10, i),
      cam: 'cam0',
      score: 1 - i / 10,
    }));
    serve({ card: { ...EMPTY, shots } });
    show();
    const photos = await region('Photos');
    expect(within(photos).getByText('The clearest front-facing shots')).toBeInTheDocument();
    expect(
      within(photos)
        .getAllByRole('img')
        .map(i => i.getAttribute('src')),
    ).toEqual([1, 2, 3, 4, 5, 6].map(i => `/faces/shot/${i}.jpg`));
  });
});

describe('recent video', () => {
  const video = (i: number, extra: Partial<Card['videos'][number]> = {}) => ({
    cam: 'cam0',
    start: today(18, 40 + i),
    end: today(18, 40 + i) + 54.5,
    duration_s: 54.5,
    thumb: `/faces/shot/${i}.jpg`,
    what: `Making coffee ${i}`,
    ...extra,
  });

  it('shows four tiles that open a floating player with a Review link at that moment', async () => {
    serve({ card: { ...EMPTY, videos: [1, 2, 3, 4, 5].map(i => video(i)) } });
    show();
    const box = await region('Recent video');
    const links = within(box).getAllByRole('button', { name: /^Play video:/ });
    expect(links).toHaveLength(4);
    const first = links[0] as HTMLButtonElement;
    expect(first).toHaveAccessibleName(/^Play video: Making coffee 1, today at 18:41$/);
    await userEvent.click(first);
    expect(await screen.findByRole('dialog', { name: 'Video of Dana' })).toBeInTheDocument();
    const review = screen.getByRole('link', { name: 'Open in Review' }) as HTMLAnchorElement;
    const hash = new URLSearchParams(new URL(review.href).hash.slice(1));
    expect(new URL(review.href).pathname).toBe('/review');
    expect(hash.get('person')).toBe('Dana');
    expect(hash.get('cam')).toBe('cam0');
    expect(hash.get('play')).toBe('1');
    expect(hash.get('t')).toBe(String(today(18, 41))); // a whole second, written without decimals (#2391)
    expect(hash.get('day')).toBe(isoDay(new Date(today(18, 41) * 1000)));
    expect(within(first).getByText('0:55')).toBeInTheDocument();
    expect(within(box).getByText('Making coffee 1')).toBeInTheDocument();
    expect(within(box).getByText('today at 18:41, cam0')).toBeInTheDocument();
    const all = within(box).getByRole('link', { name: 'See all in Review' }) as HTMLAnchorElement;
    expect(new URL(all.href).hash).toBe(`#person=Dana&day=${isoDay(new Date(today(18, 41) * 1000))}`); // the newest video's day (#392)
  });

  it('omits null-thumbnail tiles and shows pending and unavailable video counts instead of an empty claim', async () => {
    serve({ card: { ...EMPTY, videos: [video(1, { thumb: null })], videos_pending: 2, videos_unavailable: 1 } });
    show();
    const box = await region('Recent video');
    expect(within(box).queryByRole('button', { name: /^Play video:/ })).not.toBeInTheDocument();
    expect(within(box).queryByRole('img')).not.toBeInTheDocument();
    expect(within(box).getByText('Processing 2 videos in the background…')).toBeVisible();
    expect(within(box).getByText("1 video can't be shown.")).toBeVisible();
    expect(within(box).queryByText('No recorded video of Dana yet.')).not.toBeInTheDocument();
  });

  it('keeps unavailable-only video history distinct from no recorded video', async () => {
    serve({ card: { ...EMPTY, videos_unavailable: 2 } });
    show();
    const box = await region('Recent video');
    expect(within(box).getByText("2 videos can't be shown.")).toBeVisible();
    expect(within(box).queryByText(/Processing/)).toBeNull();
    expect(within(box).queryByText('No recorded video of Dana yet.')).toBeNull();
  });

  it('polls while only videos are pending and stops once ready clips and unavailable counts arrive', async () => {
    let card: Card = { ...EMPTY, videos_pending: 2 };
    let reads = 0;
    serve();
    server.use(
      http.get('/people/:name/card', () => {
        reads++;
        return HttpResponse.json(card);
      }),
    );
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    try {
      show();
      const box = await region('Recent video');
      expect(await within(box).findByText('Processing 2 videos in the background…')).toBeVisible();
      expect(within(box).queryByRole('button', { name: /^Play video:/ })).toBeNull();
      card = { ...EMPTY, videos: [video(1)], videos_unavailable: 1 };
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
      expect(await within(box).findByRole('button', { name: /^Play video:/ })).toBeVisible();
      expect(box.querySelector('img')).toHaveAttribute('src', '/faces/shot/1.jpg');
      expect(within(box).queryByText(/Processing .* video/)).toBeNull();
      expect(within(box).getByText("1 video can't be shown.")).toBeVisible();
      const done = reads;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(6000);
      });
      expect(reads).toBe(done);
    } finally {
      vi.useRealTimers();
    }
  });

  it('names the camera that saw it, by its name when it has one', async () => {
    serve({ card: { ...EMPTY, videos: [video(1, { what: null, camera_name: 'front camera' })] } });
    show();
    const box = await region('Recent video');
    expect(within(box).getByText('Seen by Front camera')).toBeInTheDocument();
    expect(within(box).getByText('today at 18:41, Front camera')).toBeInTheDocument();
  });

  it('keeps People open for playback, then opens Review from its explicit control', async () => {
    serve({ card: { ...EMPTY, videos: [video(1)] } });
    history.replaceState(null, '', '/people#Dana');
    show();
    await userEvent.click(within(await region('Recent video')).getByRole('button', { name: /^Play video:/ }));
    expect(location.pathname + location.hash).toBe('/people#Dana');
    await userEvent.click(await screen.findByRole('link', { name: 'Open in Review' }));
    expect(location.pathname).toBe('/review');
    expect(new URLSearchParams(location.hash.slice(1)).get('person')).toBe('Dana');
  });
});

describe('recent voice', () => {
  const voice = (i: number, mic: string | null = 'USB Microphone') => ({
    line: `usb-mic@${today(9, i).toFixed(3)}`,
    ts_start: today(9, i),
    text: `Line ${i}`,
    mic,
  });

  it('plays up to four lines by their start, with when and which mic', async () => {
    serve({ card: { ...EMPTY, voices: [voice(1), voice(2, null), voice(3), voice(4), voice(5)] } });
    show();
    const box = await region('Recent voice');
    expect(within(box).getAllByRole('button', { name: /^Play “/ })).toHaveLength(4);
    expect(within(box).getByText('today at 09:01, USB Microphone')).toBeInTheDocument();
    expect(within(box).getByText('today at 09:02')).toBeInTheDocument();
    await userEvent.click(within(box).getByRole('button', { name: 'Play “Line 1”' }));
    expect(usePlayer.getState()).toMatchObject({ key: `said:${voice(1).line}`, ts: today(9, 1) });
    await userEvent.click(within(box).getByRole('button', { name: 'Stop “Line 1”' }));
    expect(usePlayer.getState().key).toBeNull();
    const more = within(box).getByRole('link', { name: 'Hear more in Review' }) as HTMLAnchorElement;
    expect(new URL(more.href).hash).toBe(`#person=Dana&day=${isoDay(new Date(today(9, 1) * 1000))}`); // the newest line's day
  });

  it('keeps two lines that start in the same moment on two mics apart', async () => {
    const a = { ...voice(1), text: 'From the kitchen' };
    const b = { ...a, line: `desk-mic@${a.ts_start.toFixed(3)}`, text: 'From the desk', mic: 'Desk' };
    serve({ card: { ...EMPTY, voices: [a, b] } });
    show();
    const box = await region('Recent voice');
    await userEvent.click(within(box).getByRole('button', { name: 'Play “From the desk”' }));
    expect(within(box).getByRole('button', { name: 'Stop “From the desk”' })).toBeInTheDocument();
    expect(within(box).getByRole('button', { name: 'Play “From the kitchen”' })).toBeInTheDocument();
  });
});

describe('about', () => {
  it('lists the role, phone, other names, what was heard, what is remembered, and the note', async () => {
    serve({
      profile: {
        ...DANA,
        lines: 5,
        room_lines: 3,
        call_lines: 2,
        number: '+14155550123',
        aliases: ['Danny', 'D'],
        memory: ['Likes tea'],
        note: 'From the journal.',
      },
    });
    show();
    const about = await region('About');
    await within(about).findByText('Likes tea'); // once GET /people/<name> has answered
    for (const fact of [
      'CEO at Acme',
      'Phone +14155550123',
      'Also called Danny, D',
      '5 lines heard, 3 in the room and 2 on calls',
      'Likes tea',
      'From the journal.',
    ])
      expect(within(about).getByText(fact)).toBeInTheDocument();
    expect(within(about).getByRole('heading', { name: 'Remembered' })).toBeInTheDocument();
  });

  it('looks them up only when asked', async () => {
    serve();
    server.use(
      http.post('/people/:name/lookup', () =>
        HttpResponse.json({ result: 'Dana runs Acme.', note: 'Public sources.', looked_up: 'Oct 28' }),
      ),
    );
    show();
    const about = await region('About');
    expect(
      within(about).getByText(
        'Public information: searches the web for facts about Dana, with sources. Nothing is sent until you ask.',
      ),
    ).toBeInTheDocument();
    await userEvent.click(button('Look up Dana'));
    expect(await within(about).findByText('Dana runs Acme.')).toBeInTheDocument();
    expect(within(about).getByText('Public sources. Looked up Oct 28.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Look up Dana' })).not.toBeInTheDocument();
  });

  it("shows the lookup's reason, or says it failed", async () => {
    serve();
    server.use(
      http.post('/people/:name/lookup', () =>
        HttpResponse.json({ error: 'No agent is configured (see [agent] in settings).' }),
      ),
    );
    show();
    await profile();
    await userEvent.click(button('Look up Dana'));
    expect(await screen.findByRole('alert')).toHaveTextContent('No agent is configured (see [agent] in settings).');

    server.use(http.post('/people/:name/lookup', () => HttpResponse.error()));
    await userEvent.click(button('Look up Dana'));
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(
        'The lookup failed. Check that the agent backend is reachable, then try again.',
      ),
    );
  });
});

describe('voice clone', () => {
  it("says when their voice isn't known", async () => {
    serve({ profile: { name: 'Dana', voices: [] } });
    show();
    expect(
      await within(await region('Voice clone')).findByText("Liberational doesn't know Dana's voice yet."),
    ).toBeInTheDocument();
  });

  it('says how much was heard, how much is usable, and why it falls short (#388)', async () => {
    const clone = {
      id: 3,
      label: 'Dana',
      ready: false,
      built: false,
      clean_s: 34,
      usable_s: 6,
      rejected_s: 28,
      unchecked_s: 0,
      needs_s: 12,
    };
    serve({ clones: [clone] });
    show();
    expect(
      await within(await region('Voice clone')).findByText(
        "34 s heard, 6 s usable for cloning, needs 12 s. 28 s matched to their voice didn't sound enough like it. Name more of Dana's lines (in Live, Review, or here), pick clips, or record a reference.",
      ),
    ).toBeInTheDocument();
    expect(
      within(await region('Voice clone')).getByRole('button', { name: 'Record a new reference' }),
    ).toBeInTheDocument();
  });

  it('waits for any clear speech at all', async () => {
    serve({ clones: [{ id: 3, label: 'Dana', ready: false, built: false, clean_s: 0 }] });
    show();
    expect(
      await within(await region('Voice clone')).findByText(
        "Dana's voice can be cloned once Liberational has heard about 12 seconds of clear speech from them.",
      ),
    ).toBeInTheDocument();
  });

  it('builds the clone once there is enough', async () => {
    serve({ clones: [{ id: 3, label: 'Dana', ready: true, built: false, clean_s: 14 }] });
    const sent = posts('/clones/build', { ok: true, utterances: 4, seconds: 14 });
    show();
    await userEvent.click(await screen.findByRole('button', { name: "Clone Dana's voice" }));
    await waitFor(() => expect(sent).toEqual([{ id: 3 }]));
  });

  it('shows why a build failed', async () => {
    serve({ clones: [{ id: 3, label: 'Dana', ready: true, built: false, clean_s: 14 }] });
    posts('/clones/build', { ok: false, error: 'Not enough clean speech after all.' });
    show();
    await userEvent.click(await screen.findByRole('button', { name: "Clone Dana's voice" }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Not enough clean speech after all.');
  });

  it('removes a built clone after asking, and offers no Remove before one is built (#2455)', async () => {
    serve({ clones: [{ id: 3, label: 'Dana', ready: true, built: true }] });
    const sent = posts('/clones/remove', { ok: true, removed: ['a.wav'] });
    show();
    await userEvent.click(await within(await region('Voice clone')).findByRole('button', { name: 'Remove clone' }));
    let ask = await screen.findByRole('dialog', { name: 'Remove clone' });
    await userEvent.click(within(ask).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(sent).toEqual([]);
    await userEvent.click(within(await region('Voice clone')).getByRole('button', { name: 'Remove clone' }));
    ask = await screen.findByRole('dialog', { name: 'Remove clone' });
    await userEvent.click(within(ask).getByRole('button', { name: 'Remove' }));
    await waitFor(() => expect(sent).toEqual([{ id: 3 }]));
  });

  it('offers no Remove for a clone not built yet', async () => {
    serve({ clones: [{ id: 3, label: 'Dana', ready: true, built: false, clean_s: 14 }] });
    show();
    await screen.findByRole('button', { name: "Clone Dana's voice" });
    expect(within(await region('Voice clone')).queryByRole('button', { name: 'Remove clone' })).toBeNull();
  });

  it('plays the sample of a built clone', async () => {
    serve({ clones: [{ id: 3, label: 'Dana', ready: true, built: true }] });
    show();
    await userEvent.click(await screen.findByRole('button', { name: 'Play the sample' }));
    expect(usePlayer.getState().key).toBe('ref:3');
    await userEvent.click(button('Stop'));
    expect(usePlayer.getState().key).toBeNull();
  });

  it('says something in their cloned voice', async () => {
    serve({ clones: [{ id: 3, label: 'Dana', ready: true, built: true }] });
    let release = () => {};
    const held = new Promise<void>(r => {
      release = r;
    });
    const sent: unknown[] = [];
    server.use(
      http.post('/clones/say', async ({ request }) => {
        sent.push(await request.json());
        await held;
        return HttpResponse.json({ ok: true, url: '/clones/audio?id=3' });
      }),
    );
    show();
    const box = await screen.findByRole('textbox', { name: "Something for Dana's voice to say" });
    expect(button('Say it')).toBeDisabled();
    await userEvent.type(box, ' hello ');
    await userEvent.click(button('Say it'));
    expect(await screen.findByRole('status')).toHaveTextContent(
      'Speaking. The first time, building the reference takes 5 to 20 seconds.',
    );
    release();
    await waitFor(() => expect(usePlayer.getState().key).toMatch(/^say:\d+$/));
    expect(sent).toEqual([{ id: 3, text: 'hello' }]);
  });

  it('picks clips of their voice for the reference, in the order ticked', async () => {
    serve({ clones: [{ id: 3, label: 'Dana', ready: false, built: false, clean_s: 13, usable_s: 0, needs_s: 12 }] });
    server.use(
      http.get('/clones/:id/lines', () =>
        HttpResponse.json({
          lines: [
            { line: 'kitchen@100.000', ts_start: 100, text: 'first of her lines', dur_s: 7, confirmed: false },
            { line: 'kitchen@200.000', ts_start: 200, text: 'second of her lines', dur_s: 6, confirmed: true },
          ],
        }),
      ),
    );
    const sent = posts('/clones/pick', { ok: true, seconds: 12, clips: 2 });
    show();
    await userEvent.click(await within(await region('Voice clone')).findByRole('button', { name: 'Pick clips' }));
    const dialog = await screen.findByRole('dialog', { name: "Pick clips of Dana's voice" });
    await userEvent.click(await within(dialog).findByRole('checkbox', { name: 'Use "second of her lines"' }));
    const use = within(dialog).getByRole('button', { name: 'Use these clips' });
    expect(use).toBeDisabled(); // 6 s of the 12 it needs
    await userEvent.click(within(dialog).getByRole('checkbox', { name: 'Use "first of her lines"' }));
    expect(within(dialog).getByRole('status')).toHaveTextContent('13 s picked, needs 12 s');
    await userEvent.click(use);
    await waitFor(() => expect(sent).toEqual([{ id: 3, lines: ['kitchen@200.000', 'kitchen@100.000'] }]));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  });

  it('says why the picked clips could not make a reference', async () => {
    serve({ clones: [{ id: 3, label: 'Dana', ready: false, built: false, clean_s: 13, needs_s: 12 }] });
    server.use(
      http.get('/clones/:id/lines', () =>
        HttpResponse.json({
          lines: [{ line: 'kitchen@100.000', ts_start: 100, text: 'first of her lines', dur_s: 13, confirmed: false }],
        }),
      ),
      http.post('/clones/pick', () =>
        HttpResponse.json({ ok: false, error: "1 of those clips isn't Dana's any more." }, { status: 409 }),
      ),
    );
    show();
    await userEvent.click(await within(await region('Voice clone')).findByRole('button', { name: 'Pick clips' }));
    const dialog = await screen.findByRole('dialog', { name: "Pick clips of Dana's voice" });
    await userEvent.click(await within(dialog).findByRole('checkbox', { name: 'Use "first of her lines"' }));
    await userEvent.click(within(dialog).getByRole('button', { name: 'Use these clips' }));
    expect(await within(dialog).findByRole('alert')).toHaveTextContent("1 of those clips isn't Dana's any more.");
  });

  it('offers Pick clips for a voice with only weak matches, before any clone is listed', async () => {
    serve({ clones: [] }); // no clear speech yet, so /clones lists no clone for Dana
    server.use(http.get('/clones/:id/lines', () => HttpResponse.json({ lines: [] })));
    show();
    await userEvent.click(await within(await region('Voice clone')).findByRole('button', { name: 'Pick clips' }));
    expect(await screen.findByText("No lines of Dana's are long and clear enough to use yet.")).toBeInTheDocument();
  });

  it('never offers Pick clips for the owner, whose clone is their recorded reference', async () => {
    serve({ profile: { ...DANA, owner: true }, clones: [{ id: 3, label: 'Dana', ready: true, built: true }] });
    show();
    await within(await region('Voice clone')).findByRole('button', { name: 'Record a new reference' });
    expect(within(await region('Voice clone')).queryByRole('button', { name: 'Pick clips' })).not.toBeInTheDocument();
  });

  it('records a new reference under their name', async () => {
    serve();
    show();
    await userEvent.click(within(await region('Voice clone')).getByRole('button', { name: 'Record a new reference' }));
    const dialog = await screen.findByRole('dialog', { name: 'Record a new reference for Dana' });
    expect(within(dialog).getByRole('textbox', { name: 'Name for this voice' })).toHaveValue('Dana');
  });
});

describe('role and organization', () => {
  it('show under the name, and the form saves them at once', async () => {
    serve();
    const sent = posts('/people/describe', { ok: true, name: 'Dana', role: 'CTO', organization: 'Acme' });
    const props = show();
    const page = await profile();
    await menu('Role and organization');
    const role = screen.getByRole('textbox', { name: 'Role' });
    expect(role).toHaveValue('CEO');
    expect(screen.getByRole('textbox', { name: 'Organization' })).toHaveValue('Acme');
    await userEvent.clear(role);
    await userEvent.type(role, 'CTO');
    await userEvent.click(button('Save'));

    await waitFor(() => expect(sent).toEqual([{ name: 'Dana', role: 'CTO', organization: 'Acme' }]));
    expect(await within(page).findByText('CTO at Acme', { selector: 'p' })).toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Role' })).not.toBeInTheDocument();
    expect(props.onChanged).toHaveBeenCalledWith('Dana');
  });
});

describe('phone number', () => {
  it('saves a number and shows it', async () => {
    serve();
    const sent = posts('/people/add', { name: 'Dana' });
    const props = show();
    await profile();
    await menu('Phone number');
    await userEvent.type(screen.getByRole('textbox', { name: 'Phone number' }), '+14155550123');
    await userEvent.click(button('Save'));
    await waitFor(() => expect(sent).toEqual([{ name: 'Dana', number: '+14155550123' }]));
    expect(await within(await region('About')).findByText('Phone +14155550123')).toBeInTheDocument();
    expect(props.onChanged).toHaveBeenCalled();
  });

  it('asks for the country code before sending', async () => {
    serve();
    const sent = posts('/people/add', { name: 'Dana' });
    show();
    await profile();
    await menu('Phone number');
    await userEvent.type(screen.getByRole('textbox', { name: 'Phone number' }), '415 555 0123');
    await userEvent.click(button('Save'));
    expect(screen.getByRole('status')).toHaveTextContent('Type the number with its country code, like +14155550123.');
    expect(sent).toEqual([]);
  });

  it("shows the server's refusal", async () => {
    serve();
    server.use(
      http.post('/people/add', () =>
        HttpResponse.json({ error: 'phone must look like +14155550123' }, { status: 400 }),
      ),
    );
    show();
    await profile();
    await menu('Phone number');
    await userEvent.type(screen.getByRole('textbox', { name: 'Phone number' }), '+15550001');
    await userEvent.click(button('Save'));
    expect(await screen.findByRole('status')).toHaveTextContent('phone must look like +14155550123');
  });
});

describe('rename', () => {
  const numbered: Person[] = [
    { name: 'Dana', voices: [3], number: '+15550001' },
    { name: 'Eli', voices: [4], number: '+15550002' },
  ];

  it('saves a new name without asking', async () => {
    serve();
    const reply = { ok: true, name: 'Danielle', people: [] };
    const sent = posts('/people/rename', reply);
    const props = show();
    await profile();
    await userEvent.click(button('Rename'));
    const field = screen.getByRole('textbox', { name: 'New name' });
    expect(field).toHaveValue('Dana');
    await userEvent.clear(field);
    expect(button('Save')).toBeDisabled();
    await userEvent.type(field, 'Danielle');
    await userEvent.click(button('Save'));
    await waitFor(() => expect(props.onChanged).toHaveBeenCalledWith('Danielle', reply));
    expect(sent).toEqual([{ from: 'Dana', to: 'Danielle' }]);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('asks before merging into someone with another number, and says which number goes', async () => {
    serve({ profile: { ...DANA, number: '+15550001' } });
    const reply = { ok: true, name: 'Eli', merged: true };
    const sent = posts('/people/rename', reply);
    const props = show({ people: numbered });
    await profile();
    await userEvent.click(button('Rename'));
    const field = screen.getByRole('textbox', { name: 'New name' });
    await userEvent.clear(field);
    await userEvent.type(field, 'Eli');
    await userEvent.click(button('Save'));

    const dialog = await screen.findByRole('dialog', { name: 'Merge Dana into Eli?' });
    expect(
      within(dialog).getByText("Eli keeps the number +15550002, and Dana's number +15550001 is not kept."),
    ).toBeInTheDocument();
    expect(sent).toEqual([]);
    await userEvent.click(within(dialog).getByRole('button', { name: 'Merge' }));
    await waitFor(() => expect(props.onChanged).toHaveBeenCalledWith('Eli', reply));
    expect(sent).toEqual([{ from: 'Dana', to: 'Eli' }]);
  });

  it('does nothing when the name is the same, or the merge is cancelled', async () => {
    serve({ profile: { ...DANA, number: '+15550001' } });
    const sent = posts('/people/rename', { ok: true, name: 'Eli' });
    const props = show({ people: numbered });
    await profile();
    await userEvent.click(button('Rename'));
    await userEvent.click(button('Save'));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();

    const field = screen.getByRole('textbox', { name: 'New name' });
    await userEvent.clear(field);
    await userEvent.type(field, 'Eli');
    await userEvent.click(button('Save'));
    const dialog = await screen.findByRole('dialog', { name: 'Merge Dana into Eli?' });
    await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(sent).toEqual([]);
    expect(props.onChanged).not.toHaveBeenCalled();
    expect(screen.getByRole('textbox', { name: 'New name' })).toBeInTheDocument();
  });

  it("shows the server's refusal", async () => {
    serve();
    server.use(http.post('/people/rename', () => HttpResponse.json({ error: 'That name is taken' }, { status: 400 })));
    const props = show();
    await profile();
    await userEvent.click(button('Rename'));
    await userEvent.type(screen.getByRole('textbox', { name: 'New name' }), 'x');
    await userEvent.click(button('Save'));
    const error = await screen.findByText('That name is taken');
    const field = screen.getByRole('textbox', { name: 'New name' });
    expect(field).toHaveAttribute('aria-invalid', 'true');
    expect(field).toHaveAccessibleDescription('That name is taken');
    expect(
      error.compareDocumentPosition(screen.getByRole('region', { name: 'Reference media' })) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(props.onChanged).not.toHaveBeenCalled();
  });
});

describe('same person as', () => {
  it('always asks before merging', async () => {
    serve();
    const reply = { ok: true, name: 'Eli', merged: true };
    const sent = posts('/people/rename', reply);
    const props = show();
    await profile();
    await menu('Same person as');
    expect(button('Merge')).toBeDisabled();
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Same person as' }), 'Eli');
    await userEvent.click(button('Merge'));

    const dialog = await screen.findByRole('dialog', { name: 'Merge Dana into Eli?' });
    expect(within(dialog).getByText("Their voices, faces, phone, and lines become Eli's.")).toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole('button', { name: 'Merge' }));
    await waitFor(() => expect(props.onChanged).toHaveBeenCalledWith('Eli', reply));
    expect(sent).toEqual([{ from: 'Dana', to: 'Eli' }]);
  });

  it('says which number goes when both have one', async () => {
    serve({ profile: { ...DANA, number: '+15550001' } });
    show({
      people: [
        { name: 'Dana', voices: [3], number: '+15550001' },
        { name: 'Eli', voices: [4], number: '+15550002' },
      ],
    });
    await profile();
    await menu('Same person as');
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Same person as' }), 'Eli');
    await userEvent.click(button('Merge'));
    const dialog = await screen.findByRole('dialog', { name: 'Merge Dana into Eli?' });
    expect(
      within(dialog).getByText(
        "Their voices, faces, and lines become Eli's. Eli keeps the number +15550002, and Dana's number +15550001 is not kept.",
      ),
    ).toBeInTheDocument();
  });

  it('is left out of the menu when nobody else is known', async () => {
    serve();
    show({ people: [PEOPLE[0] as Person] });
    await profile();
    expect(await menuItems()).toEqual([
      'Role and organization',
      'Phone number',
      "Pick what's theirs",
      'This is me',
      'Delete',
    ]);
  });
});

describe('delete', () => {
  it('asks in a dialog, then reports what was forgotten', async () => {
    serve();
    const reply = {
      ok: true,
      name: 'Dana',
      removed: { voices: 2, faces: 3, photos: 4, sightings: 5 },
      notes: ['Their lines are renamed in a moment.'],
    };
    const sent = posts('/people/delete', reply);
    const props = show();
    await profile();
    await menu('Delete');
    const ask = screen.getByRole('dialog', { name: 'Delete Dana' });
    expect(ask).toHaveTextContent('Delete Dana? This forgets their face and voice. Recordings stay.');
    await userEvent.click(within(ask).getByRole('button', { name: 'Delete' }));
    await waitFor(() =>
      expect(props.onDeleted).toHaveBeenCalledWith(
        reply,
        'Forgot Dana: 2 voices, 3 face samples, 4 photos, 5 recorded sightings. Their lines are renamed in a moment.',
      ),
    );
    expect(sent).toEqual([{ name: 'Dana' }]);
  });

  it('closes on Cancel or Escape before opening another form', async () => {
    serve();
    show();
    await profile();
    await menu('Delete');
    await userEvent.click(
      within(screen.getByRole('dialog', { name: 'Delete Dana' })).getByRole('button', { name: 'Cancel' }),
    );
    expect(screen.queryByRole('dialog', { name: 'Delete Dana' })).not.toBeInTheDocument();

    await menu('Delete');
    await userEvent.keyboard('{Escape}');
    await userEvent.click(button('Rename'));
    expect(screen.queryByRole('dialog', { name: 'Delete Dana' })).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'New name' })).toBeInTheDocument();
    await userEvent.click(button('Rename'));
    expect(screen.queryByRole('textbox', { name: 'New name' })).not.toBeInTheDocument();
  });
});

describe('the owner', () => {
  it('has a You pill, and no Delete or Same person as', async () => {
    serve({ profile: { ...DANA, owner: true } });
    show({ person: { ...(PEOPLE[0] as Person), owner: true } });
    const page = await profile();
    expect(within(page).getByText('You')).toBeInTheDocument();
    expect(await menuItems()).toEqual(['Role and organization', 'Phone number', "Pick what's theirs"]);
  });
});

describe('back to people', () => {
  it('goes back', async () => {
    serve();
    const props = show();
    await profile();
    await userEvent.click(button('Back to people'));
    expect(props.onBack).toHaveBeenCalled();
  });
});

describe('This is me', () => {
  it('confirms the current voices before granting owner identity', async () => {
    serve();
    const sent: unknown[] = [];
    const reply = { ok: true, name: 'Alice', merged: true };
    server.use(
      http.post('/people/me', async ({ request }) => {
        const body = (await request.json()) as { voices?: number[] };
        sent.push(body);
        return body.voices
          ? HttpResponse.json(reply)
          : HttpResponse.json({ ok: false, confirmation: true, voices: [3, 5] }, { status: 409 });
      }),
    );
    const props = show();
    await profile();
    await menu('This is me');
    const dialog = await screen.findByRole('dialog', { name: 'Merge Dana into you?' });
    expect(within(dialog).getByText(/also has 2 voices.*become yours/)).toBeInTheDocument();
    expect(sent).toEqual([{ name: 'Dana' }]);
    expect(props.onChanged).not.toHaveBeenCalled();
    await userEvent.click(within(dialog).getByRole('button', { name: 'This is me' }));
    await waitFor(() => expect(props.onChanged).toHaveBeenCalledWith('Alice', reply));
    expect(sent).toEqual([{ name: 'Dana' }, { name: 'Dana', voices: [3, 5] }]);
  });

  it('does not transfer voices when confirmation is cancelled', async () => {
    serve();
    const sent = posts('/people/me', { ok: false, confirmation: true, voices: [3] });
    const props = show();
    await profile();
    await menu('This is me');
    const dialog = await screen.findByRole('dialog');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    expect(sent).toEqual([{ name: 'Dana' }]);
    expect(props.onChanged).not.toHaveBeenCalled();
  });

  it('merges a person without voices from the explicit action without an extra dialog', async () => {
    serve({ profile: { ...DANA, voices: [] } });
    const reply = { ok: true, name: 'Alice', merged: true };
    const sent = posts('/people/me', reply);
    const props = show();
    await profile();
    await menu('This is me');
    await waitFor(() => expect(props.onChanged).toHaveBeenCalledWith('Alice', reply));
    expect(sent).toEqual([{ name: 'Dana' }]);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});

it('routes a rename to the owner through explicit voice confirmation', async () => {
  serve();
  vi.stubGlobal('NAMES', { owner: 'Alice' });
  const sent = posts('/people/me', { ok: false, confirmation: true, voices: [3] });
  const ordinary = posts('/people/rename', { ok: true });
  show();
  await profile();
  await userEvent.click(button('Rename'));
  const field = screen.getByRole('textbox', { name: 'New name' });
  await userEvent.clear(field);
  await userEvent.type(field, 'Alice');
  await userEvent.click(button('Save'));
  const dialog = await screen.findByRole('dialog', { name: 'Merge Dana into you?' });
  expect(sent).toEqual([{ name: 'Dana' }]);
  expect(ordinary).toEqual([]);
  await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
  vi.unstubAllGlobals();
});

it('requires a fresh action when the voice set changes after confirmation', async () => {
  serve();
  const sent = posts('/people/me', { ok: false, confirmation: true, voices: [3] });
  const props = show();
  await profile();
  await menu('This is me');
  const dialog = await screen.findByRole('dialog');
  await userEvent.click(within(dialog).getByRole('button', { name: 'This is me' }));
  expect(await screen.findByText(/Their voices changed/)).toBeInTheDocument();
  expect(sent).toEqual([{ name: 'Dana' }, { name: 'Dana', voices: [3] }]);
  expect(props.onChanged).not.toHaveBeenCalled();
});
