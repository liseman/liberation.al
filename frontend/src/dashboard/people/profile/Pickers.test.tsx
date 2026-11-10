import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http, type JsonBodyType } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { TIMEOUT_MS } from '../../../shared/http';
import { newQueryClient, Providers } from '../../../shared/Providers';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { SLOW_MS } from '../../lib/loading';
import { stop, usePlayer } from '../../lib/player';
import { type Card, type Person, UNNAMED_KEY, type Unnamed } from '../api';
import { Pickers } from './Pickers';
import { Profile } from './Profile';

const SAM_CARD: Card = { name: 'Sam', photo: null, shots: [], videos: [], voices: [], last_seen: null, here_now: null };
const t0 = new Date(2026, 9, 26, 19, 30).getTime() / 1000; // a Monday, 19:30 local
const UNNAMED: Unnamed = {
  faces: [
    { id: 17, photos: ['/faces/unknown/17.jpg', '/faces/shot/170.jpg'], count: 17, last_seen: t0, cam: 'cam0' },
    { id: 18, photos: ['/faces/unknown/18.jpg'], count: 1, last_seen: null, cam: null },
  ],
  videos: [
    {
      id: 'cam0@1793212345.50',
      cam: 'cam0',
      start: t0,
      end: t0 + 54.5,
      duration_s: 54.5,
      thumb: '/saved/video-1.jpg',
      what: null,
    },
    {
      id: 'cam1@1793212400.00',
      cam: 'cam1',
      camera_name: 'rear camera',
      start: t0 - 3600,
      end: t0 - 3570,
      duration_s: 30,
      thumb: '/saved/video-2.jpg',
      what: 'Washing up',
    },
  ],
  voices: [
    {
      id: 9,
      label: 'Unnamed voice A',
      lines: 12,
      last_heard: t0,
      sample: { line: 'usb-mic@1793212345.500', ts_start: 1793212345.5, text: 'Is the kettle on?' },
    },
  ],
};

function serve({ unnamed = UNNAMED, card = SAM_CARD }: { unnamed?: Unnamed; card?: Card } = {}) {
  server.use(
    http.get('/inbox/unnamed', () => HttpResponse.json(unnamed)),
    http.get('/people/:name', ({ params }) => HttpResponse.json({ name: params.name, voices: [] })),
    http.get('/people/:name/card', () => HttpResponse.json(card)),
    http.get('/clones', () => HttpResponse.json({ clones: [] })),
  );
}

/** Record each JSON body posted to path, and answer with reply (and status). */
function posts(path: string, reply: JsonBodyType, status = 200) {
  const bodies: unknown[] = [];
  server.use(
    http.post(path, async ({ request }) => {
      bodies.push(await request.json());
      return HttpResponse.json(reply, { status });
    }),
  );
  return bodies;
}

function show({ isNew = true, people = [] as Person[] } = {}) {
  const props = {
    name: 'Sam',
    person: undefined,
    people,
    onChanged: vi.fn(),
    onDeleted: vi.fn(),
    onBack: vi.fn(),
    onPickDone: vi.fn(),
    isNew,
  };
  renderPage(<Profile {...props} />);
  return props;
}

const footer = () => screen.getByRole('region', { name: 'Picked for Sam' });
const region = (name: string) => screen.findByRole('region', { name });

beforeEach(() => {
  vi.useFakeTimers({ now: new Date(2026, 9, 28, 21, 5), toFake: ['Date'] }); // "Mon 19:30" stays two days back
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
});
afterEach(() => {
  stop();
  vi.useRealTimers();
});

describe("a new person's card", () => {
  it('shows unavailable-only unnamed video history instead of claiming everything has a name', async () => {
    serve({ unnamed: { faces: [], voices: [], videos: [], videos_unavailable: 2 } });
    renderPage(<Pickers name="Sam" onAssigned={vi.fn()} onSkip={vi.fn()} />);
    const videos = await region('Video with nobody named');
    expect(within(videos).getByText("2 videos can't be shown.")).toBeVisible();
    expect(within(videos).queryByText(/Processing/)).toBeNull();
    expect(within(videos).queryByText('No video with only unnamed people in it.')).toBeNull();
    expect(screen.queryByRole('region', { name: 'Nothing to pick' })).toBeNull();
  });

  it('polls pending video independently of photos and lists only ready thumbnail picks', async () => {
    const first = UNNAMED.videos[0];
    if (!first) throw new Error('fixture video missing');
    let unnamed: Unnamed = {
      faces: [],
      voices: [],
      videos: [{ ...first, thumb: null }],
      videos_pending: 1,
      videos_unavailable: 2,
    };
    let reads = 0;
    serve();
    server.use(
      http.get('/inbox/unnamed', () => {
        reads++;
        return HttpResponse.json(unnamed);
      }),
    );
    vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] });
    renderPage(<Pickers name="Sam" onAssigned={vi.fn()} onSkip={vi.fn()} />);
    const videos = await region('Video with nobody named');
    expect(await within(videos).findByText('Processing 1 video in the background…')).toBeVisible();
    expect(within(videos).getByText("2 videos can't be shown.")).toBeVisible();
    expect(within(videos).queryByRole('button', { name: /^Video:/ })).toBeNull();
    expect(videos.querySelector('img')).toBeNull();
    expect(within(videos).queryByText('No video with only unnamed people in it.')).toBeNull();
    expect(screen.queryByRole('region', { name: 'Nothing to pick' })).toBeNull();
    unnamed = { ...unnamed, videos_pending: 0, videos: [...UNNAMED.videos, { ...first, id: 'bad', thumb: null }] };
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await waitFor(() => expect(within(videos).getAllByRole('button', { name: /^Video:/ })).toHaveLength(2));
    expect(videos.querySelectorAll('img')).toHaveLength(2);
    expect(within(videos).queryByText(/Processing .* video/)).toBeNull();
    const done = reads;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(reads).toBe(done);
  });

  it('keeps unavailable pictures out of the picker while pending crops arrive without hiding video metadata', async () => {
    let unnamed: Unnamed = {
      faces: [{ id: 17, photos: [], count: 17, last_seen: t0, cam: 'cam0', shots_pending: 3 }],
      videos: UNNAMED.videos.slice(0, 1),
      voices: [],
      shots_pending: 3,
    };
    serve();
    server.use(http.get('/inbox/unnamed', () => HttpResponse.json(unnamed)));
    vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] });
    renderPage(<Pickers name="Sam" onAssigned={vi.fn()} onSkip={vi.fn()} />);
    const faces = await region("Faces nobody's named");
    expect(await within(faces).findByText('Processing 3 detections in the background…')).toBeVisible();
    expect(faces.querySelector('img')).toBeNull();
    expect(within(faces).queryByText('Every face seen so far has a name.')).toBeNull();
    const videos = await region('Video with nobody named');
    expect(within(videos).getByRole('button', { name: /^Video:/ })).toBeVisible();
    expect(videos.querySelector('img')).toHaveAttribute('src', '/saved/video-1.jpg');
    unnamed = {
      ...unnamed,
      shots_pending: 0,
      shots_unavailable: 2,
      faces: [{ id: 17, photos: ['/faces/shot/ready.jpg'], count: 17, last_seen: t0, cam: 'cam0', shots_pending: 0 }],
    };
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await waitFor(() => expect(within(faces).queryByText(/Processing .* detections/)).toBeNull());
    expect(faces.querySelector('img')).toHaveAttribute('src', '/faces/shot/ready.jpg');
    expect(faces.querySelector('img')).toHaveAttribute('loading', 'lazy');
    expect(within(faces).getByText("2 detections can't be shown.")).toBeVisible();
  });

  it('opens on picking what is theirs, with nothing picked yet', async () => {
    serve();
    show();
    expect(
      await screen.findByText("New. Pick the photos, video, and voice that are Sam's, and the card fills in."),
    ).toBeInTheDocument();
    const faces = await region("Faces nobody's named");
    expect(within(faces).getByRole('button', { name: 'Face 1 of 2: Seen 17 times, Mon 19:30' })).toHaveAttribute(
      'aria-pressed',
      'false',
    );
    expect(within(faces).getByRole('button', { name: 'Face 2 of 2: Seen once' })).toBeInTheDocument();
    expect(
      within(await region('Video with nobody named')).getByRole('button', { name: /^Video: Mon 19:30, cam0, 0:55$/ }),
    ).toBeInTheDocument();
    expect(within(await region("Voices nobody's named")).getByText('Unnamed voice A, 12 lines')).toBeInTheDocument();
    expect(within(footer()).getByText('0 picked for Sam')).toBeInTheDocument();
    expect(within(footer()).getByRole('button', { name: 'Add to Sam' })).toBeDisabled();
    expect(screen.queryByRole('region', { name: 'Photos' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'About' })).not.toBeInTheDocument();
  });

  it('gives what was picked to them, then shows the filled card with Undo', async () => {
    serve();
    const reply = {
      ok: true,
      name: 'Sam',
      assigned: { faces: 1, voices: 1, videos: 1 },
      failed: [],
      undo: 'tok-1',
      people: [],
    };
    const sent = posts('/people/assign', reply);
    const props = show();
    await userEvent.click(await screen.findByRole('button', { name: /^Face 1 of 2: Seen 17 times/ }));
    await userEvent.click(screen.getByRole('button', { name: /^Video: Mon 19:30, cam0/ }));
    await userEvent.click(screen.getByRole('switch', { name: 'Unnamed voice A: this is Sam' }));
    expect(screen.getByRole('button', { name: /^Face 1 of 2: Seen 17 times/ })).toHaveAttribute('aria-pressed', 'true');
    expect(within(footer()).getByText('3 picked for Sam')).toBeInTheDocument();
    await userEvent.click(within(footer()).getByRole('button', { name: 'Add 3 to Sam' }));

    expect(await screen.findByText('Added 3 to Sam.')).toBeInTheDocument();
    expect(sent).toEqual([{ name: 'Sam', faces: [17], voices: [9], videos: ['cam0@1793212345.50'] }]);
    expect(await region('Photos')).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Picked for Sam' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Undo' })).toBeInTheDocument();
    expect(props.onChanged).toHaveBeenCalledWith(undefined, reply);
    expect(props.onPickDone).toHaveBeenCalled();
  });

  it('unpicks a second click, and sends only what is still picked', async () => {
    serve();
    const sent = posts('/people/assign', { ok: true, name: 'Sam', assigned: { faces: 1, voices: 0, videos: 0 } });
    show();
    const face17 = await screen.findByRole('button', { name: /^Face 1 of 2: Seen 17 times/ });
    await userEvent.click(face17);
    await userEvent.click(screen.getByRole('button', { name: 'Face 2 of 2: Seen once' }));
    await userEvent.click(face17);
    expect(face17).toHaveAttribute('aria-pressed', 'false');
    await userEvent.click(within(footer()).getByRole('button', { name: 'Add 1 to Sam' }));
    await waitFor(() => expect(sent).toEqual([{ name: 'Sam', faces: [18], voices: [], videos: [] }]));
  });

  it("says what didn't stick when only some of it did", async () => {
    serve();
    posts('/people/assign', {
      ok: true,
      name: 'Sam',
      assigned: { faces: 1, voices: 0, videos: 0 },
      failed: [{ kind: 'voice', id: 9, error: 'that voice is gone' }],
      undo: 'tok-2',
    });
    show();
    await userEvent.click(await screen.findByRole('button', { name: /^Face 1 of 2: Seen 17 times/ }));
    await userEvent.click(screen.getByRole('switch', { name: 'Unnamed voice A: this is Sam' }));
    await userEvent.click(within(footer()).getByRole('button', { name: 'Add 2 to Sam' }));
    expect(await screen.findByText("Added 1 to Sam. 1 item didn't stick: that voice is gone.")).toBeInTheDocument();
  });

  it('stays on the pickers with the picks kept when nothing stuck', async () => {
    serve();
    posts(
      '/people/assign',
      { ok: false, error: 'Nothing was added.', failed: [{ kind: 'face', id: 17, error: 'face group is gone' }] },
      400,
    );
    const props = show();
    await userEvent.click(await screen.findByRole('button', { name: /^Face 1 of 2: Seen 17 times/ }));
    await userEvent.click(within(footer()).getByRole('button', { name: 'Add 1 to Sam' }));
    expect(
      await within(footer()).findByText("Nothing was added. Couldn't add a face (face group is gone)."),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Face 1 of 2: Seen 17 times/ })).toHaveAttribute('aria-pressed', 'true');
    expect(props.onPickDone).not.toHaveBeenCalled();
  });

  it('says why when a 200 reply refuses, with what failed', async () => {
    serve();
    posts('/people/assign', {
      ok: false,
      error: 'Nothing was added.',
      failed: [{ kind: 'face', id: 17, error: 'face group is gone' }],
    });
    show();
    await userEvent.click(await screen.findByRole('button', { name: /^Face 1 of 2: Seen 17 times/ }));
    await userEvent.click(within(footer()).getByRole('button', { name: 'Add 1 to Sam' }));
    expect(
      await within(footer()).findByText("Nothing was added. Couldn't add a face (face group is gone)."),
    ).toBeInTheDocument();
  });

  it('skips to the empty card without sending anything', async () => {
    serve();
    const sent = posts('/people/assign', { ok: true });
    const props = show();
    await userEvent.click(
      await within(await screen.findByRole('region', { name: 'Picked for Sam' })).findByRole('button', {
        name: 'Skip for now',
      }),
    );
    expect(await region('Photos')).toBeInTheDocument();
    expect(sent).toEqual([]);
    expect(props.onPickDone).toHaveBeenCalled();
  });

  it('drops a pick whose item is gone after a refresh, from the count and from what is sent', async () => {
    let unnamed = UNNAMED;
    serve();
    server.use(http.get('/inbox/unnamed', () => HttpResponse.json(unnamed)));
    const sent = posts('/people/assign', { ok: true, name: 'Sam', assigned: { faces: 0, voices: 1, videos: 0 } });
    const { client } = renderPage(
      <Profile
        name="Sam"
        person={undefined}
        people={[]}
        onChanged={vi.fn()}
        onDeleted={vi.fn()}
        onBack={vi.fn()}
        isNew
      />,
    );
    await userEvent.click(await screen.findByRole('button', { name: /^Face 1 of 2: Seen 17 times/ }));
    await userEvent.click(screen.getByRole('switch', { name: 'Unnamed voice A: this is Sam' }));
    unnamed = { ...UNNAMED, faces: UNNAMED.faces.slice(1) }; // someone else named face 17 meanwhile
    await client.refetchQueries({ queryKey: UNNAMED_KEY });
    expect(await within(footer()).findByText('1 picked for Sam')).toBeInTheDocument();
    await userEvent.click(within(footer()).getByRole('button', { name: 'Add 1 to Sam' }));
    await waitFor(() => expect(sent).toEqual([{ name: 'Sam', faces: [], voices: [9], videos: [] }]));
  });

  it('keeps Rename and the More menu out of the way until picking is done', async () => {
    serve();
    show();
    await region("Faces nobody's named");
    expect(screen.queryByRole('button', { name: 'Rename' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'More for Sam' })).not.toBeInTheDocument();
    await userEvent.click(within(footer()).getByRole('button', { name: 'Skip for now' }));
    expect(await screen.findByRole('button', { name: 'Rename' })).toBeInTheDocument();
  });

  it("names a video's camera by its name when it has one", async () => {
    serve();
    show();
    expect(
      await screen.findByRole('button', { name: 'Video: Washing up, Mon 18:30, Rear camera, 0:30' }),
    ).toBeInTheDocument();
  });

  it('plays a voice before you pick it', async () => {
    serve();
    show();
    await userEvent.click(await screen.findByRole('button', { name: 'Play Unnamed voice A' }));
    expect(usePlayer.getState()).toMatchObject({ key: 'said:usb-mic@1793212345.500', ts: 1793212345.5 });
  });

  it('says when there is nothing unnamed to pick', async () => {
    serve({ unnamed: { faces: [], videos: [], voices: [] } });
    show();
    expect(
      await screen.findByText(
        "Everything seen and heard so far has a name. You can link Sam's face and voice later from Live or Who's this?",
      ),
    ).toBeInTheDocument();
  });

  it('says it could not load, and loads on retry', async () => {
    serve();
    let fail = true;
    server.use(
      http.get('/inbox/unnamed', () =>
        fail ? HttpResponse.json({ error: 'sightings busy' }, { status: 503 }) : HttpResponse.json(UNNAMED),
      ),
    );
    show();
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('sightings busy');
    fail = false;
    await userEvent.click(within(alert).getByRole('button', { name: 'Retry' }));
    expect(await region("Faces nobody's named")).toBeInTheDocument();
  });
});

describe("Pick what's theirs, for someone already known", () => {
  it('opens from the More menu, without calling them new', async () => {
    serve();
    show({ isNew: false });
    expect(await region('Photos')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'More for Sam' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: "Pick what's theirs" }));
    expect(
      await screen.findByText("Pick the photos, video, and voice that are Sam's, and the card fills in."),
    ).toBeInTheDocument();
    expect(await region("Faces nobody's named")).toBeInTheDocument();
  });
});

it('keeps loaded picks visible when their refresh fails', async () => {
  serve();
  const { client } = renderPage(<Pickers name="Sam" onAssigned={vi.fn()} onSkip={vi.fn()} />);
  await region("Faces nobody's named");
  server.use(http.get('/inbox/unnamed', () => HttpResponse.json({ error: 'history busy' }, { status: 503 })));
  await act(async () => {
    await client.refetchQueries({ queryKey: UNNAMED_KEY });
  });
  expect(screen.getByRole('region', { name: "Faces nobody's named" })).toBeInTheDocument();
  expect(await screen.findByRole('alert')).toHaveTextContent('history busy');
  expect(within(screen.getByRole('alert')).getByRole('button', { name: 'Retry' })).toBeEnabled();
});

it('shows slow loading, reports the first deadline, and recovers profile picks on Retry', async () => {
  vi.useRealTimers();
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  const client = newQueryClient();
  let waiting = true;
  const fetcher = vi.fn<typeof fetch>((url, init) => {
    expect(url).toBe('/inbox/unnamed');
    if (!waiting) return Promise.resolve(Response.json(UNNAMED));
    return new Promise<Response>((_, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
    });
  });
  vi.stubGlobal('fetch', fetcher);
  try {
    render(
      <Providers client={client} test>
        <Pickers name="Sam" onAssigned={vi.fn()} onSkip={vi.fn()} />
      </Providers>,
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByText('Loading what nobody has named…')).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(SLOW_MS);
    });
    expect(screen.getByText(/Still loading what nobody has named/)).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(TIMEOUT_MS);
    });
    expect(screen.getByRole('alert')).toHaveTextContent('The server took too long to answer');
    expect(screen.queryByText('Everything seen and heard so far has a name.')).not.toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(1);
    waiting = false;
    await act(async () => {
      fireEvent.click(within(screen.getByRole('alert')).getByRole('button', { name: 'Retry' }));
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByRole('region', { name: "Faces nobody's named" })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(2);
  } finally {
    client.clear();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  }
});

it('keeps partial voice picks usable and retries missing face history', async () => {
  serve({ unnamed: { ...UNNAMED, videos: [], incomplete: 'Could not load face photos and video clips.' } });
  renderPage(<Pickers name="Sam" onAssigned={vi.fn()} onSkip={vi.fn()} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Could not load face photos and video clips.');
  expect(screen.getByText(/Is the kettle on\?/)).toBeInTheDocument();
  expect(screen.queryByText('No video with only unnamed people in it.')).not.toBeInTheDocument();
  serve();
  await userEvent.click(within(screen.getByRole('alert')).getByRole('button', { name: 'Retry' }));
  await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
  expect(screen.getByText('Washing up')).toBeInTheDocument();
});

it('does not describe an incomplete empty response as everything named', async () => {
  serve({ unnamed: { faces: [], voices: [], videos: [], incomplete: 'Could not load face photos and video clips.' } });
  renderPage(<Pickers name="Sam" onAssigned={vi.fn()} onSkip={vi.fn()} />);
  await screen.findByRole('alert');
  expect(screen.queryByText('Nothing to pick')).not.toBeInTheDocument();
  expect(screen.queryByText(/Everything seen and heard so far has a name/)).not.toBeInTheDocument();
});
