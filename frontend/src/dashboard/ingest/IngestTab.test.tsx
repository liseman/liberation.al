import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { describe, expect, it, onTestFinished, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { IngestItem, IngestView, SourceInfo } from './api';
import { INGEST_KEY, wyzeValues } from './api';
import { busy, IngestTab, mediaTime } from './IngestTab';

const source = (name: string, label: string, over: Partial<SourceInfo> = {}): SourceInfo => ({
  name,
  label,
  ongoing: false,
  configured: false,
  message: 'Not yet',
  seen: 0,
  ...over,
});

const item = (over: Partial<IngestItem> = {}): IngestItem => ({
  id: 1,
  sha256: 'ab'.repeat(32),
  source: 'upload',
  source_key: null,
  name: 'beach.jpg',
  kind: 'photo',
  original_ts: 1766629800,
  time_from: 'exif',
  added_ts: 1767000000,
  status: 'done',
  detail: '',
  stream: 'ingest-upload',
  outputs: [{ kind: 'video', stream: 'ingest-upload.video_v1', chunk_id: 'c1' }],
  ...over,
});

const view = (over: Partial<IngestView> = {}): IngestView => ({
  items: [],
  deletions: [],
  jobs: [],
  control: true,
  sources: [
    source('link', 'Link', { configured: true }),
    source('wyze', 'Wyze cameras', { ongoing: true, message: 'No cameras listed (ingest.wyze_cameras)' }),
    source('google_photos', 'Google Photos', { message: 'Not signed in' }),
    source('icloud_photos', 'iCloud Photos', { message: "Not yet: there's no iCloud Photos adapter." }),
  ],
  wyze: {
    cameras: [],
    rtsp_url: 'rtsp://127.0.0.1:8554',
    segment_s: 300,
    email: false,
    password: true,
    api_id: false,
    api_key: false,
  },
  ...over,
});

const serve = (v: IngestView) => server.use(http.get('/ingest/api/items', () => HttpResponse.json(v)));

describe('mediaTime', () => {
  const now = new Date(2026, 10, 6, 15, 0); // Fri 6 Nov 2026, 15:00
  const at = (...a: [number, number, number, number, number]) => new Date(...a).getTime() / 1000;
  it.each([
    [at(2026, 10, 6, 13, 9), 'today, 13:09'],
    [at(2026, 10, 5, 9, 5), 'yesterday, 09:05'],
    [
      at(2026, 8, 27, 15, 9),
      `${new Date(2026, 8, 27).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })}, 15:09`,
    ],
    // Media can be years old: another year says which (#2713)
    [
      at(2019, 7, 17, 14, 12),
      `${new Date(2019, 7, 17).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })}, 14:12`,
    ],
  ])('%s reads %j', (ts, words) => {
    expect(mediaTime(ts, now)).toBe(words);
  });
});

describe('the Ingest tab', () => {
  it('asks before deleting an item, and Cancel leaves it alone', async () => {
    const sent: unknown[] = [];
    serve(view({ items: [item()] }));
    server.use(
      http.post('/ingest/api/delete', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ok: true, deleted: true, pending: false });
      }),
    );
    renderPage(<IngestTab />);
    await userEvent.click(await screen.findByRole('button', { name: 'Delete beach.jpg' }));
    const confirm = await screen.findByRole('dialog', { name: 'Delete beach.jpg' });
    expect(
      within(confirm).getByText("Delete this and everything found in it? This can't be undone."),
    ).toBeInTheDocument();
    expect(sent).toEqual([]);
    await userEvent.click(within(confirm).getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByText('beach.jpg')).toBeInTheDocument();
    expect(sent).toEqual([]);
  });

  it('waits for a confirmed deletion and refreshes cached analysis views', async () => {
    const sent: unknown[] = [];
    let exists = true;
    let finish: () => void = () => {};
    server.use(
      http.get('/ingest/api/items', () =>
        HttpResponse.json(view({ items: exists ? [item({ status: 'converting' })] : [] })),
      ),
      http.post('/ingest/api/delete', async ({ request }) => {
        sent.push(await request.json());
        return new Promise<Response>(resolve => {
          finish = () => {
            exists = false;
            resolve(HttpResponse.json({ ok: true, deleted: true, pending: false }));
          };
        });
      }),
    );
    const { client } = renderPage(<IngestTab />);
    const cached = [
      ['people', 'card', 'Dana'],
      ['faces', 'unknown', 'groups'],
      ['voices', 'unknown', 'groups'],
      ['clones'],
      ['review', 'day', '2026-10-31', 'transcript'],
      ['live', 'speakers'],
      ['objects', 'overview'],
    ];
    for (const key of cached) client.setQueryData(key, { retained: true });
    const oldCard = {
      name: 'Dana',
      photo: null,
      shots: [],
      voices: [],
      last_seen: null,
      here_now: null,
      videos: [{ cam: 'ingest-upload', start: 10, end: 20, duration_s: 10, thumb: '/old.jpg', what: 'Deleted scene' }],
    };
    client.setQueryData(['people', 'card', 'Dana'], oldCard);
    client.setQueryData(['config'], { unrelated: true });
    await userEvent.click(await screen.findByRole('button', { name: 'Delete beach.jpg' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Delete beach.jpg' })).getByRole('button', {
        name: 'Delete',
      }),
    );
    await waitFor(() => expect(sent).toEqual([{ id: 1 }]));
    expect(screen.getByRole('button', { name: 'Delete beach.jpg' })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('Deleting');
    expect(screen.getByText('beach.jpg')).toBeInTheDocument();
    await act(async () => {
      finish();
    });
    await waitFor(() => expect(screen.queryByText('beach.jpg')).not.toBeInTheDocument());
    for (const key of cached) expect(client.getQueryData(key)).toBeUndefined();
    expect(client.getQueryData(['config'])).toEqual({ unrelated: true });
    expect(screen.getByText('Nothing brought in yet')).toBeInTheDocument();
  });

  it('automatically finishes pending cleanup after the keeper grace without another click', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    let removed = false;
    let ready = false;
    const sent: unknown[] = [];
    server.use(
      http.get('/ingest/api/items', () =>
        removed
          ? HttpResponse.json({ error: 'List unavailable' }, { status: 503 })
          : HttpResponse.json(view({ items: [item()] })),
      ),
      http.post('/ingest/api/delete', async ({ request }) => {
        sent.push(await request.json());
        removed = true;
        return HttpResponse.json({ ok: true, deleted: true, pending: !ready });
      }),
    );
    const { client } = renderPage(<IngestTab />);
    await user.click(await screen.findByRole('button', { name: 'Delete beach.jpg' }));
    await user.click(
      within(await screen.findByRole('dialog', { name: 'Delete beach.jpg' })).getByRole('button', { name: 'Delete' }),
    );
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Delete beach.jpg' })).not.toBeInTheDocument());
    expect(screen.getByText('Deleting…')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Retry cleanup for beach.jpg' })).not.toBeInTheDocument();
    expect(client.getQueryData<IngestView>(INGEST_KEY)?.items).toEqual([]);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(28_000);
    });
    expect(screen.getByText('Deleting…')).toBeInTheDocument();
    // Analysis can still be winding down before the actual purge runs on a later poll.
    client.setQueryData(['people', 'card', 'Dana'], { name: 'Dana', videos: [{ what: 'Not yet purged' }] });
    client.setQueryData(['voices', 'unknown', 'groups'], { groups: [{ id: 7 }] });
    ready = true;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });
    expect(screen.getByText('Deleted beach.jpg.')).toBeInTheDocument();
    expect(client.getQueryData(['people', 'card', 'Dana'])).toBeUndefined();
    expect(client.getQueryData(['voices', 'unknown', 'groups'])).toBeUndefined();
    expect(screen.queryByText('Deleting…')).not.toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(sent[0]).toEqual({ id: 1 });
    expect(sent.slice(1).length).toBeGreaterThan(0);
    for (const request of sent.slice(1)) expect(request).toEqual({ id: 1, poll: true });
  });

  it('automatically polls persisted cleanup notices and offers Retry after a polling error', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    let sent: unknown;
    serve(view({ deletions: [{ id: 7, name: 'old.mp4', detail: 'Storage cleanup is pending.' }] }));
    server.use(
      http.post('/ingest/api/delete', async ({ request }) => {
        sent = await request.json();
        return HttpResponse.json({ error: 'Cleanup is unavailable. Please retry.' }, { status: 503 });
      }),
    );
    renderPage(<IngestTab />);
    expect(await screen.findByText('Deleting…')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Retry cleanup for old.mp4' })).not.toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(await screen.findByRole('status')).toHaveTextContent('Cleanup is unavailable. Please retry.');
    expect(sent).toEqual({ id: 7, poll: true });
    expect(screen.getByRole('button', { name: 'Retry cleanup for old.mp4' })).toBeEnabled();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('offers Retry after the two-minute cleanup polling window and resumes without a new confirmation', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    let ready = false;
    let complete = false;
    server.use(
      http.get('/ingest/api/items', () =>
        HttpResponse.json(
          view({ deletions: complete ? [] : [{ id: 7, name: 'old.mp4', detail: '', blocked: false }] }),
        ),
      ),
      http.post('/ingest/api/delete', () => {
        complete = ready;
        return HttpResponse.json({ ok: true, deleted: true, pending: !ready });
      }),
    );
    const { client } = renderPage(<IngestTab />);
    expect(await screen.findByText('Deleting…')).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
      await client.invalidateQueries({ queryKey: INGEST_KEY });
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(screen.getByRole('button', { name: 'Retry cleanup for old.mp4' })).toBeEnabled();
    expect(screen.queryByText('Deleted old.mp4.')).not.toBeInTheDocument();
    ready = true;
    await user.click(screen.getByRole('button', { name: 'Retry cleanup for old.mp4' }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByText('Deleted old.mp4.')).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('does not automatically retry a persisted permanent cleanup block', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    let polls = 0;
    serve(
      view({
        deletions: [{ id: 7, name: 'old.mp4', detail: 'Stored media metadata must be repaired.', blocked: true }],
      }),
    );
    server.use(
      http.post('/ingest/api/delete', () => {
        polls++;
        return HttpResponse.json({ ok: true, deleted: true, pending: true });
      }),
    );
    renderPage(<IngestTab />);
    expect(await screen.findByText('Deletion needs manual attention.')).toBeInTheDocument();
    expect(await screen.findByText('Stored media metadata must be repaired.')).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120_000);
    });
    expect(polls).toBe(0);
    expect(screen.queryByRole('button', { name: 'Retry cleanup for old.mp4' })).not.toBeInTheDocument();
    expect(screen.queryByText('Deleted old.mp4.')).not.toBeInTheDocument();
  });

  it('stops after a permanent polling conflict and preserves the blocked notice', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    let polls = 0;
    server.use(
      http.get('/ingest/api/items', () =>
        polls
          ? HttpResponse.json({ error: 'List unavailable' }, { status: 503 })
          : HttpResponse.json(view({ deletions: [{ id: 7, name: 'old.mp4', detail: '' }] })),
      ),
      http.post('/ingest/api/delete', () => {
        polls++;
        return HttpResponse.json(
          { error: 'Stored media metadata must be repaired.', permanent: true },
          { status: 409 },
        );
      }),
    );
    const { client } = renderPage(<IngestTab />);
    expect(await screen.findByText('Deleting…')).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(screen.getByText('Deletion needs manual attention.')).toBeInTheDocument();
    expect(await screen.findByText('Stored media metadata must be repaired.')).toBeInTheDocument();
    expect(client.getQueryData<IngestView>(INGEST_KEY)?.deletions?.[0]?.blocked).toBe(true);
    expect(screen.queryByRole('button', { name: 'Retry cleanup for old.mp4' })).not.toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(polls).toBe(1);
  });

  it('shows a permanent initial deletion refusal as manual attention without another Delete action', async () => {
    let blocked = false;
    server.use(
      http.get('/ingest/api/items', () =>
        HttpResponse.json(
          view(
            blocked
              ? {
                  deletions: [
                    { id: 1, name: 'beach.jpg', detail: 'Stored media metadata must be repaired.', blocked: true },
                  ],
                }
              : { items: [item()] },
          ),
        ),
      ),
      http.post('/ingest/api/delete', () => {
        blocked = true;
        return HttpResponse.json(
          { error: 'Stored media metadata must be repaired.', permanent: true },
          { status: 409 },
        );
      }),
    );
    renderPage(<IngestTab />);
    await userEvent.click(await screen.findByRole('button', { name: 'Delete beach.jpg' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Delete beach.jpg' })).getByRole('button', { name: 'Delete' }),
    );
    expect(await screen.findByText('Deletion needs manual attention.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Delete beach.jpg' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Retry cleanup for beach.jpg' })).not.toBeInTheDocument();
  });

  it('keeps an item and reports a failed deletion so it can be retried', async () => {
    serve(view({ items: [item({ status: 'failed', detail: 'Analysis failed' })] }));
    server.use(
      http.post('/ingest/api/delete', () =>
        HttpResponse.json({ error: 'Purge is unavailable. Please retry.' }, { status: 503 }),
      ),
    );
    renderPage(<IngestTab />);
    await userEvent.click(await screen.findByRole('button', { name: 'Delete beach.jpg' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Delete beach.jpg' })).getByRole('button', {
        name: 'Delete',
      }),
    );
    expect(await screen.findByRole('status')).toHaveTextContent('Purge is unavailable. Please retry.');
    expect(screen.getByText('beach.jpg')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete beach.jpg' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeEnabled();
  });

  it('points to Settings for sources instead of listing them (#1331)', async () => {
    serve(view());
    renderPage(<IngestTab />);
    expect(screen.getByRole('heading', { name: 'Ingest', level: 1 })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'network cameras' })).toHaveAttribute('href', '/settings#network-cameras');
    expect(screen.getByRole('link', { name: 'photo libraries' })).toHaveAttribute('href', '/settings#imported-media');
    expect(await screen.findByText('Nothing brought in yet')).toBeInTheDocument();
    expect(screen.queryByText('Google Photos')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Wyze email')).not.toBeInTheDocument();
  });

  it('shows each item with its time and where the time came from, and retries a failed one', async () => {
    let retried: unknown;
    serve(
      view({
        items: [
          item(),
          item({
            id: 2,
            name: 'talk.m4a',
            kind: 'audio',
            status: 'failed',
            detail: 'ffmpeg failed',
            time_from: 'upload',
          }),
        ],
      }),
    );
    server.use(
      http.post('/ingest/api/retry', async ({ request }) => {
        retried = await request.json();
        return HttpResponse.json({ job: { id: 1, what: 'retry 2', status: 'queued', error: '', item: null } });
      }),
    );
    renderPage(<IngestTab />);
    const beach = (await screen.findByText('beach.jpg')).closest('li') as HTMLElement;
    expect(beach).toHaveTextContent(`photo, added here, taken ${mediaTime(1766629800)}`);
    expect(within(beach).getByLabelText('Status: Added')).toBeInTheDocument();
    expect(within(beach).queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
    const talk = screen.getByText('talk.m4a').closest('li') as HTMLElement;
    expect(talk).toHaveTextContent(`audio, added ${mediaTime(1766629800)}`); // no time of its own: said once
    expect(talk).toHaveTextContent('ffmpeg failed');
    await userEvent.click(within(talk).getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(retried).toEqual({ id: 2 }));
  });

  it('says where an item came from in words, and keeps a long name readable (#2689)', async () => {
    const link = 'https://example.invalid/a/very/long/path/to/a/video/that/does/not/fit?v=abc123';
    serve(
      view({
        items: [
          item({ id: 1, name: 'front-door.mp4', kind: 'video', source: 'wyze' }),
          item({ id: 2, name: link, kind: 'video', source: 'link' }),
          item({ id: 3, name: 'old.jpg', source: 'someday_source' }),
        ],
      }),
    );
    renderPage(<IngestTab />);
    const row = async (name: string) => (await screen.findByText(name)).closest('li') as HTMLElement;
    expect(await row('front-door.mp4')).toHaveTextContent('video, from Wyze cameras,');
    expect(await row(link)).toHaveTextContent('video, from a link,');
    expect(await row('old.jpg')).toHaveTextContent('photo, from someday_source,'); // a source the view doesn't list
    expect(screen.getByText(link)).toHaveAttribute('title', link);
  });

  it("says a failed source job in words, with a way to where it's fixed (#2689)", async () => {
    serve(
      view({
        jobs: [
          { id: 1, what: 'poll google_photos', status: 'failed', error: 'Sign in again.', item: null },
          { id: 2, what: 'poll wyze', status: 'failed', error: 'The bridge is down.', item: null },
          {
            id: 3,
            what: 'link https://example.invalid/v',
            status: 'failed',
            error: "couldn't fetch the link: 404 Client Error: Not Found for url",
            item: null,
          },
          { id: 4, what: 'link https://example.invalid/w', status: 'running', error: '', item: null },
        ],
      }),
    );
    renderPage(<IngestTab />);
    const google = await screen.findByText(/Checking Google Photos: Sign in again\./);
    expect(within(google).getByRole('link', { name: 'Fix it in Settings' })).toHaveAttribute(
      'href',
      expect.stringMatching(/#imported-media$/),
    );
    const wyze = screen.getByText(/Checking Wyze cameras: The bridge is down\./);
    expect(within(wyze).getByRole('link', { name: 'Fix it in Settings' })).toHaveAttribute(
      'href',
      expect.stringMatching(/#network-cameras$/),
    );
    const link = screen.getByText('Fetching https://example.invalid/v:').parentElement as HTMLElement;
    expect(link).toHaveTextContent("That link didn't load"); // in plain words (#2712)
    expect(within(link).queryByRole('link')).not.toBeInTheDocument(); // a link's failure isn't fixed in Settings
    expect(screen.getByText('Fetching https://example.invalid/w: working')).toBeInTheDocument();
  });

  it('keeps Retry in words on a failed row and Delete a quiet icon on every row (#2715)', async () => {
    serve(view({ items: [item(), item({ id: 2, name: 'talk.m4a', status: 'failed', detail: 'ffmpeg failed' })] }));
    renderPage(<IngestTab />);
    const talk = (await screen.findByText('talk.m4a')).closest('li') as HTMLElement;
    expect(within(talk).getByRole('button', { name: 'Retry' })).toHaveTextContent('Retry');
    for (const name of ['beach.jpg', 'talk.m4a']) {
      const remove = screen.getByRole('button', { name: `Delete ${name}` });
      expect(remove).toHaveTextContent(''); // an icon: its name is its label
      expect(remove).toHaveAttribute('title', 'Delete');
    }
  });

  it.each([
    [
      "yt-dlp couldn't get it: ERROR: [generic] Unable to download webpage: HTTP Error 404: Not Found",
      "That link didn't load",
    ],
    ['not a photo, video, or audio file (Invalid data found when processing input)', "This isn't a photo, video"],
    ['bigger than 4096 MB (ingest.max_mb)', 'Raise ingest.max_mb in Settings'],
    ["ffmpeg couldn't convert it: moov atom not found", "It couldn't be converted"],
    ['Upload interrupted. Please retry.', 'The upload was cut off. Add it again.'],
  ])('says %j in plain words, the original under Details (#2712)', async (detail, plain) => {
    serve(view({ items: [item({ status: 'failed', detail })] }));
    renderPage(<IngestTab />);
    const row = (await screen.findByText('beach.jpg')).closest('li') as HTMLElement;
    expect(row).toHaveTextContent(plain);
    const details = row.querySelector('details') as HTMLDetailsElement;
    expect(details).not.toHaveAttribute('open');
    expect(details).toHaveTextContent(detail);
  });

  it('shows a failure it has no words for as it is', async () => {
    serve(view({ items: [item({ status: 'failed', detail: 'the moon is in the wrong phase' })] }));
    renderPage(<IngestTab />);
    const row = (await screen.findByText('beach.jpg')).closest('li') as HTMLElement;
    expect(row).toHaveTextContent('the moon is in the wrong phase');
    expect(row.querySelector('details')).toBeNull();
  });

  it('groups the list by what needs you: in progress, needs attention, then added (#2716)', async () => {
    serve(
      view({
        items: [
          item({ id: 1, name: 'new.jpg', status: 'converting' }),
          item({ id: 2, name: 'old.jpg' }),
          item({ id: 3, name: 'bad.mov', status: 'failed', detail: 'ffmpeg failed' }),
        ],
        deletions: [{ id: 9, name: 'gone.mp4', detail: '' }],
      }),
    );
    renderPage(<IngestTab />);
    const group = async (name: string) => screen.findByRole('region', { name });
    const names = async (name: string) =>
      [...(await group(name)).querySelectorAll('li')].map(li => li.querySelector('p')?.textContent);
    expect(await names('In progress')).toEqual(['new.jpg']);
    expect(await names('Needs attention')).toEqual(['bad.mov', 'gone.mp4']);
    expect(await names('Added')).toEqual(['old.jpg']);
    const headings = screen.getAllByRole('heading', { level: 3 }).map(h => h.textContent);
    expect(headings).toEqual(['In progress 1', 'Needs attention 2', 'Added 1']);
  });

  it('shows the newest ten added, and the rest on Show all (#2716)', async () => {
    const many = Array.from({ length: 12 }, (_, k) => item({ id: k + 1, name: `photo-${k + 1}.jpg` }));
    serve(view({ items: many }));
    renderPage(<IngestTab />);
    const added = await screen.findByRole('region', { name: 'Added' });
    expect(added.querySelectorAll('li')).toHaveLength(10);
    await userEvent.click(within(added).getByRole('button', { name: 'Show all 12' }));
    expect(added.querySelectorAll('li')).toHaveLength(12);
    expect(screen.queryByRole('region', { name: 'In progress' })).toBeNull(); // an empty group isn't shown
  });

  it("says what's working and what needs attention, and shows a working bar on a row coming in (#2718)", async () => {
    serve(
      view({
        items: [
          item({ id: 1, name: 'new.jpg', status: 'converting' }),
          item({ id: 2, name: 'next.jpg', status: 'queued' }),
          item({ id: 3, name: 'bad.mov', status: 'failed', detail: 'ffmpeg failed' }),
          item({ id: 4, name: 'old.jpg' }),
        ],
      }),
    );
    renderPage(<IngestTab />);
    expect(await screen.findByText('2 working · 1 need attention')).toBeInTheDocument();
    const coming = (await screen.findByText('new.jpg')).closest('li') as HTMLElement;
    expect(within(coming).getByRole('progressbar', { name: 'Converting…' })).toBeInTheDocument();
    const added = screen.getByText('old.jpg').closest('li') as HTMLElement;
    expect(within(added).queryByRole('progressbar')).toBeNull();
  });

  it.each([
    [{ state: 'waiting' } as const, 'Reading it for words and pictures…', false],
    [{ state: 'retrying' } as const, "Couldn't read it this time; trying again soon.", false],
    [
      { state: 'failed', error: 'ffmpeg pipe was not created' } as const,
      "Couldn't read it: ffmpeg pipe was not created",
      false,
    ],
    [{ state: 'done', frames: 30, captions: 2, lines: 41 } as const, '2 captions · 41 lines heard', true],
    [{ state: 'done', frames: 1, captions: 1, lines: 0 } as const, '1 caption', false],
    [{ state: 'done', frames: 1, captions: 0, lines: 0 } as const, 'Nothing in it to describe or hear', false],
  ])('says what came of an added item (%j) (#2719)', async (analysis, words, review) => {
    serve(view({ items: [item({ analysis })] }));
    renderPage(<IngestTab />);
    const row = (await screen.findByText('beach.jpg')).closest('li') as HTMLElement;
    expect(row).toHaveTextContent(words);
    const link = within(row).queryByRole('link', { name: 'Open in Review' });
    if (review) expect(link).toHaveAttribute('href', expect.stringMatching(/#.*day=\d{4}-\d{2}-\d{2}.*t=1766629800/));
    else expect(link).toBeNull();
  });

  it('says nothing of reading an item still coming in, or from a server without it', async () => {
    serve(view({ items: [item({ status: 'converting', analysis: null }), item({ id: 2, name: 'old.jpg' })] }));
    renderPage(<IngestTab />);
    expect(await screen.findByText('old.jpg')).toBeInTheDocument();
    expect(screen.queryByText(/Reading it|heard|caption|Couldn't read/)).toBeNull();
  });

  it('shows a retry failure on the item', async () => {
    serve(view({ items: [item({ status: 'failed', detail: 'Conversion failed' })] }));
    server.use(
      http.post('/ingest/api/retry', () =>
        HttpResponse.json({ error: 'Temporarily unavailable. Please retry.' }, { status: 503 }),
      ),
    );
    renderPage(<IngestTab />);
    await userEvent.click(await screen.findByRole('button', { name: 'Retry' }));
    const row = screen.getByText('beach.jpg').closest('li') as HTMLElement;
    expect(await within(row).findByRole('status')).toHaveTextContent('Temporarily unavailable. Please retry.');
  });

  it('uploads each chosen file as its own request, named, and says which were already here', async () => {
    const got: { name: string | null; body: string }[] = [];
    serve(view());
    server.use(
      http.post('/ingest/api/upload', async ({ request }) => {
        const name = new URL(request.url).searchParams.get('name');
        got.push({ name, body: await request.text() });
        return HttpResponse.json({ item: item({ name: name ?? '' }), new: name !== 'dup.jpg' });
      }),
    );
    renderPage(<IngestTab />);
    const files = [
      new File(['one'], 'a b.jpg', { type: 'image/jpeg' }),
      new File(['two'], 'dup.jpg', { type: 'image/jpeg' }),
    ];
    await userEvent.upload(screen.getByLabelText('Choose files'), files);
    expect(await screen.findByText('Added 1 file; 1 was already here.')).toBeInTheDocument();
    expect(got).toEqual([
      { name: 'a b.jpg', body: 'one' },
      { name: 'dup.jpg', body: 'two' },
    ]);
    await userEvent.upload(screen.getByLabelText('Choose files'), files);
    await waitFor(() =>
      expect(got).toEqual([
        { name: 'a b.jpg', body: 'one' },
        { name: 'dup.jpg', body: 'two' },
        { name: 'a b.jpg', body: 'one' },
        { name: 'dup.jpg', body: 'two' },
      ]),
    );
  });

  it('says which item a file that was already here matched, and shows it (#2720)', async () => {
    const kept = item({ id: 7, name: 'beach day.jpg' });
    serve(view({ items: [kept] }));
    server.use(http.post('/ingest/api/upload', () => HttpResponse.json({ item: kept, new: false })));
    renderPage(<IngestTab />);
    const row = (await screen.findByText('beach day.jpg')).closest('li') as HTMLElement;
    row.scrollIntoView = vi.fn();
    await userEvent.upload(screen.getByLabelText('Choose files'), [
      new File(['x'], 'copy.jpg', { type: 'image/jpeg' }),
    ]);
    expect(await screen.findByText('1 file was already here.')).toBeInTheDocument(); // not "Added 0 files"
    const said = await screen.findByText(/copy\.jpg was already here as beach day\.jpg/);
    await userEvent.click(within(said).getByRole('button', { name: 'Show it' }));
    expect(row.scrollIntoView).toHaveBeenCalled();
    expect(row).toHaveFocus();
  });

  it('uploads a batch as a queue and keeps going past a failure (#2721)', async () => {
    serve(view());
    server.use(
      http.post('/ingest/api/upload', ({ request }) => {
        const name = new URL(request.url).searchParams.get('name') ?? '';
        return name === 'bad.jpg'
          ? HttpResponse.json({ error: 'Choose a nonempty file up to 4096 MB.' }, { status: 413 })
          : HttpResponse.json({ item: item({ name }), new: true });
      }),
    );
    renderPage(<IngestTab />);
    await userEvent.upload(screen.getByLabelText('Choose files'), [
      new File(['1'], 'one.jpg', { type: 'image/jpeg' }),
      new File([''], 'bad.jpg', { type: 'image/jpeg' }),
      new File(['3'], 'three.jpg', { type: 'image/jpeg' }),
    ]);
    expect(await screen.findByText('Added 2 files; 1 failed.')).toBeInTheDocument();
    const queue = screen.getByRole('list', { name: 'Upload queue' });
    const rows = within(queue).getAllByRole('listitem');
    expect(rows.map(r => r.textContent)).toEqual([
      'one.jpgadded',
      'bad.jpgfailed: Choose a nonempty file up to 4096 MB.',
      'three.jpgadded',
    ]);
  });

  it('takes files dropped anywhere on the page, outlining it while they are dragged (#2721)', async () => {
    const got: (string | null)[] = [];
    serve(view());
    server.use(
      http.post('/ingest/api/upload', ({ request }) => {
        const name = new URL(request.url).searchParams.get('name');
        got.push(name);
        return HttpResponse.json({ item: item({ name: name ?? '' }), new: true });
      }),
    );
    renderPage(<IngestTab />);
    await screen.findByText('Add media');
    const files = [new File(['x'], 'dropped.jpg', { type: 'image/jpeg' })];
    const dataTransfer = { files, types: ['Files'] };
    fireEvent.dragEnter(document.body, { dataTransfer });
    expect(await screen.findByText('Drop to add')).toBeInTheDocument();
    fireEvent.drop(document.body, { dataTransfer });
    expect(screen.queryByText('Drop to add')).not.toBeInTheDocument();
    expect(await screen.findByText('Added 1 file.')).toBeInTheDocument();
    expect(got).toEqual(['dropped.jpg']);
    fireEvent.dragEnter(document.body, { dataTransfer: { files: [], types: ['text/plain'] } }); // dragged text isn't a file
    expect(screen.queryByText('Drop to add')).not.toBeInTheDocument();
  });

  it("shows an upload's refusal", async () => {
    serve(view());
    server.use(
      http.post('/ingest/api/upload', () =>
        HttpResponse.json({ error: 'Choose a nonempty file up to 4096 MB.' }, { status: 413 }),
      ),
    );
    renderPage(<IngestTab />);
    await userEvent.upload(screen.getByLabelText('Choose files'), new File([''], 'empty.jpg', { type: 'image/jpeg' }));
    expect(await screen.findByText('empty.jpg: Choose a nonempty file up to 4096 MB.')).toBeInTheDocument();
  });

  it('sends a link and shows when the server refuses it', async () => {
    const sent: unknown[] = [];
    serve(view());
    server.use(
      http.post('/ingest/api/link', async ({ request }) => {
        const body = (await request.json()) as { url: string };
        sent.push(body);
        return body.url.startsWith('http')
          ? HttpResponse.json(
              { job: { id: 1, what: 'link', status: 'queued', error: '', item: null } },
              { status: 202 },
            )
          : HttpResponse.json({ error: 'a link starts with http:// or https://' }, { status: 400 });
      }),
    );
    renderPage(<IngestTab />);
    expect(screen.getByText('Add media')).toBeInTheDocument(); // files and links share one card
    const box = screen.getByRole('textbox', { name: /Add from a link/ });
    await userEvent.type(box, '  https://example.com/v  ');
    await userEvent.click(screen.getByRole('button', { name: 'Add' }));
    expect(await screen.findByText('Fetching it in the background.')).toBeInTheDocument();
    expect(box).toHaveValue('');
    await userEvent.type(box, 'ftp://x');
    await userEvent.click(screen.getByRole('button', { name: 'Add' }));
    expect(await screen.findByText('a link starts with http:// or https://')).toBeInTheDocument();
    expect(sent).toEqual([{ url: 'https://example.com/v' }, { url: 'ftp://x' }]);
  });
});

describe('busy', () => {
  it('is true while an item or a job is still being worked on', () => {
    expect(busy(undefined)).toBe(false);
    expect(busy(view({ items: [item()] }))).toBe(false);
    expect(busy(view({ items: [item({ status: 'converting' })] }))).toBe(true);
    expect(busy(view({ jobs: [{ id: 1, what: 'x', status: 'running', error: '', item: null }] }))).toBe(true);
    expect(busy(view({ jobs: [{ id: 1, what: 'x', status: 'failed', error: 'no', item: null }] }))).toBe(false);
  });
});

describe('wyzeValues', () => {
  it('splits cameras at commas and new lines and leaves blank credentials out', () => {
    expect(
      wyzeValues({ cameras: 'a,\n b ,,', rtsp_url: ' rtsp://x ', email: ' ', password: 'pw', api_id: '', api_key: '' }),
    ).toEqual({ 'ingest.wyze_cameras': ['a', 'b'], 'ingest.wyze_rtsp_url': 'rtsp://x', 'ingest.wyze_password': 'pw' });
  });
});
