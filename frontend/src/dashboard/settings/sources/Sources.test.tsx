import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, describe, expect, it } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import type { IngestView, SourceInfo } from '../../ingest/api';
import { CONFIG_KEY } from '../model';
import { ImportedMedia, NetworkCameras } from './Sources';

const source = (name: string, label: string, over: Partial<SourceInfo> = {}): SourceInfo => ({
  name,
  label,
  ongoing: false,
  configured: false,
  message: 'Not yet',
  seen: 0,
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

const when = (ts: number) => new Date(ts * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

const serve = (v: IngestView) => server.use(http.get('/ingest/api/items', () => HttpResponse.json(v)));

const rowNames = (card: HTMLElement) =>
  within(card)
    .queryAllByRole('listitem')
    .map(li => li.getAttribute('aria-label'));

// Each configured network camera's card reads capture's word on it, for its preview and status.
beforeEach(() => server.use(http.get('/vision', () => HttpResponse.json({}))));

describe("Settings' sources", () => {
  it('shows camera sources on Network cameras and the rest under imported media, never links', async () => {
    serve(view());
    renderPage(
      <>
        <NetworkCameras />
        <ImportedMedia />
      </>,
    );
    const cameras = screen.getByRole('region', { name: 'Network cameras' });
    const media = screen.getByRole('region', { name: 'Accounts and apps' });
    expect(cameras).toHaveAttribute('id', 'network-cameras');
    expect(media).toHaveAttribute('id', 'imported-media');
    expect(await within(cameras).findByText('Wyze cameras')).toBeInTheDocument();
    expect(rowNames(cameras)).toEqual(['Wyze cameras']);
    expect(rowNames(media)).toEqual(['Google Photos', 'iCloud Photos']);
    const icloud = within(media).getByRole('listitem', { name: 'iCloud Photos' });
    expect(icloud).toHaveTextContent("Not set upNot yet: there's no iCloud Photos adapter.");
    expect(screen.queryByText('Link')).not.toBeInTheDocument();
  });

  it('lists what can be connected first, and the sources with no way in yet as one line', async () => {
    const later = { later: true, message: 'Not yet' };
    serve(
      view({
        sources: [
          source('icloud_photos', 'iCloud Photos', later),
          source('google_photos', 'Google Photos'),
          source('allsound', 'AllSound', later),
          source('thinking_hat', 'Thinking Hat', later),
        ],
      }),
    );
    renderPage(<ImportedMedia />);
    const media = screen.getByRole('region', { name: 'Accounts and apps' });
    expect(
      await within(media).findByText(/^Coming later: iCloud Photos, AllSound, and Thinking Hat\./),
    ).toBeInTheDocument();
    expect(rowNames(media)).toEqual(['Google Photos']);
    expect(within(media).getByRole('link', { name: 'upload their exports on Ingest' })).toHaveAttribute(
      'href',
      '/ingest',
    );
  });

  it('says so when there are no sources of a kind', async () => {
    serve(view({ sources: [source('google_photos', 'Google Photos')] }));
    renderPage(<NetworkCameras />);
    expect(await screen.findByText('No network camera sources.')).toBeInTheDocument();
  });

  it('checks an ongoing source that is set up for new items', async () => {
    let polled = '';
    serve(view({ sources: [source('wyze', 'Wyze cameras', { ongoing: true, configured: true, message: 'Ready' })] }));
    server.use(
      http.post('/ingest/api/sources/:name/poll', ({ params }) => {
        polled = String(params.name);
        return HttpResponse.json({ job: { id: 1, what: 'poll', status: 'queued', error: '', item: null } });
      }),
    );
    renderPage(<NetworkCameras />);
    await userEvent.click(await screen.findByRole('button', { name: 'Check now' }));
    expect(await screen.findByText('Checking for new items.')).toBeInTheDocument();
    expect(polled).toBe('wyze');
  });

  it('saves Wyze settings through /config, sending only the credentials typed in', async () => {
    let saved: unknown;
    serve(view());
    server.use(
      http.post('/config', async ({ request }) => {
        saved = await request.json();
        return HttpResponse.json({ ok: true, restarted: [] });
      }),
    );
    const { client } = renderPage(<NetworkCameras />);
    client.setQueryData(CONFIG_KEY, { stale: true });
    // The form is folded until it's asked for
    await userEvent.click(await screen.findByRole('button', { name: 'Set up Wyze cameras' }));
    expect(await screen.findByLabelText('Wyze password')).toHaveAttribute('placeholder', 'Saved; type to replace');
    await userEvent.type(screen.getByLabelText('Wyze email'), 'dana@example.com');
    await userEvent.type(screen.getByRole('textbox', { name: /Cameras/ }), 'front-door, garage');
    await userEvent.click(screen.getByRole('button', { name: 'Save Wyze settings' }));
    expect(await screen.findByText('Saved.')).toBeInTheDocument();
    expect(saved).toEqual({
      values: {
        'ingest.wyze_cameras': ['front-door', 'garage'],
        'ingest.wyze_rtsp_url': 'rtsp://127.0.0.1:8554',
        'ingest.wyze_email': 'dana@example.com',
      },
    });
    expect(screen.getByLabelText('Wyze email')).toHaveValue('');
    // Settings' copy of the config is out of date now, so Advanced reads it again.
    expect(client.getQueryState(CONFIG_KEY)?.isInvalidated).toBe(true);
  });

  it("doesn't offer to change Wyze settings or open the sign-in window from another computer", async () => {
    serve(view({ control: false }));
    renderPage(
      <>
        <NetworkCameras />
        <ImportedMedia />
      </>,
    );
    await userEvent.click(await screen.findByRole('button', { name: 'Set up Wyze cameras' }));
    expect(await screen.findByText('Wyze settings can only be changed from this computer.')).toBeInTheDocument();
    expect(screen.getByLabelText('Wyze email')).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Open sign-in window' })).not.toBeInTheDocument();
  });

  it('browses Google Photos and imports the ones picked', async () => {
    let imported: unknown;
    const photo = (key: string, taken_at: number) => ({
      key,
      name: `${key}.jpg`,
      taken_at,
      url: `https://lh3.googleusercontent.com/${key}=d`,
      kind: 'photo',
      thumb: `https://lh3.googleusercontent.com/${key}=w256-h256-c`,
      label: `Photo ${key}`,
    });
    serve(view({ sources: [source('google_photos', 'Google Photos', { configured: true, message: 'Signed in' })] }));
    server.use(
      http.get('/ingest/api/google_photos/albums', () =>
        HttpResponse.json({ albums: [{ id: 'A1', title: 'Lisbon' }] }),
      ),
      http.get('/ingest/api/google_photos/photos', ({ request }) =>
        HttpResponse.json({
          items:
            new URL(request.url).searchParams.get('album') === 'A1'
              ? [photo('P3', 1700000000)]
              : [photo('P1', 1700000000), photo('P2', 1700000100)],
        }),
      ),
      http.post('/ingest/api/google_photos/import', async ({ request }) => {
        imported = await request.json();
        return HttpResponse.json({ job: { id: 1, what: 'gp', status: 'queued', error: '', item: null } });
      }),
    );
    renderPage(<ImportedMedia />);
    await userEvent.click(await screen.findByRole('button', { name: 'Browse photos' }));
    expect(await screen.findByAltText('Photo P1')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Import/ })).toBeDisabled();
    await userEvent.click(screen.getByRole('checkbox', { name: when(1700000100) }));
    await userEvent.click(screen.getByRole('button', { name: 'Import 1' }));
    expect(await screen.findByText('Importing 1 item in the background.')).toBeInTheDocument();
    expect((imported as { items: { key: string }[] }).items.map(i => i.key)).toEqual(['P2']);
    await userEvent.selectOptions(await screen.findByRole('combobox', { name: 'Album' }), 'A1');
    expect(await screen.findByAltText('Photo P3')).toBeInTheDocument();
  });

  it('says why the sources could not be loaded', async () => {
    server.use(http.get('/ingest/api/items', () => HttpResponse.json({ error: 'down' }, { status: 500 })));
    renderPage(<ImportedMedia />);
    expect(await screen.findByText("Couldn't load sources")).toBeInTheDocument();
  });
});

it('shows native configured cameras even when ingest is unavailable', async () => {
  server.use(
    http.get('/ingest/api/items', () => HttpResponse.json({ error: 'Import worker unavailable' }, { status: 503 })),
    http.get('/network-cameras', () =>
      HttpResponse.json({
        capture_ready: false,
        cameras: [
          {
            id: 'native-test',
            name: 'Native test camera',
            source: 'rtsp',
            enabled: true,
            url_display: 'rtsp://192.168.40.8/[redacted]',
          },
        ],
      }),
    ),
  );
  renderPage(<NetworkCameras />);
  const card = screen.getByRole('region', { name: 'Network cameras' });
  expect(await within(card).findByText('Native test camera')).toBeInTheDocument();
  expect(within(card).getByRole('button', { name: 'Discover' })).toBeInTheDocument();
});
