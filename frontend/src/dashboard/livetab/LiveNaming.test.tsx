import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { newQueryClient, Providers } from '../../shared/Providers';
import { server } from '../../test/server';
import { setClockForTests, setGatewayForTests } from '../camera/gateway';
import { controlVisibility, fakeBitmaps, fakeFrames, fakeGateway } from '../camera/testFakes';
import { liveKey } from '../live/feed';
import { reduce } from '../live/reduce';
import { useLiveStatus } from '../live/status';
import { useRoute } from '../router';
import { LiveTab } from './LiveTab';

const NOW = 1_793_000_000;
let frames: ReturnType<typeof fakeFrames>;
let net: ReturnType<typeof fakeGateway>;
let visibility: ReturnType<typeof controlVisibility>;
let posts: unknown[];
let named: string | null;

beforeEach(() => {
  history.replaceState(null, '', '/?view=keep#camera=cam0');
  useRoute.setState({ tab: 'live' });
  vi.stubGlobal('NAMES', { assistant: 'Assistant', owner: 'Owner', cams: ['cam0'], hotkeys: false });
  visibility = controlVisibility();
  frames = fakeFrames();
  fakeBitmaps();
  net = fakeGateway();
  setGatewayForTests(net.connect);
  setClockForTests(() => NOW);
  useLiveStatus.setState({ now: NOW });
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    x: 0,
    y: 0,
    top: 0,
    left: 0,
    right: 640,
    bottom: 360,
    width: 640,
    height: 360,
    toJSON() {
      return {};
    },
  });
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,cGljdHVyZQ==');
  posts = [];
  named = null;
  server.use(
    http.get('/people', () =>
      HttpResponse.json({
        people: [
          { name: 'Dana', voices: [], here_now: { cam: 'cam0', since: NOW - 10 } },
          ...(named && named !== 'Dana' ? [{ name: named, voices: [], here_now: { cam: 'cam0', since: NOW } }] : []),
        ],
      }),
    ),
    http.post('/people/claim', async ({ request }) => {
      const body = (await request.json()) as { name: string };
      posts.push(body);
      named = body.name;
      return HttpResponse.json({ name: body.name });
    }),
    http.get('/transcript', () => HttpResponse.json({ entries: [] })),
    http.get('/speakers', () => HttpResponse.json({ speakers: [], merged: {} })),
    http.get('/live/happening', () => HttpResponse.json({ items: [] })),
    http.get('/stats', () => HttpResponse.json({})),
    http.get('/devices', () => HttpResponse.json({ control: true, mics: [], cameras: [], notes: {} })),
    http.get('/config/voice_ref', () => HttpResponse.json({ exists: false })),
    http.get('/config', () => HttpResponse.json({ ok: true, control: true, sections: [] })),
  );
});
afterEach(() => {
  visibility.restore();
  setClockForTests(() => Date.now() / 1000);
  history.replaceState(null, '', '/');
  vi.unstubAllGlobals();
});

async function live(face = true, otherCamera = false) {
  if (otherCamera) history.replaceState(null, '', '/?view=keep#camera=cam1');
  if (otherCamera)
    vi.stubGlobal('NAMES', { assistant: 'Assistant', owner: 'Owner', cams: ['cam0', 'cam1'], hotkeys: false });
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false } });
  client.setQueryData(liveKey('ambient'), { ts: NOW });
  client.setQueryData(liveKey('scene'), [
    { cam: 'cam0', ts: NOW, caption: 'Kitchen frame' },
    ...(otherCamera ? [{ cam: 'cam1', ts: NOW, caption: 'Hall frame' }] : []),
  ]);
  client.setQueryData(liveKey('vision'), {});
  client.setQueryData(liveKey('transcript'), reduce('transcript', undefined, { entries: [], reset: true }));
  client.setQueryData(liveKey('speakers'), reduce('speakers', undefined, { speakers: [], merged: {} }));
  render(
    <Providers client={client} test>
      <LiveTab />
    </Providers>,
  );
  await waitFor(() => expect(net.open().length).toBeGreaterThan(0));
  act(() => {
    for (const connection of net.open()) {
      connection.hello();
      connection.streams({ 'cam0.video': 1, 'cam1.video': 2 });
      connection.live(1);
    }
    const cameras = net.open().filter(connection => connection.last?.type === 'live');
    if (!cameras.length) throw new Error('camera fixture did not subscribe');
    for (const connection of cameras) {
      const selected = connection.last?.stream === (otherCamera ? 2 : 1);
      connection.jpeg(NOW * 1e6);
      if (face && selected)
        connection.overlay(NOW * 1e6, [
          { id: otherCamera ? 'run:cam1:7' : 'run:cam0:7', kind: 'face', label: 'Face 5', xyxy: [0.1, 0.1, 0.4, 0.6] },
        ]);
    }
  });
  await waitFor(() => expect(document.querySelector('canvas.live-canvas')).not.toHaveClass('off'));
  frames.step(performance.now() + 100);
}

it.each([
  ['large', 'Dana'],
  ['large', 'Morgan'],
  ['here', 'Dana'],
  ['here', 'Morgan'],
  ['setup', 'Dana'],
  ['setup', 'Morgan'],
])('%s names a current face as %s without leaving Live or waiting for another frame', async (entry, name) => {
  await live();
  if (entry === 'large') await userEvent.click(screen.getByRole('button', { name: "Who's this?" }));
  else if (entry === 'here')
    await userEvent.click(
      within(screen.getByRole('region', { name: 'Here now' })).getByRole('link', { name: "Who's this?" }),
    );
  else await userEvent.click(await screen.findByRole('link', { name: 'Open Who’s this?' }));
  const dialog = await screen.findByRole('dialog', { name: "Who's this?" });
  const input = await within(dialog).findByRole('combobox', { name: 'Person' });
  if (name === 'Dana') {
    await userEvent.click(input);
    await userEvent.click(await screen.findByRole('option', { name: 'Dana' }));
    await userEvent.click(within(dialog).getByRole('button', { name: 'Use selected person' }));
  } else {
    await userEvent.type(input, name);
    await userEvent.click(await screen.findByRole('option', { name: 'Add “Morgan”' }));
  }
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  expect(posts).toEqual([
    {
      kind: 'face',
      id: 5,
      name,
      face_source: { camera: 'cam0', at: NOW, track: 'run:cam0:7', group: 5 },
      face_image: 'data:image/jpeg;base64,cGljdHVyZQ==',
    },
  ]);
  expect(location.pathname + location.search + location.hash).toBe('/?view=keep#camera=cam0');
  expect(useRoute.getState().tab).toBe('live');
  frames.step(performance.now() + 200);
  // The new name reaches the large view's face once the claim's update renders: under load that's after this line.
  expect(
    await within(screen.getByRole('figure', { name: 'cam0, large' })).findByRole('button', { name }),
  ).toBeInTheDocument();
  expect(screen.getByText('Kitchen frame')).toBeInTheDocument();
});

it.each(['here', 'setup'])('%s falls back to People when no current unknown face is available', async entry => {
  await live(false);
  const link =
    entry === 'here'
      ? within(screen.getByRole('region', { name: 'Here now' })).getByRole('link', { name: "Who's this?" })
      : await screen.findByRole('link', { name: 'Open Who’s this?' });
  await userEvent.click(link);
  expect(location.pathname).toBe('/people');
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(posts).toEqual([]);
});

it('names the selected network camera using only that camera’s displayed face evidence', async () => {
  await live(true, true);
  await waitFor(() => expect(document.querySelector('canvas[data-cam="cam1"]')).not.toHaveClass('off'));
  await userEvent.click(
    within(screen.getByRole('region', { name: 'Here now' })).getByRole('link', { name: "Who's this?" }),
  );
  const dialog = await screen.findByRole('dialog', { name: "Who's this?" });
  await userEvent.click(await within(dialog).findByRole('combobox', { name: 'Person' }));
  await userEvent.click(await screen.findByRole('option', { name: 'Dana' }));
  await userEvent.click(within(dialog).getByRole('button', { name: 'Use selected person' }));
  await waitFor(() => expect(posts).toHaveLength(1));
  expect(posts[0]).toMatchObject({
    name: 'Dana',
    face_source: { camera: 'cam1', at: NOW, track: 'run:cam1:7', group: 5 },
  });
  expect(screen.getByRole('figure', { name: 'cam1, large' })).toHaveTextContent('Hall frame');
  expect(location.pathname + location.search + location.hash).toBe('/?view=keep#camera=cam1');
  expect(useRoute.getState().tab).toBe('live');
});
