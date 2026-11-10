import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { BoxLayerProps } from '../camera/BoxLayer';
import type { BoxPick } from '../camera/boxes';
import { Stage } from '../livetab/Stage';
import { EngineContext } from './context';
import { ReviewEngine } from './engine';
import { type MakeVideo, Player } from './Player';

vi.mock('../camera/BoxLayer', () => ({
  BoxLayer: ({ boxes, onPick }: BoxLayerProps) => (
    <>
      {boxes?.map(box => (
        <button type="button" key={box.id} onClick={() => onPick?.({ ...box, left: 0, top: 0, width: 10, height: 10 })}>
          {box.label}
        </button>
      ))}
    </>
  ),
}));
vi.mock('../camera/LiveBoxes', () => ({
  LiveBoxes: ({ onPick }: { onPick: (value: BoxPick) => void }) => (
    <button
      type="button"
      onClick={() =>
        onPick({
          id: 'live-face',
          kind: 'face',
          tone: 'unknown',
          label: "Who's this?",
          box: [0.25, 0.25, 0.5, 0.75],
          faceSource: { camera: 'cam0', at: 100.1, track: 'run:cam0:7', group: 5 },
          left: 0,
          top: 0,
          width: 10,
          height: 10,
        })
      }
    >
      Who's this?
    </button>
  ),
}));
vi.mock('../camera/LiveCam', async original => ({
  ...(await original<typeof import('../camera/LiveCam')>()),
  LiveCam: () => <canvas className="live-canvas" width={640} height={360} data-cam="cam0" data-captured-at="100.25" />,
}));
const IMAGE = 'data:image/jpeg;base64,cGljdHVyZQ==';
let posts: unknown[];
const drawImage = vi.fn();
let rows: Record<string, unknown>[];
const makeVideo: MakeVideo = canvas => {
  canvas.width = 640;
  canvas.height = 360;
  canvas.dataset.capturedAt = '100.25';
  canvas.dataset.frameMetadata = JSON.stringify({
    capturedAtUs: 100_250_000,
    width: 640,
    height: 360,
    flip: false,
    profileStatus: 'absent',
    nativeGeometry: { width: 640, height: 360 },
  });
  return {
    src: '',
    preload: 'auto',
    currentTime: 0,
    readyState: 4,
    duration: 10,
    error: null,
    seeking: false,
    paused: true,
    ended: false,
    play: async () => {},
    pause: () => {},
    load: () => {},
    close: () => {},
    getAttribute: name => canvas.getAttribute(name),
    removeAttribute: name => canvas.removeAttribute(name),
    toggleAttribute: (name, force) => canvas.toggleAttribute(name, force),
    addEventListener: () => {},
    removeEventListener: () => {},
  };
};
beforeEach(() => {
  posts = [];
  rows = [
    {
      ts: 100.1,
      id: 'face:?1',
      kind: 'face',
      box: [0.25, 0.25, 0.5, 0.75],
      label: "Who's this?",
      tone: 'unknown',
      group: 5,
      track: 'run:cam0:7',
    },
  ];
  history.replaceState(null, '', '/review');
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage,
  } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue(IMAGE);
  server.use(
    http.get('/vision', () => HttpResponse.json({ cameras: {} })),
    http.get('/ambient', () => HttpResponse.json({ devices: {} })),
    http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }] })),
    http.get('/review/boxes', () => HttpResponse.json({ boxes: rows })),
    http.get('/transcript', () => HttpResponse.json({ entries: [] })),
    http.post('/people/claim', async ({ request }) => {
      posts.push(await request.json());
      return HttpResponse.json({ name: 'Dana' });
    }),
  );
});
afterEach(() => {
  vi.restoreAllMocks();
  drawImage.mockClear();
  history.replaceState(null, '', '/');
});
function show(mode: 'Live' | 'Review', cams = ['cam0']) {
  history.replaceState(null, '', mode === 'Live' ? '/' : '/review');
  const engine = new ReviewEngine({ cams, fetchNotes: async () => ({ cams: {} }) });
  engine.ui.setState({ t: 100.9, avail: cams });
  renderPage(
    mode === 'Live' ? (
      <Stage last={{}} live={['cam0']} />
    ) : (
      <EngineContext value={engine}>
        <Player makeVideo={makeVideo} />
      </EngineContext>
    ),
  );
  return engine;
}
it.each(['Live', 'Review'] as const)(
  '%s opens the shared crop picker and names an existing person in place',
  async mode => {
    const engine = show(mode);
    await userEvent.click(await screen.findByRole('button', { name: "Who's this?" }));
    expect(screen.getByRole('img', { name: 'Selected face' })).toHaveAttribute('src', IMAGE);
    // Playback and newer boxes after opening must not change the selected evidence.
    for (const canvas of document.querySelectorAll<HTMLCanvasElement>('canvas.live-canvas'))
      canvas.dataset.capturedAt = '101.5';
    vi.mocked(HTMLCanvasElement.prototype.toDataURL).mockReturnValue('data:image/jpeg;base64,bmV3');
    act(() => engine.ui.setState({ t: 101.5 }));
    await userEvent.click(await screen.findByRole('combobox', { name: 'Person' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Dana' }));
    expect(posts).toEqual([]);
    await userEvent.click(screen.getByRole('button', { name: 'Use selected person' }));
    await waitFor(() =>
      expect(posts).toEqual([
        {
          kind: 'face',
          id: 5,
          name: 'Dana',
          face_source: { camera: 'cam0', at: mode === 'Review' ? 100.25 : 100.1, track: 'run:cam0:7', group: 5 },
          face_image: IMAGE,
        },
      ]),
    );
    expect(location.pathname).toBe(mode === 'Live' ? '/' : '/review');
    expect(drawImage).toHaveBeenCalledWith(expect.any(HTMLCanvasElement), 160, 90, 160, 180, 0, 0, 160, 180);
  },
);
it.each(['Live', 'Review'] as const)(
  '%s adds a new person through the same evidence API without navigation',
  async mode => {
    show(mode);
    await userEvent.click(await screen.findByRole('button', { name: "Who's this?" }));
    await userEvent.type(await screen.findByRole('combobox', { name: 'Person' }), 'Morgan');
    await userEvent.click(await screen.findByRole('option', { name: 'Add “Morgan”' }));
    await waitFor(() =>
      expect(posts).toEqual([
        {
          kind: 'face',
          id: 5,
          name: 'Morgan',
          face_source: { camera: 'cam0', at: mode === 'Review' ? 100.25 : 100.1, track: 'run:cam0:7', group: 5 },
          face_image: IMAGE,
        },
      ]),
    );
    expect(location.pathname).toBe(mode === 'Live' ? '/' : '/review');
  },
);

it('Review uses the active double-buffer canvas rather than the hidden standby picture', async () => {
  show('Review');
  const [standby, active] = document.querySelectorAll<HTMLCanvasElement>('canvas.live-canvas');
  if (!standby || !active) throw new Error('missing fixture canvases');
  standby.setAttribute('data-standby', '');
  active.removeAttribute('data-standby');
  active.width = 1280;
  active.height = 720;
  active.dataset.capturedAt = '100.6';
  active.dataset.frameMetadata = JSON.stringify({
    capturedAtUs: 100_600_000,
    width: 1280,
    height: 720,
    flip: false,
    profileStatus: 'absent',
    nativeGeometry: { width: 1280, height: 720 },
  });
  await userEvent.click(await screen.findByRole('button', { name: "Who's this?" }));
  await userEvent.type(await screen.findByRole('combobox', { name: 'Person' }), 'Morgan');
  await userEvent.click(await screen.findByRole('option', { name: 'Add “Morgan”' }));
  await waitFor(() =>
    expect(posts).toEqual([
      {
        kind: 'face',
        id: 5,
        name: 'Morgan',
        face_source: { camera: 'cam0', at: 100.6, track: 'run:cam0:7', group: 5 },
        face_image: IMAGE,
      },
    ]),
  );
  expect(drawImage).toHaveBeenCalledWith(active, 320, 180, 320, 360, 0, 0, 320, 360);
});
it('Review refuses a face whose displayed pixels lose their capture timestamp before the pick', async () => {
  show('Review');
  const pick = await screen.findByRole('button', { name: "Who's this?" });
  for (const canvas of document.querySelectorAll<HTMLCanvasElement>('canvas.live-canvas'))
    delete canvas.dataset.capturedAt;
  await userEvent.click(pick);
  expect(await screen.findByRole('alert')).toHaveTextContent('No displayed face picture');
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(posts).toEqual([]);
  expect(location.pathname).toBe('/review');
});
it('Review picks the actual displayed track when the requested playhead reuses its visual box id', async () => {
  rows.push({ ...rows[0], ts: 100.8, track: 'run:cam0:8' });
  show('Review');
  await userEvent.click(await screen.findByRole('button', { name: "Who's this?" }));
  await userEvent.type(await screen.findByRole('combobox', { name: 'Person' }), 'Morgan');
  await userEvent.click(await screen.findByRole('option', { name: 'Add “Morgan”' }));
  await waitFor(() =>
    expect(posts).toEqual([
      {
        kind: 'face',
        id: 5,
        name: 'Morgan',
        face_source: { camera: 'cam0', at: 100.25, track: 'run:cam0:7', group: 5 },
        face_image: IMAGE,
      },
    ]),
  );
  expect(location.pathname).toBe('/review');
});
it('Review does not invent a source track when historical evidence lacks one', async () => {
  rows[0] = { ...rows[0], track: undefined };
  show('Review');
  await userEvent.click(await screen.findByRole('button', { name: "Who's this?" }));
  expect(await screen.findByRole('alert')).toHaveTextContent('displayed face evidence is not available');
  expect(posts).toEqual([]);
  expect(location.pathname).toBe('/review');
});

it('Review retains the selected camera source when the large tile switches while the picker is open', async () => {
  server.use(
    http.get('/review/boxes', ({ request }) => {
      const cam = new URL(request.url).searchParams.get('cam');
      return HttpResponse.json({ boxes: rows.map(row => ({ ...row, track: `run:${cam}:7` })) });
    }),
  );
  const engine = show('Review', ['cam0', 'cam1']);
  await userEvent.click(screen.getByRole('button', { name: /Show the .* large/ }));
  await userEvent.click(await screen.findByRole('button', { name: "Who's this?" }));
  act(() => engine.ui.setState({ camera: 'cam0', cameraIntent: 1 }));
  expect(screen.getByRole('img', { name: 'Selected face' })).toHaveAttribute('src', IMAGE);
  await userEvent.type(await screen.findByRole('combobox', { name: 'Person' }), 'Morgan');
  await userEvent.click(await screen.findByRole('option', { name: 'Add “Morgan”' }));
  await waitFor(() =>
    expect(posts).toEqual([
      {
        kind: 'face',
        id: 5,
        name: 'Morgan',
        face_source: { camera: 'cam1', at: 100.25, track: 'run:cam1:7', group: 5 },
        face_image: IMAGE,
      },
    ]),
  );
});
