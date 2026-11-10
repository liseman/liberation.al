import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { BoxLayerProps } from '../camera/BoxLayer';
import { EngineContext } from './context';
import { ReviewEngine } from './engine';
import { ReviewBoxes } from './ReviewBoxes';

vi.mock('../camera/BoxLayer', () => ({
  BoxLayer: ({ boxes, onPick }: BoxLayerProps) => (
    <>
      {boxes?.map(box => (
        <button type="button" key={box.id} onClick={() => onPick?.({ ...box, left: 0, top: 0, width: 1, height: 1 })}>
          {box.label}
        </button>
      ))}
    </>
  ),
}));
afterEach(() => vi.restoreAllMocks());

const frameAt = (seconds: number) => {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  canvas.dataset.capturedAt = String(seconds);
  canvas.dataset.frameMetadata = JSON.stringify({
    capturedAtUs: seconds * 1e6,
    width: 640,
    height: 360,
    flip: false,
    profileStatus: 'absent',
    nativeGeometry: { width: 640, height: 360 },
  });
  return canvas;
};

it.each([
  [true, ['Blue mug']],
  [false, ['sofa', 'Blue mug']],
])('with known only %s, Objects draws %j', async (knownOnly, labels) => {
  server.use(
    http.get('/review/boxes', () =>
      HttpResponse.json({
        boxes: [
          {
            ts: 50,
            until: 62,
            id: 'label:sofa:0',
            kind: 'label',
            box: [0.6, 0.2, 1, 0.6],
            label: 'sofa',
            tone: 'unknown',
          },
          { ts: 58, id: 'object:1', kind: 'object', box: [0.4, 0.4, 0.6, 0.6], label: 'Blue mug', tone: 'known' },
        ],
      }),
    ),
  );
  const canvas = frameAt(59.9);
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.ui.setState({ t: 80, avail: ['cam0'] });
  renderPage(
    <EngineContext value={engine}>
      <ReviewBoxes cam="cam0" kinds={['objects']} knownOnly={knownOnly} getCanvas={() => canvas} onFace={vi.fn()} />
    </EngineContext>,
  );
  await screen.findByRole('button', { name: 'Blue mug' });
  expect(screen.getAllByRole('button').map(button => button.textContent)).toEqual(labels);
});

it('draws enabled pet and object evidence at the displayed frame without invoking face naming', async () => {
  server.use(
    http.get('/review/boxes', () =>
      HttpResponse.json({
        boxes: [
          { ts: 59.9, id: 'pet:1', kind: 'pet', box: [0.1, 0.1, 0.3, 0.4], label: 'Mia · cat', tone: 'known' },
          { ts: 58, id: 'object:1', kind: 'object', box: [0.4, 0.4, 0.6, 0.6], label: 'Blue mug', tone: 'known' },
          { ts: 59.9, id: 'face:1', kind: 'face', box: [0.6, 0.1, 0.8, 0.3], label: 'Dana', tone: 'known' },
        ],
      }),
    ),
  );
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  canvas.dataset.capturedAt = '59.9';
  canvas.dataset.frameMetadata = JSON.stringify({
    capturedAtUs: 59_900_000,
    width: 640,
    height: 360,
    flip: false,
    profileStatus: 'absent',
    nativeGeometry: { width: 640, height: 360 },
  });
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.ui.setState({ t: 80, avail: ['cam0'] });
  const onFace = vi.fn();
  renderPage(
    <EngineContext value={engine}>
      <ReviewBoxes cam="cam0" kinds={['pets', 'objects']} getCanvas={() => canvas} onFace={onFace} />
    </EngineContext>,
  );
  await userEvent.click(await screen.findByRole('button', { name: 'Mia · cat' }));
  expect(screen.getByRole('button', { name: 'Blue mug' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Dana' })).not.toBeInTheDocument();
  expect(onFace).not.toHaveBeenCalled();
});

it('loads and picks the displayed frame from the prior minute instead of the requested playhead minute', async () => {
  const starts: string[] = [];
  server.use(
    http.get('/review/boxes', ({ request }) => {
      const start = new URL(request.url).searchParams.get('start') ?? '';
      starts.push(start);
      return HttpResponse.json({
        boxes:
          start === '0'
            ? [
                {
                  ts: 59.9,
                  id: 'face:still',
                  kind: 'face',
                  box: [0.25, 0.25, 0.5, 0.75],
                  label: "Who's this?",
                  tone: 'unknown',
                  group: 5,
                  track: 'run:cam0:7',
                },
              ]
            : [],
      });
    }),
  );
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  canvas.dataset.capturedAt = '59.9';
  canvas.dataset.frameMetadata = JSON.stringify({
    capturedAtUs: 59_900_000,
    width: 640,
    height: 360,
    flip: false,
    profileStatus: 'absent',
    nativeGeometry: { width: 640, height: 360 },
  });
  const drawImage = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage,
  } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,cGljdHVyZQ==');
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.ui.setState({ t: 80, avail: ['cam0'] });
  const onFace = vi.fn();
  renderPage(
    <EngineContext value={engine}>
      <ReviewBoxes cam="cam0" getCanvas={() => canvas} onFace={onFace} />
    </EngineContext>,
  );
  await userEvent.click(await screen.findByRole('button', { name: "Who's this?" }));
  expect(starts).toContain('0');
  expect(onFace).toHaveBeenCalledWith({
    source: { camera: 'cam0', at: 59.9, track: 'run:cam0:7', group: 5 },
    image: 'data:image/jpeg;base64,cGljdHVyZQ==',
  });
  expect(drawImage).toHaveBeenCalledWith(canvas, 160, 90, 160, 180, 0, 0, 160, 180);
});

it("names an unknown person's body from its track's face, drawing only the body", async () => {
  const person = { kind: 'person', label: "Who's this?", tone: 'unknown', track: '7', group: 5 };
  server.use(
    http.get('/review/boxes', ({ request }) =>
      HttpResponse.json({
        boxes:
          new URL(request.url).searchParams.get('start') === '0'
            ? [
                { ts: 59.8, id: 'person:7', box: [0.1, 0.1, 0.9, 0.9], ...person },
                { ts: 59.9, id: 'person:7', box: [0.1, 0.1, 0.9, 0.9], ...person },
                {
                  ts: 59.6,
                  id: 'face:#3',
                  kind: 'face',
                  box: [0.25, 0.25, 0.5, 0.75],
                  label: "Who's this?",
                  tone: 'unknown',
                  group: 5,
                  track: '7',
                },
              ]
            : [],
      }),
    ),
  );
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  canvas.dataset.capturedAt = '59.9';
  canvas.dataset.frameMetadata = JSON.stringify({
    capturedAtUs: 59_900_000,
    width: 640,
    height: 360,
    flip: false,
    profileStatus: 'absent',
    nativeGeometry: { width: 640, height: 360 },
  });
  const drawImage = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage,
  } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,cGljdHVyZQ==');
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.ui.setState({ t: 80, avail: ['cam0'] });
  const onFace = vi.fn();
  renderPage(
    <EngineContext value={engine}>
      <ReviewBoxes cam="cam0" getCanvas={() => canvas} onFace={onFace} />
    </EngineContext>,
  );
  const body = await screen.findByRole('button', { name: "Who's this?" }); // one: the face hides behind its body
  await userEvent.click(body);
  expect(onFace).toHaveBeenCalledWith({
    source: { camera: 'cam0', at: 59.9, track: '7', group: 5 },
    image: 'data:image/jpeg;base64,cGljdHVyZQ==',
  });
  expect(drawImage).toHaveBeenCalledWith(canvas, 160, 90, 160, 180, 0, 0, 160, 180); // the face, not the body
});

it('fetches the boxes of the minute a scrub settles on, not of every minute it crosses', async () => {
  const asked: number[] = [];
  server.use(
    http.get('/review/boxes', ({ request }) => {
      asked.push(Number(new URL(request.url).searchParams.get('start')));
      return HttpResponse.json({ boxes: [] });
    }),
  );
  let canvas = frameAt(30);
  const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
  engine.ui.setState({ t: 30, avail: ['cam0'] });
  renderPage(
    <EngineContext value={engine}>
      <ReviewBoxes cam="cam0" getCanvas={() => canvas} onFace={vi.fn()} />
    </EngineContext>,
  );
  await vi.waitFor(() => expect(asked).toEqual([0]));
  // A scrub through 20 minutes, a step every few milliseconds.
  for (let minute = 1; minute <= 20; minute++) {
    canvas = frameAt(minute * 60 + 30);
    act(() => engine.ui.setState({ t: minute * 60 + 30 }));
  }
  await vi.waitFor(() => expect(asked).toEqual([0, 1200]));
  await new Promise(r => setTimeout(r, 300));
  expect(asked).toEqual([0, 1200]);
});

describe('"Not <name>" on a matched box (#2607)', () => {
  const show = (boxes: object[]) => {
    server.use(http.get('/review/boxes', () => HttpResponse.json({ boxes })));
    const canvas = frameAt(59.9);
    const engine = new ReviewEngine({ cams: ['cam0'], fetchNotes: async () => ({ cams: {} }) });
    engine.ui.setState({ t: 59.9, avail: ['cam0'] });
    renderPage(
      <EngineContext value={engine}>
        <ReviewBoxes cam="cam0" kinds={['people', 'pets', 'objects']} getCanvas={() => canvas} onFace={vi.fn()} />
      </EngineContext>,
    );
  };
  const dana = {
    ts: 59.9,
    id: 'person:7',
    kind: 'person',
    box: [0.2, 0.2, 0.4, 0.8],
    label: 'Dana',
    tone: 'known',
    name: 'Dana',
    track: '7',
  };

  it("takes a person's track off them, draws the box unnamed at once, and Undo puts it back", async () => {
    const sent: unknown[] = [];
    const undone: unknown[] = [];
    server.use(
      http.post('/people/correct', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ok: true, undo: 'tok-1', card: null });
      }),
      http.post('/people/undo', async ({ request }) => {
        undone.push(await request.json());
        return HttpResponse.json({ ok: true, errors: [], people: [] });
      }),
    );
    show([dana]);
    await userEvent.click(await screen.findByRole('button', { name: 'Dana' }));
    await userEvent.click(screen.getByRole('button', { name: 'Not Dana' }));
    expect(sent).toEqual([{ name: 'Dana', kind: 'track', id: 'cam0:7@59.90', action: 'remove' }]);
    expect(await screen.findByRole('button', { name: "Who's this?" })).toBeInTheDocument(); // before any refetch
    await userEvent.click(await screen.findByRole('button', { name: 'Undo' }));
    await vi.waitFor(() => expect(undone).toEqual([{ undo: 'tok-1' }]));
    expect(await screen.findByRole('button', { name: 'Dana' })).toBeInTheDocument();
  });

  it("marks a pet's track not that pet, showing its species, and Undo takes that back", async () => {
    const sent: unknown[] = [];
    server.use(
      http.post('/pets/sightings/not', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ marked: 1 });
      }),
    );
    show([
      {
        ts: 59.9,
        id: 'pet:p1',
        kind: 'pet',
        box: [0.1, 0.1, 0.3, 0.4],
        label: 'Mia · cat',
        tone: 'known',
        name: 'Mia',
        species: 'cat',
        pet_id: 'mia',
        track: 'p1',
      },
    ]);
    await userEvent.click(await screen.findByRole('button', { name: 'Mia · cat' }));
    expect(screen.queryByRole('button', { name: 'Card' })).toBeNull(); // a person's card only
    await userEvent.click(screen.getByRole('button', { name: 'Not Mia' }));
    expect(await screen.findByRole('button', { name: 'cat' })).toBeInTheDocument();
    await userEvent.click(await screen.findByRole('button', { name: 'Undo' }));
    await vi.waitFor(() =>
      expect(sent).toEqual([
        { track_ids: ['p1'], pet: 'mia', not: true },
        { track_ids: ['p1'], pet: 'mia', not: false },
      ]),
    );
  });

  it('offers nothing on a thing, which has no match to take back', async () => {
    show([{ ts: 58, id: 'object:1', kind: 'object', box: [0.4, 0.4, 0.6, 0.6], label: 'Blue mug', tone: 'known' }]);
    await userEvent.click(await screen.findByRole('button', { name: 'Blue mug' }));
    expect(screen.queryByRole('button', { name: /^Not / })).toBeNull();
  });

  it('keeps the box named and says so when the correction fails', async () => {
    server.use(
      http.post('/people/correct', () =>
        HttpResponse.json({ ok: false, error: "That person in view isn't Dana." }, { status: 404 }),
      ),
    );
    show([dana]);
    await userEvent.click(await screen.findByRole('button', { name: 'Dana' }));
    await userEvent.click(screen.getByRole('button', { name: 'Not Dana' }));
    expect(await screen.findByText("That person in view isn't Dana.")).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Dana' })).toBeInTheDocument();
  });
});
