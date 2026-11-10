import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { Activity, StrictMode, useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../test/render';
import { server } from '../test/server';
import type { MemoryObject, Segmentation, Sighting, Snapshot, Status } from './api';
import { cropLiveObject, handoffReviewFrame } from './handoff';
import { ObjectMemoryPage } from './ObjectMemory';
import { Remember } from './Remember';

const preview = vi.hoisted(() => ({ ready: true, capturedAt: 1700000000, image: 'data:image/jpeg;base64,PREVIEW' }));
vi.mock('../dashboard/camera/LiveCam', () => ({
  LiveCam: ({
    cam,
    alt,
    canvasRef,
  }: {
    cam: string;
    alt: string;
    canvasRef: { current: HTMLCanvasElement | null };
  }) => (
    <canvas
      role="img"
      aria-label={alt}
      data-cam={cam}
      ref={canvas => {
        if (canvas) {
          if (preview.ready) canvas.dataset.capturedAt = String(preview.capturedAt);
          canvas.toDataURL = vi.fn(() => preview.image);
        }
        canvasRef.current = canvas;
      }}
    />
  ),
}));

const NOW = Date.now() / 1000;

const status = (over: Partial<Status> = {}): Status => ({
  objects: 1,
  cameras: [
    { camera: 'cam0', available: true, stale: false, source: 'camera' },
    { camera: 'cam1', available: true, stale: true, source: 'camera' },
    { camera: 'cam2', available: true, stale: false, source: 'camera' },
  ],
  observer: { finished_at: NOW - 3 },
  observer_error: null,
  ...over,
});

const sighting = (over: Partial<Sighting> = {}): Sighting => ({
  camera: 'cam0',
  captured_at: NOW - 60,
  kind: 'manual_label',
  evidence: 'aaaa.jpg',
  bbox: [0.1, 0.1, 0.2, 0.2],
  stale: false,
  ...over,
});

const ssd: MemoryObject = {
  id: 'o1',
  name: 'my SSD',
  aliases: ['drive'],
  note: 'orange sticker',
  template: 'tttt.jpg',
  mode: 'visual',
  last_seen: sighting(),
};

const snapshot: Snapshot = {
  token: 'snap1',
  evidence: 'snap1.jpg',
  camera: 'cam2',
  captured_at: NOW,
  stale: false,
  source: 'camera',
};

const selection = (over: Partial<Segmentation> = {}): Segmentation => ({
  selection: 'selection1',
  snapshot: 'snap1',
  bbox: [0.12, 0.15, 0.4, 0.35],
  mask: 'mask:selection1',
  score: 0.95,
  model: 'edgetam-test-model',
  ...over,
});

function serve(objects: MemoryObject[] = [ssd], st: Status = status()) {
  server.use(
    http.get('/objects/api/status', () => HttpResponse.json(st)),
    http.get('/objects/api/objects', () => HttpResponse.json({ objects })),
  );
}

/** An EdgeTAM endpoint that records each request and answers with `answer` (a selection for that snapshot by default). */
function segments(answer?: (body: Record<string, unknown>) => Response | Promise<Response>) {
  const calls: Record<string, unknown>[] = [];
  server.use(
    http.post('/objects/api/segment', async ({ request }) => {
      const body = (await request.json()) as Record<string, unknown>;
      calls.push(body);
      if (answer) return answer(body);
      return HttpResponse.json(selection({ snapshot: String(body.snapshot) }));
    }),
  );
  return calls;
}

function clickSegment(canvas: HTMLElement, x = 50, y = 50) {
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 320, height: 180 }) as DOMRect;
  fireEvent.click(canvas, { clientX: x, clientY: y });
}

beforeEach(() => {
  preview.ready = true;
  preview.image = 'data:image/jpeg;base64,PREVIEW';
  server.use(http.post('/objects/api/cancel', () => HttpResponse.json({ ok: true, cancelled: true })));
});

describe('status line', () => {
  it('counts saved objects and cameras ready for capture', async () => {
    serve();
    renderPage(<ObjectMemoryPage />);
    expect(await screen.findByText('1 object · 2/3 cameras ready')).toBeInTheDocument();
  });

  it('says the observer errored over its last run', async () => {
    serve([ssd], status({ observer_error: 'RuntimeError: boom' }));
    renderPage(<ObjectMemoryPage />);
    expect(await screen.findByText(/matching unavailable$/)).toBeInTheDocument();
  });

  it("shows the server's reason when object memory is down", async () => {
    server.use(
      http.get('/objects/api/status', () =>
        HttpResponse.json({ error: 'Object memory is unavailable.' }, { status: 503 }),
      ),
      http.get('/objects/api/objects', () => HttpResponse.json({ objects: [] })),
    );
    renderPage(<ObjectMemoryPage />);
    expect(await screen.findByText('Object memory unavailable: Object memory is unavailable.')).toBeInTheDocument();
  });

  it('says so when nothing is remembered yet', async () => {
    serve([]);
    renderPage(<ObjectMemoryPage />);
    expect(await screen.findByText(/No objects remembered yet/)).toBeInTheDocument();
  });
});

describe('finding an object', () => {
  it('shows the answer, its evidence, and the earlier observations', async () => {
    serve();
    let asked = '';
    server.use(
      http.get('/objects/api/query', ({ request }) => {
        asked = new URL(request.url).searchParams.get('q') ?? '';
        return HttpResponse.json({
          status: 'last_seen',
          answer: 'You manually labeled my SSD in cam0.',
          object: ssd,
          evidence: sighting(),
          history: [sighting(), sighting({ camera: 'cam1', kind: 'visual_match', evidence: 'bbbb.jpg' })],
          uncertainty: 'Camera view only.',
        });
      }),
    );
    renderPage(<ObjectMemoryPage />);
    await userEvent.type(await screen.findByLabelText('Object question'), 'where is my SSD?');
    await userEvent.click(screen.getByRole('button', { name: 'Find last seen' }));

    expect(await screen.findByText('You manually labeled my SSD in cam0.')).toBeInTheDocument();
    expect(asked).toBe('where is my SSD?');
    expect(screen.getByRole('img', { name: /^Evidence: cam0 · .* · manually labeled$/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Earlier observations' }));
    // The first history entry is the evidence above, so only the second is listed as earlier.
    const earlier = screen.getAllByRole('link').filter(a => a.getAttribute('href')?.startsWith('/objects/evidence/'));
    expect(earlier.map(a => a.getAttribute('href'))).toEqual(['/objects/evidence/bbbb.jpg']);
  });

  it("shows the server's error as the answer", async () => {
    serve();
    server.use(
      http.get('/objects/api/query', () => HttpResponse.json({ error: 'Provide a question.' }, { status: 400 })),
    );
    renderPage(<ObjectMemoryPage />);
    await userEvent.type(await screen.findByLabelText('Object question'), 'x');
    await userEvent.click(screen.getByRole('button', { name: 'Find last seen' }));
    expect(await screen.findByText('Provide a question.')).toBeInTheDocument();
  });

  it("asks by the object's name from its Last seen button", async () => {
    serve();
    let asked = '';
    server.use(
      http.get('/objects/api/query', ({ request }) => {
        asked = new URL(request.url).searchParams.get('q') ?? '';
        return HttpResponse.json({ status: 'unknown', answer: 'No evidence.', objects: [] });
      }),
    );
    renderPage(<ObjectMemoryPage />);
    await userEvent.click(await screen.findByRole('button', { name: 'Last seen' }));
    expect(await screen.findByText('No evidence.')).toBeInTheDocument();
    expect(asked).toBe('my SSD');
    expect(screen.getByLabelText('Object question')).toHaveValue('my SSD');
  });
});

describe('remembering an object', () => {
  it('retains an imported frame while the configured camera list is still loading', async () => {
    let finishStatus!: () => void;
    const waiting = new Promise<void>(resolve => {
      finishStatus = resolve;
    });
    serve();
    server.use(
      http.get('/objects/api/status', async () => {
        await waiting;
        return HttpResponse.json(
          status({ cameras: [{ camera: 'cam3', available: true, stale: false, source: 'camera' }] }),
        );
      }),
      http.post('/objects/api/import', () => HttpResponse.json({ ...snapshot, camera: 'cam3', source: 'full_frame' })),
    );
    handoffReviewFrame('data:image/jpeg;base64,AAAA', 'cam3', NOW);
    renderPage(<ObjectMemoryPage />);
    try {
      expect(await screen.findByLabelText('Click an object to outline it')).toBeInTheDocument();
      expect(screen.getByText(/cam3 · captured .* · frozen, not live/)).toBeInTheDocument();
    } finally {
      finishStatus();
    }
    await screen.findByText('1 object · 1/1 cameras ready');
    expect(screen.getByLabelText('Click an object to outline it')).toBeInTheDocument();
    expect(screen.getByLabelText('Camera')).toHaveValue('cam3');
  });

  it('names each camera by its place in the select and the frozen caption, else by its id', async () => {
    const cameras = [
      { camera: 'cam0', name: 'living room', available: true, stale: false, source: 'camera' },
      { camera: 'cam1', available: true, stale: false, source: 'camera' },
      { camera: 'cam2', name: 'patio', available: true, stale: false, source: 'camera' },
    ];
    serve([ssd], status({ cameras }));
    server.use(http.post('/objects/api/import', () => HttpResponse.json(snapshot)));
    renderPage(<ObjectMemoryPage />);
    await screen.findByText(/cameras ready/);
    const options = within(screen.getByLabelText('Camera')).getAllByRole('option');
    expect(options.map(o => [o.getAttribute('value'), o.textContent])).toEqual([
      ['cam0', 'Living room'],
      ['cam1', 'cam1'],
      ['cam2', 'Patio'],
    ]);
    await userEvent.selectOptions(screen.getByLabelText('Camera'), 'cam2');
    const preview = screen.getByRole('img', { name: 'Live view from cam2' }).parentElement;
    expect(within(preview as HTMLElement).getByText('Patio')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Freeze view' }));
    expect(await screen.findByText(/^Patio · captured .* · frozen, not live/)).toBeInTheDocument();
  });

  it('previews the first configured camera when status replaces the initial camera list', async () => {
    serve([], status({ cameras: [{ camera: 'cam2', available: true, stale: false, source: 'camera' }] }));
    renderPage(<ObjectMemoryPage />);
    expect(await screen.findByRole('img', { name: 'Live view from cam2' })).toBeVisible();
    expect(screen.getByLabelText('Camera')).toHaveValue('cam2');
  });

  it('shows the selected live camera before Freeze and waits for its first picture', async () => {
    preview.ready = false;
    serve();
    renderPage(<ObjectMemoryPage />);
    expect(await screen.findByRole('img', { name: 'Live view from cam0' })).toBeVisible();
    await userEvent.selectOptions(screen.getByLabelText('Camera'), 'cam2');
    expect(screen.getByRole('img', { name: 'Live view from cam2' })).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Freeze view' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Wait for a picture');
    expect(screen.queryByLabelText('Click an object to outline it')).not.toBeInTheDocument();
  });

  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 320,
      height: 180,
    } as DOMRect);
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockImplementation(
      () =>
        ({
          clearRect() {},
          setTransform() {},
          drawImage() {},
          fillRect() {},
          strokeRect() {},
          beginPath() {},
          moveTo() {},
          lineTo() {},
          closePath() {},
          stroke() {},
        }) as unknown as CanvasRenderingContext2D,
    );
  });

  const frame = () => screen.getByLabelText('Click an object to outline it');
  const idle = () =>
    waitFor(() => expect(screen.queryByRole('status', { name: 'Processing this view' })).not.toBeInTheDocument());

  it('shows progress from the first Freeze click while capture is still pending', async () => {
    serve();
    let finish!: () => void;
    const pending = new Promise<void>(resolve => {
      finish = resolve;
    });
    server.use(
      http.post('/objects/api/import', async () => {
        await pending;
        return HttpResponse.json(snapshot);
      }),
    );
    renderPage(<ObjectMemoryPage />);
    await userEvent.click(await screen.findByRole('button', { name: 'Freeze view' }));
    const progressDuringCapture = screen.queryByRole('status', { name: 'Processing this view' });
    finish();
    await idle();
    expect(progressDuringCapture).not.toBeNull();
  });

  it('runs no EdgeTAM on a frozen view until the owner clicks, then outlines that point once and saves it', async () => {
    serve();
    let captured: unknown;
    let enrolled: unknown;
    const calls = segments();
    server.use(
      http.post('/objects/api/import', async ({ request }) => {
        captured = await request.json();
        return HttpResponse.json(snapshot);
      }),
      http.post('/objects/api/enroll', async ({ request }) => {
        enrolled = await request.json();
        return HttpResponse.json({ object: { ...ssd, id: 'o2', name: 'keys' } });
      }),
    );
    renderPage(<ObjectMemoryPage />);
    await screen.findByText(/cameras ready/);
    await userEvent.selectOptions(screen.getByLabelText('Camera'), 'cam2');
    await userEvent.click(screen.getByRole('button', { name: 'Freeze view' }));
    expect(await screen.findByText('Click the object, or draw a box around it, to outline it.')).toBeInTheDocument();
    expect(calls).toEqual([]);
    expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeDisabled();
    clickSegment(frame());
    await waitFor(() => expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeEnabled());
    expect(calls).toEqual([
      { snapshot: 'snap1', point: [expect.any(Number), expect.any(Number)], job: expect.any(String) },
    ]);
    const [x = -1, y = -1] = (calls[0]?.point ?? []) as number[];
    expect(x).toBeGreaterThan(0);
    expect(x).toBeLessThan(1);
    expect(y).toBeGreaterThan(0);
    expect(y).toBeLessThan(1);
    await userEvent.type(screen.getByLabelText(/Object name/), 'keys');
    await userEvent.type(
      screen.getByRole('combobox', { name: 'Other names (comma separated)' }),
      'key ring,house keys{Enter}spare keys',
    );
    expect(enrolled).toBeUndefined();
    await userEvent.click(screen.getByRole('button', { name: 'Remember selected object' }));
    await screen.findByText(/^Remembered\. Looking for conservative visual matches/);
    expect(captured).toEqual({
      kind: 'full_frame',
      camera: 'cam2',
      image: preview.image,
      captured_at: preview.capturedAt,
    });
    expect(calls).toHaveLength(1);
    expect(enrolled).toEqual({
      name: 'keys',
      camera: 'cam2',
      snapshot: 'snap1',
      selection: 'selection1',
      bbox: [0.12, 0.15, 0.4, 0.35],
      aliases: 'key ring, house keys, spare keys',
      note: '',
    });
    expect(screen.queryByLabelText('Click an object to outline it')).not.toBeInTheDocument();
  });

  it('finds the objects in view on request, and a click picks one to remember', async () => {
    serve();
    const clicks = segments();
    const finds: Record<string, unknown>[] = [];
    let enrolled: unknown;
    server.use(
      http.post('/objects/api/import', () => HttpResponse.json(snapshot)),
      http.post('/objects/api/find', async ({ request }) => {
        finds.push((await request.json()) as Record<string, unknown>);
        return HttpResponse.json({
          snapshot: 'snap1',
          ms: 1234,
          objects: [
            {
              ...selection({ selection: 'whole', mask: 'mask:whole' }),
              outline: [
                [
                  [0, 0],
                  [1, 0],
                  [1, 1],
                  [0, 1],
                ],
              ],
            },
            { ...selection({ selection: 'other', mask: 'mask:other' }), outline: [] },
          ],
        });
      }),
      http.post('/objects/api/enroll', async ({ request }) => {
        enrolled = await request.json();
        return HttpResponse.json({ object: { ...ssd, id: 'o3', name: 'lamp' } });
      }),
    );
    renderPage(<ObjectMemoryPage />);
    await screen.findByText(/cameras ready/);
    await userEvent.click(screen.getByRole('button', { name: 'Freeze view' }));
    await screen.findByText('Click the object, or draw a box around it, to outline it.');
    expect(finds).toEqual([]);
    await userEvent.click(screen.getByRole('button', { name: 'Find objects' }));
    expect(await screen.findByText('Found 2 objects in 1.2 s. Click one to select it.')).toBeInTheDocument();
    expect(finds).toEqual([{ snapshot: 'snap1', crop: expect.any(Array), job: expect.any(String) }]);
    clickSegment(frame());
    expect(await screen.findByText('Selected. Name it, then remember it.')).toBeInTheDocument();
    expect(clicks).toEqual([]);
    await userEvent.type(screen.getByLabelText(/Object name/), 'lamp');
    await userEvent.click(screen.getByRole('button', { name: 'Remember selected object' }));
    await screen.findByText(/^Remembered\./);
    expect(enrolled).toMatchObject({ snapshot: 'snap1', selection: 'whole', name: 'lamp' });
  });

  it("remembers the object picked last, even when an earlier pick's mask loads after it", async () => {
    serve();
    const half = (x0: number, x1: number): [number, number][][] => [
      [
        [x0, 0],
        [x1, 0],
        [x1, 1],
        [x0, 1],
      ],
    ];
    let enrolled: unknown;
    server.use(
      http.post('/objects/api/import', () => HttpResponse.json(snapshot)),
      http.post('/objects/api/find', () =>
        HttpResponse.json({
          snapshot: 'snap1',
          ms: 500,
          objects: [
            { ...selection({ selection: 'a', mask: 'slow-mask-a' }), outline: half(0, 0.5) },
            { ...selection({ selection: 'b', mask: 'mask-b' }), outline: half(0.5, 1) },
          ],
        }),
      ),
      http.post('/objects/api/enroll', async ({ request }) => {
        enrolled = await request.json();
        return HttpResponse.json({ object: { ...ssd, id: 'o4', name: 'mug' } });
      }),
    );
    // A's mask loads only when released: after B's
    let releaseA!: () => void;
    class Held {
      naturalWidth = 640;
      naturalHeight = 360;
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      #src = '';
      get src() {
        return this.#src;
      }
      set src(value: string) {
        this.#src = value;
        if (value.includes('slow-mask-a')) releaseA = () => this.onload?.();
        else queueMicrotask(() => this.onload?.());
      }
    }
    vi.stubGlobal('Image', Held);
    try {
      renderPage(<ObjectMemoryPage />);
      await screen.findByText(/cameras ready/);
      await userEvent.click(screen.getByRole('button', { name: 'Freeze view' }));
      await screen.findByText('Click the object, or draw a box around it, to outline it.');
      await userEvent.click(screen.getByRole('button', { name: 'Find objects' }));
      await screen.findByText(/^Found 2 objects/);
      clickSegment(frame(), 50); // A, on the left: its mask is still loading
      clickSegment(frame(), 250); // then B, on the right
      expect(await screen.findByText('Selected. Name it, then remember it.')).toBeInTheDocument();
      releaseA(); // A's mask comes in last
      await new Promise(resolve => setTimeout(resolve, 0));
      await userEvent.type(screen.getByLabelText(/Object name/), 'mug');
      await userEvent.click(screen.getByRole('button', { name: 'Remember selected object' }));
      await screen.findByText(/^Remembered\./);
      expect(enrolled).toMatchObject({ selection: 'b', name: 'mug' });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it.each(['success', 'error'] as const)(
    'fades the frozen image while EdgeTAM outlines a click and restores it on %s',
    async outcome => {
      serve();
      let finish!: () => void;
      const pending = new Promise<void>(resolve => {
        finish = resolve;
      });
      segments(async body => {
        await pending;
        return outcome === 'success'
          ? HttpResponse.json(selection({ snapshot: String(body.snapshot) }))
          : HttpResponse.json({ error: 'Segmentation unavailable.' }, { status: 503 });
      });
      server.use(http.post('/objects/api/import', () => HttpResponse.json(snapshot)));
      renderPage(<ObjectMemoryPage />);
      await userEvent.click(await screen.findByRole('button', { name: 'Freeze view' }));
      await screen.findByText(/Click the object/);
      clickSegment(frame());
      expect(await screen.findByRole('status', { name: 'Processing this view' })).toBeInTheDocument();
      expect(frame().parentElement).toHaveAttribute('aria-busy', 'true');
      expect(frame()).toHaveStyle({ opacity: '0.3', pointerEvents: 'none' });
      expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeDisabled();
      finish();
      await waitFor(() => expect(frame().parentElement).toHaveAttribute('aria-busy', 'false'));
      expect(frame()).toHaveStyle({ opacity: '1', pointerEvents: 'auto' });
      if (outcome === 'error')
        expect(screen.getByRole('alert')).toHaveTextContent('Could not outline that: Segmentation unavailable.');
    },
  );

  it('cannot save an outline whose mask preview failed to load', async () => {
    serve();
    segments(() => HttpResponse.json(selection({ mask: 'missing-mask' })));
    server.use(http.post('/objects/api/import', () => HttpResponse.json(snapshot)));
    renderPage(<ObjectMemoryPage />);
    await userEvent.click(await screen.findByRole('button', { name: 'Freeze view' }));
    await screen.findByText(/Click the object/);
    clickSegment(frame());
    await screen.findByText(/Could not outline that: Image unavailable/);
    expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeDisabled();
  });

  it('cancels the owned outline before replacing a frozen frame and ignores its stale result', async () => {
    serve();
    let releaseFirst!: () => void;
    const firstResponse = new Promise<void>(resolve => {
      releaseFirst = resolve;
    });
    const cancelled: string[] = [];
    let captures = 0;
    let enrolled: unknown;
    const calls = segments(async body => {
      if (body.snapshot === 'snap1') await firstResponse;
      return HttpResponse.json(selection({ snapshot: String(body.snapshot), selection: `selected-${body.snapshot}` }));
    });
    server.use(
      http.post('/objects/api/cancel', async ({ request }) => {
        cancelled.push(((await request.json()) as { job: string }).job);
        return HttpResponse.json({ ok: true, cancelled: true });
      }),
      http.post('/objects/api/import', () => HttpResponse.json({ ...snapshot, token: `snap${++captures}` })),
      http.post('/objects/api/enroll', async ({ request }) => {
        enrolled = await request.json();
        return HttpResponse.json({ object: ssd });
      }),
    );
    renderPage(<ObjectMemoryPage />);
    await userEvent.click(await screen.findByRole('button', { name: 'Freeze view' }));
    await screen.findByText(/Click the object/);
    clickSegment(frame());
    await waitFor(() => expect(calls).toHaveLength(1));
    await userEvent.click(screen.getByRole('button', { name: 'Freeze view' }));
    await waitFor(() => expect(cancelled).toEqual([calls[0]?.job]));
    await waitFor(() => expect(captures).toBe(2));
    await screen.findByText(/Click the object/);
    releaseFirst();
    clickSegment(frame());
    await waitFor(() => expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeEnabled());
    expect(calls.map(c => c.snapshot)).toEqual(['snap1', 'snap2']);
    await userEvent.type(screen.getByLabelText(/Object name/), 'keys');
    await userEvent.click(screen.getByRole('button', { name: 'Remember selected object' }));
    await screen.findByText(/^Remembered\. Looking/);
    expect(enrolled).toMatchObject({ snapshot: 'snap2', selection: 'selected-snap2' });
  });

  it('retries a failed cancellation before allowing a replacement capture', async () => {
    serve();
    let release!: () => void;
    const pending = new Promise<void>(resolve => {
      release = resolve;
    });
    let captures = 0;
    let cancels = 0;
    const calls = segments(async body => {
      await pending;
      return HttpResponse.json(selection({ snapshot: String(body.snapshot) }));
    });
    server.use(
      http.post('/objects/api/import', () => HttpResponse.json({ ...snapshot, token: `snap${++captures}` })),
      http.post('/objects/api/cancel', () =>
        ++cancels === 1
          ? HttpResponse.json({ error: 'termination unconfirmed' }, { status: 503 })
          : HttpResponse.json({ ok: true, cancelled: true }),
      ),
    );
    renderPage(<ObjectMemoryPage />);
    const freeze = await screen.findByRole('button', { name: 'Freeze view' });
    await userEvent.click(freeze);
    await screen.findByText(/Click the object/);
    clickSegment(frame());
    await waitFor(() => expect(calls).toHaveLength(1));
    await userEvent.click(freeze);
    await screen.findByText('termination unconfirmed');
    expect(captures).toBe(1);
    await userEvent.click(freeze);
    await waitFor(() => expect(captures).toBe(2));
    expect(cancels).toBe(2);
    release();
  });

  it('cancels a pending outline and clears the frozen view when the camera changes', async () => {
    serve();
    let release!: () => void;
    const pending = new Promise<void>(resolve => {
      release = resolve;
    });
    let returned!: () => void;
    const responseReturned = new Promise<void>(resolve => {
      returned = resolve;
    });
    const calls = segments(async body => {
      await pending;
      returned();
      return HttpResponse.json(selection({ snapshot: String(body.snapshot) }));
    });
    server.use(http.post('/objects/api/import', () => HttpResponse.json(snapshot)));
    renderPage(<ObjectMemoryPage />);
    await userEvent.click(await screen.findByRole('button', { name: 'Freeze view' }));
    await screen.findByText(/Click the object/);
    clickSegment(frame());
    await waitFor(() => expect(calls).toHaveLength(1));
    await userEvent.selectOptions(screen.getByLabelText('Camera'), 'cam1');
    expect(screen.queryByRole('status', { name: 'Processing this view' })).not.toBeInTheDocument();
    release();
    await responseReturned;
    expect(screen.queryByLabelText('Click an object to outline it')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Remember selected object' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Camera')).toHaveValue('cam1');
  });

  it('loads a spoken capture with its name filled in and waits for a click', async () => {
    serve();
    const calls = segments();
    server.use(
      http.get('/objects/api/draft', () =>
        HttpResponse.json({ draft: { snapshot, name: 'my wallet', saved_at: NOW } }),
      ),
    );
    renderPage(<ObjectMemoryPage />);
    await userEvent.click(await screen.findByRole('button', { name: 'Use what I said' }));
    await screen.findByText(/Click the object/);
    expect(screen.getByLabelText(/Object name/)).toHaveValue('my wallet');
    expect(screen.getByLabelText('Camera')).toHaveValue('cam2');
    expect(calls).toEqual([]);
  });

  it('says how to make a spoken capture when there is none', async () => {
    serve();
    server.use(http.get('/objects/api/draft', () => HttpResponse.json({ draft: null })));
    renderPage(<ObjectMemoryPage />);
    await userEvent.click(await screen.findByRole('button', { name: 'Use what I said' }));
    expect(await screen.findByText(/^Nothing said to remember yet\./)).toBeInTheDocument();
  });

  const LIVE_BOX: [number, number, number, number] = [0.05, 0.05, 0.9, 0.9];
  const liveDraft = {
    camera: 'cam2',
    name: 'mug',
    image: 'data:image/jpeg;base64,AAAA',
    captured_at: NOW,
    box: LIVE_BOX,
  };

  it('outlines a box clicked on Live once, with that box, and saves its mask', async () => {
    let imported: unknown;
    let enrolled: unknown;
    const calls = segments();
    server.use(
      http.post('/objects/api/import', async ({ request }) => {
        imported = await request.json();
        return HttpResponse.json({ ...snapshot, source: 'live_crop' });
      }),
      http.post('/objects/api/enroll', async ({ request }) => {
        enrolled = await request.json();
        return HttpResponse.json({ object: ssd });
      }),
    );
    renderPage(<Remember cameras={['cam2']} draft={liveDraft} onSaved={() => {}} />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeEnabled());
    expect(imported).toEqual({ camera: 'cam2', name: 'mug', image: 'data:image/jpeg;base64,AAAA', captured_at: NOW });
    expect(calls).toEqual([{ snapshot: 'snap1', bbox: LIVE_BOX, job: expect.stringMatching(/^[0-9a-f]{32}$/) }]);
    await userEvent.click(screen.getByRole('button', { name: 'Remember selected object' }));
    await waitFor(() => expect(enrolled).toMatchObject({ selection: 'selection1', bbox: [0.12, 0.15, 0.4, 0.35] }));
    expect(enrolled).not.toHaveProperty('full_crop');
    expect(calls).toHaveLength(1);
  });

  it.each(['pending', 'failed'] as const)('remembers the whole Live crop while its outline is %s', async state => {
    let finish!: () => void;
    const pending = new Promise<void>(resolve => {
      finish = resolve;
    });
    let enrolled: unknown;
    let cancelled = false;
    const saved = vi.fn();
    segments(async () => {
      if (state === 'pending') await pending;
      return HttpResponse.json({ error: 'No usable mask' }, { status: 503 });
    });
    server.use(
      http.post('/objects/api/import', () => HttpResponse.json({ ...snapshot, source: 'live_crop' })),
      http.post('/objects/api/cancel', () => {
        cancelled = true;
        return HttpResponse.json({ ok: true, cancelled: true });
      }),
      http.post('/objects/api/enroll', async ({ request }) => {
        enrolled = await request.json();
        return HttpResponse.json({ object: { ...ssd, name: 'my cup' } });
      }),
    );
    renderPage(<Remember cameras={['cam2']} draft={liveDraft} onSaved={saved} />);
    await waitFor(() => expect(screen.getByLabelText(/Object name/)).toHaveValue('mug'));
    if (state === 'pending')
      expect(await screen.findByRole('status', { name: 'Processing this view' })).toBeInTheDocument();
    else await screen.findByRole('alert');
    await userEvent.click(screen.getByRole('button', { name: 'Remember object' }));
    await waitFor(() => expect(saved).toHaveBeenCalledOnce());
    expect(enrolled).toEqual({
      name: 'mug',
      camera: 'cam2',
      snapshot: 'snap1',
      full_crop: true,
      bbox: [0, 0, 1, 1],
      aliases: '',
      note: '',
    });
    if (state === 'pending') expect(cancelled).toBe(true);
    finish();
    await waitFor(() => expect(screen.queryByLabelText('Click an object to outline it')).not.toBeInTheDocument());
  });

  describe('manual crops', () => {
    beforeEach(() => {
      class TestPointerEvent extends MouseEvent {
        pointerId: number;
        pointerType: string;
        constructor(type: string, init: PointerEventInit = {}) {
          super(type, init);
          this.pointerId = init.pointerId ?? 1;
          this.pointerType = init.pointerType ?? 'mouse';
        }
      }
      vi.stubGlobal('PointerEvent', TestPointerEvent);
    });
    afterEach(() => vi.unstubAllGlobals());

    async function drawCrop() {
      await userEvent.click(await screen.findByRole('button', { name: 'Crop mode' }));
      const canvas = frame();
      fireEvent.pointerDown(canvas, { pointerId: 1, pointerType: 'mouse', button: 0, clientX: 32, clientY: 18 });
      fireEvent.pointerMove(canvas, { pointerId: 1, pointerType: 'mouse', clientX: 160, clientY: 90 });
      fireEvent.pointerUp(canvas, { pointerId: 1, pointerType: 'mouse', clientX: 160, clientY: 90 });
      await screen.findByText('Crop: 256 × 144 pixels');
    }

    it('outlines a drawn box once, on demand, then saves its mask', async () => {
      let enrolled: unknown;
      const calls = segments();
      server.use(
        http.post('/objects/api/import', () => HttpResponse.json(snapshot)),
        http.post('/objects/api/enroll', async ({ request }) => {
          enrolled = await request.json();
          return HttpResponse.json({ object: ssd });
        }),
      );
      renderPage(<Remember cameras={['cam2']} onSaved={() => {}} />);
      await userEvent.click(screen.getByRole('button', { name: 'Freeze view' }));
      await screen.findByText(/Click the object/);
      await drawCrop();
      expect(calls).toEqual([]);
      await userEvent.click(screen.getByRole('button', { name: 'Find object in box' }));
      await waitFor(() => expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeEnabled());
      expect(calls).toEqual([{ snapshot: 'snap1', bbox: [0.1, 0.1, 0.4, 0.4], job: expect.any(String) }]);
      await userEvent.type(screen.getByLabelText(/Object name/), 'my keys');
      await userEvent.click(screen.getByRole('button', { name: 'Remember selected object' }));
      await waitFor(() => expect(enrolled).toMatchObject({ selection: 'selection1' }));
      expect(enrolled).not.toHaveProperty('direct_crop');
    });

    it('saves a direct crop, retaining it after a save error', async () => {
      const enrolled: unknown[] = [];
      let fail = true;
      const saved = vi.fn();
      const calls = segments();
      server.use(
        http.post('/objects/api/import', () => HttpResponse.json({ ...snapshot, source: 'full_frame' })),
        http.post('/objects/api/enroll', async ({ request }) => {
          enrolled.push(await request.json());
          return fail
            ? HttpResponse.json({ error: 'Please retry saving.' }, { status: 503 })
            : HttpResponse.json({ object: ssd });
        }),
      );
      renderPage(
        <Remember
          cameras={['cam2']}
          draft={{ kind: 'full_frame', camera: 'cam2', image: 'data:image/jpeg;base64,AAAA', captured_at: NOW }}
          onSaved={saved}
        />,
      );
      await screen.findByText(/Click the object/);
      await drawCrop();
      await userEvent.click(screen.getByRole('button', { name: 'Use this crop' }));
      await userEvent.type(screen.getByLabelText(/Object name/), 'my keys');
      await userEvent.click(screen.getByRole('button', { name: 'Remember crop' }));
      await screen.findByText('Please retry saving.');
      expect(screen.getByText('Crop: 256 × 144 pixels')).toBeInTheDocument();
      expect(enrolled).toEqual([
        {
          name: 'my keys',
          camera: 'cam2',
          snapshot: 'snap1',
          crop: [64, 36, 256, 144],
          direct_crop: true,
          bbox: [0.1, 0.1, 0.4, 0.4],
          aliases: '',
          note: '',
        },
      ]);
      fail = false;
      await userEvent.click(screen.getByRole('button', { name: 'Remember crop' }));
      await waitFor(() => expect(saved).toHaveBeenCalledOnce());
      expect(calls).toEqual([]);
    });

    it('Escape clears the staged crop and blocks the Live whole-image fallback until the crop is cleared', async () => {
      segments(() => HttpResponse.json({ error: 'No usable mask' }, { status: 503 }));
      server.use(http.post('/objects/api/import', () => HttpResponse.json({ ...snapshot, source: 'live_crop' })));
      renderPage(<Remember cameras={['cam2']} draft={liveDraft} onSaved={() => {}} />);
      await screen.findByRole('alert');
      expect(screen.getByRole('button', { name: 'Remember object' })).toBeEnabled();
      await drawCrop();
      expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeDisabled();
      await userEvent.click(screen.getByRole('button', { name: 'Use this crop' }));
      expect(screen.getByRole('button', { name: 'Remember crop' })).toBeEnabled();
      fireEvent.keyDown(screen.getByTestId('frame-editor'), { key: 'Escape' });
      expect(screen.queryByRole('button', { name: 'Use this crop' })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Remember crop' })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Remember object' })).toBeEnabled();
    });
  });
});

describe('served through the dashboard', () => {
  beforeEach(() => history.pushState(null, '', '/objects/'));
  afterEach(() => history.pushState(null, '', '/'));

  it('cancels the owned SAM job under /objects after navigation changes the page route', async () => {
    let finish!: () => void;
    const pending = new Promise<void>(resolve => {
      finish = resolve;
    });
    const cancellations: string[] = [];
    const cancel = ({ request }: { request: Request }) => {
      cancellations.push(new URL(request.url).pathname);
      return HttpResponse.json({ ok: true, cancelled: true });
    };
    const calls = segments(async body => {
      await pending;
      return HttpResponse.json(selection({ snapshot: String(body.snapshot) }));
    });
    server.use(
      http.get('/objects/api/status', () => HttpResponse.json(status())),
      http.get('/objects/api/objects', () => HttpResponse.json({ objects: [] })),
      http.post('/objects/api/import', () => HttpResponse.json({ ...snapshot, source: 'live_crop' })),
      http.post('/objects/api/cancel', cancel),
      http.post('/api/cancel', cancel),
    );
    const { unmount } = renderPage(
      <Remember
        cameras={['cam2']}
        draft={{
          camera: 'cam2',
          name: 'mug',
          image: 'data:image/jpeg;base64,AAAA',
          captured_at: NOW,
          box: [0, 0, 1, 1],
        }}
        onSaved={() => {}}
      />,
    );
    await waitFor(() => expect(calls).toHaveLength(1));
    history.pushState(null, '', '/');
    unmount();
    await waitFor(() => expect(cancellations).toHaveLength(1));
    finish();
    expect(cancellations).toEqual(['/objects/api/cancel']);
  });

  it('sends every request under /objects/', async () => {
    server.use(
      http.get('/objects/api/status', () => HttpResponse.json(status())),
      http.get('/objects/api/objects', () => HttpResponse.json({ objects: [ssd] })),
    );
    renderPage(<ObjectMemoryPage />);
    const item = (await screen.findByText('my SSD')).closest('div')?.parentElement as HTMLElement;
    await waitFor(() =>
      expect(within(item).getByRole('img', { name: 'Reference crop for my SSD' })).toHaveAttribute(
        'src',
        '/objects/evidence/tttt.jpg',
      ),
    );
  });
});

it('loads a Live crop once under StrictMode with an editable suggestion and no automatic save', async () => {
  serve();
  const draft = { camera: 'cam2', name: 'mug', image: 'data:image/jpeg;base64,AAAA', captured_at: NOW };
  let imports = 0;
  let enrolled = false;
  const calls = segments();
  server.use(
    http.post('/objects/api/import', async ({ request }) => {
      expect(await request.json()).toEqual(draft);
      imports++;
      return HttpResponse.json({ ...snapshot, source: 'live_crop' });
    }),
    http.post('/objects/api/enroll', () => {
      enrolled = true;
      return HttpResponse.json({ object: ssd });
    }),
  );
  renderPage(
    <StrictMode>
      <Remember cameras={['cam0', 'cam2']} draft={{ ...draft, box: [0.1, 0.1, 0.8, 0.8] }} onSaved={() => {}} />
    </StrictMode>,
  );
  await waitFor(() => expect(screen.getByLabelText(/Object name/)).toHaveValue('mug'));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeEnabled());
  expect(imports).toBe(1);
  expect(calls).toHaveLength(1);
  expect(enrolled).toBe(false);
  await userEvent.clear(screen.getByLabelText(/Object name/));
  await userEvent.type(screen.getByLabelText(/Object name/), 'my mug');
  expect(screen.getByLabelText(/Object name/)).toHaveValue('my mug');
});

function VisitedObjects() {
  const [shown, setShown] = useState(true);
  return (
    <>
      <button type="button" onClick={() => setShown(!shown)}>
        Toggle Objects
      </button>
      <Activity mode={shown ? 'visible' : 'hidden'}>
        <ObjectMemoryPage />
      </Activity>
    </>
  );
}

it('consumes a new Live crop when a previously visited Objects tab becomes visible again', async () => {
  serve();
  const calls = segments(() => HttpResponse.json({ error: 'Synthetic unavailable model' }, { status: 503 }));
  server.use(http.post('/objects/api/import', () => HttpResponse.json(snapshot)));
  renderPage(<VisitedObjects />);
  await screen.findByText(/cameras ready/);
  await userEvent.click(screen.getByRole('button', { name: 'Toggle Objects' }));
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,AAAA');
  cropLiveObject(canvas, [0.2, 0.2, 0.8, 0.8], 'cam0', 'mug');
  await userEvent.click(screen.getByRole('button', { name: 'Toggle Objects' }));
  await waitFor(() => expect(screen.getByLabelText(/Object name/)).toHaveValue('mug'));
  await waitFor(() => expect(calls).toHaveLength(1));
  expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeDisabled();
});

it('imports a whole Review frame and runs no EdgeTAM until the owner points at an object', async () => {
  serve();
  const capturedAt = NOW - 7 * 86400 + 0.125;
  const image = 'data:image/jpeg;base64,completeRecordedFrame';
  let imported: unknown;
  let enrolled = false;
  const calls = segments();
  server.use(
    http.post('/objects/api/import', async ({ request }) => {
      imported = await request.json();
      return HttpResponse.json({ ...snapshot, captured_at: capturedAt, source: 'full_frame' });
    }),
    http.post('/objects/api/enroll', () => {
      enrolled = true;
      return HttpResponse.json({ object: ssd });
    }),
  );
  handoffReviewFrame(image, 'cam2', capturedAt);
  renderPage(<ObjectMemoryPage />);
  await screen.findByText(/Click the object/);
  expect(imported).toEqual({ kind: 'full_frame', image, camera: 'cam2', captured_at: capturedAt });
  expect(calls).toEqual([]);
  expect(screen.getByLabelText(/Object name/)).toHaveValue('');
  expect(screen.getByRole('button', { name: 'Remember selected object' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Remember object' })).not.toBeInTheDocument();
  expect(enrolled).toBe(false);
});

describe('the layout when nothing is remembered', () => {
  it('waits to offer Find until something is remembered, and says how to remember one', async () => {
    serve([]);
    renderPage(<ObjectMemoryPage />);
    expect(await screen.findByText(/^Remember your keys, a bag, or a laptop once/)).toBeInTheDocument();
    expect(screen.queryByLabelText('Object question')).not.toBeInTheDocument();
  });

  it('offers Find once something is remembered', async () => {
    serve();
    renderPage(<ObjectMemoryPage />);
    expect(await screen.findByRole('button', { name: 'Last seen' })).toBeInTheDocument();
    expect(screen.getByLabelText('Object question')).toBeInTheDocument();
  });
});
