import { act, fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { hms } from '../lib/format';
import { setPref } from '../lib/prefs';
import { EngineContext } from './context';
import { ReviewEngine } from './engine';
import { nextSpeed, Player } from './Player';
import type { Span } from './types';

const T = new Date(2026, 9, 27, 10, 15).getTime() / 1000;
const MOMENTS: Span[] = [
  [T - 20, T - 15],
  [T, T + 5],
  [T + 20, T + 25],
];

function show(
  state: Partial<ReturnType<ReviewEngine['ui']['getState']>> & { moments?: Span[] | null },
  cams = ['cam0'],
) {
  const engine = new ReviewEngine({ cams, fetchNotes: async () => ({ cams: {} }) });
  const { moments, ...ui } = state;
  if (moments !== undefined) engine.present = moments;
  engine.ui.setState({ t: T, avail: [], ...ui });
  renderPage(
    <EngineContext value={engine}>
      <Player />
    </EngineContext>,
  );
  return engine;
}
const video = () => screen.getByRole('region', { name: 'Video' });

it('puts noninteractive moment key hints inside the large video', () => {
  show({ avail: ['cam0'], moments: MOMENTS });
  const large = screen.getByLabelText('Front camera video').parentElement as HTMLElement;
  const controls = within(large).getByLabelText('Moment navigation shortcuts');
  expect(within(controls).queryByRole('button')).not.toBeInTheDocument();
  expect(controls).not.toHaveAttribute('tabindex');
  expect(within(controls).getByText('[').tagName).toBe('KBD');
  expect(within(controls).getByText(']').tagName).toBe('KBD');
  expect(controls).toHaveTextContent('[ prev 2 of 3 next ]');
});

it.each([null, [], [[T, T + 5]]] as (Span[] | null)[])(
  'hides the hint without multiple known moments (%j)',
  moments => {
    show({ avail: ['cam0'], moments });
    expect(screen.queryByLabelText('Moment navigation shortcuts')).not.toBeInTheDocument();
  },
);

it('hides unavailable sides at the first and last moment and follows a refreshed list', () => {
  const engine = show({ avail: ['cam0'], moments: MOMENTS, t: T - 20 });
  const hint = screen.getByLabelText('Moment navigation shortcuts');
  expect(hint).toHaveTextContent('1 of 3 next ]');
  expect(within(hint).queryByText('[')).toBeNull();
  act(() => engine.ui.setState({ t: T + 20 }));
  expect(hint).toHaveTextContent('[ prev 3 of 3');
  expect(within(hint).queryByText(']')).toBeNull();
  act(() => {
    engine.present = [[T, T + 5]];
    engine.ui.setState(s => ({ dataVersion: s.dataVersion + 1 }));
  });
  expect(screen.queryByLabelText('Moment navigation shortcuts')).not.toBeInTheDocument();
});

it('does not turn the key hint into a clickable navigation control', () => {
  const engine = show({ avail: ['cam0'], sel: new Set(['object:cup']), moments: MOMENTS });
  const step = vi.spyOn(engine, 'stepChip');
  const canvas = screen.getByLabelText('Front camera video');
  fireEvent.click(screen.getByLabelText('Moment navigation shortcuts'));
  expect(step).not.toHaveBeenCalled();
  expect(screen.getByLabelText('Front camera video')).toBe(canvas);
});

it('keeps engine navigation and text editing unchanged with the persistent hint', async () => {
  const engine = show({ avail: ['cam0'], sel: new Set(['object:cup']), moments: MOMENTS });
  history.replaceState(null, '', '/review');
  const step = vi.spyOn(engine, 'stepChip').mockImplementation(() => {});
  try {
    act(() => {
      engine.start();
      engine.present = MOMENTS;
      engine.t = T;
      engine.ui.setState({ t: T, sel: new Set(['object:cup']) });
    });
    const overlay = screen.getByLabelText('Moment navigation shortcuts');
    await act(async () => {
      fireEvent.keyDown(document.body, { key: ']' });
      await Promise.resolve();
    });
    expect(step).toHaveBeenCalledExactlyOnceWith(1);
    expect(overlay).toBeVisible();
    const input = document.createElement('input');
    document.body.append(input);
    await act(async () => {
      fireEvent.keyDown(input, { key: '[' });
      await Promise.resolve();
    });
    input.remove();
    expect(step).toHaveBeenCalledTimes(1);
    await act(async () => {
      fireEvent.keyDown(document.body, { key: '[' });
      await Promise.resolve();
    });
    expect(step).toHaveBeenLastCalledWith(-1);
    expect(overlay).toBeVisible();
  } finally {
    engine.stop();
  }
});

beforeEach(() => {
  vi.stubGlobal('NAMES', { views: { cam0: 'front camera', cam1: 'rear camera' } });
  server.use(
    http.get('/review/boxes', () => HttpResponse.json({ boxes: [] })),
    http.get('/ambient', () => HttpResponse.json({ devices: {} })),
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
  setPref('reviewBoxes', undefined);
  setPref('reviewMarkers', undefined);
  setPref('reviewObjectsKnownOnly', undefined);
});

describe('the player with no picture', () => {
  it("says the room's sound is playing where only kept audio plays (#196)", () => {
    show({ playing: true, roomOnly: true });
    expect(screen.getByText(`No video at ${hms(T)}. Playing the room's sound.`)).toBeInTheDocument();
  });

  it('just says there is no video otherwise', () => {
    show({ playing: false, roomOnly: false });
    expect(screen.getByText(`No video at ${hms(T)}.`)).toBeInTheDocument();
  });
});

describe('expired video (#1748)', () => {
  const seg = (cam: string, i: number, broken?: 'expired') => ({
    cam,
    start: T - 10,
    end: T + 10,
    bytes: 0,
    url: `gateway:${i}:${(T - 10) * 1e6}:${(T + 10) * 1e6}`,
    ...(broken ? { broken } : {}),
  });

  it("says the shown camera's video expired, even while another camera still plays", async () => {
    const engine = show({ avail: ['cam1'] }, ['cam0', 'cam1']);
    engine.segs = { cam0: [seg('cam0', 1, 'expired')], cam1: [seg('cam1', 2)] };
    await userEvent.click(screen.getByRole('button', { name: 'Show the front camera large' }));
    expect(
      screen.getByText(
        `The video around ${hms(T)} has expired and was deleted. What was seen and heard in it is still kept.`,
      ),
    ).toBeInTheDocument();
  });

  it("doesn't blame the shown camera for another camera's expired video", () => {
    const engine = show({ avail: [], gap: 'expired' }, ['cam0', 'cam1']);
    engine.segs = { cam1: [seg('cam1', 2, 'expired')] };
    expect(screen.getByText(`No video at ${hms(T)}.`)).toBeInTheDocument();
  });
});

describe("the journal's kept evidence in place of expired video (#2838)", () => {
  const DAY = '2026-10-27';
  const expiredSeg = {
    cam: 'cam0',
    start: T - 10,
    end: T + 10,
    bytes: 0,
    url: 'gateway:1:0:0',
    broken: 'expired' as const,
  };
  const cited = (over: Record<string, unknown>) => ({
    label: 'h1',
    stream: 'camera.caption',
    ts_start: T - 1,
    ts_end: T + 1,
    caption: 'Someone at the kitchen counter',
    camera: 'cam0',
    t: T,
    ...over,
  });

  function showExpired(sources: unknown[], cams = ['cam0']) {
    server.use(
      http.get('/journal/evidence', ({ request }) =>
        new URL(request.url).searchParams.get('day') === DAY
          ? HttpResponse.json({ day: DAY, sources })
          : HttpResponse.json({ error: 'wrong day' }, { status: 400 }),
      ),
    );
    const engine = new ReviewEngine({ cams, fetchNotes: async () => ({ cams: {} }) });
    engine.segs = Object.fromEntries(cams.map(cam => [cam, [{ ...expiredSeg, cam }]]));
    engine.ui.setState({ t: T, avail: [], day: DAY });
    renderPage(
      <EngineContext value={engine}>
        <Player />
      </EngineContext>,
    );
  }

  it('plays the promoted clip, its still as the poster, with what was seen', async () => {
    const url = `/journal/evidence/${DAY}/cam0-1.mp4`;
    showExpired([cited({ stage: 'clip', url, still: `/journal/evidence/${DAY}/cam0-1.jpg` })]);
    expect(await screen.findByText('Kept clip: Someone at the kitchen counter')).toBeInTheDocument();
    const clip = document.querySelector('video');
    expect(clip).toHaveAttribute('src', url);
    expect(clip).toHaveAttribute('poster', `/journal/evidence/${DAY}/cam0-1.jpg`);
    expect(screen.queryByText(/has expired and was deleted/)).not.toBeInTheDocument();
  });

  it('shows the promoted still when no clip could be made', async () => {
    showExpired([cited({ stage: 'still', url: `/journal/evidence/${DAY}/cam0-1.jpg` })]);
    const still = await screen.findByRole('img', { name: 'Someone at the kitchen counter' });
    expect(still).toHaveAttribute('src', `/journal/evidence/${DAY}/cam0-1.jpg`);
  });

  it('says the footage expired and gives what the journal cites, rather than a dead picture', async () => {
    showExpired([cited({ stage: 'expired', note: 'footage expired' })]);
    expect(
      await screen.findByText(/has expired and was deleted\..* The journal cites it: Someone at the kitchen counter/),
    ).toBeInTheDocument();
  });

  it('matches a source to the large camera by its camera id, the native name both sides use', async () => {
    showExpired([cited({ stage: 'still', url: '/kept/rear.jpg', camera: 'cam1' })], ['cam0', 'cam1']);
    expect(await screen.findByText(/has expired and was deleted/)).toBeInTheDocument(); // cam0 is large: not its
    await userEvent.click(screen.getByRole('button', { name: 'Show the rear camera large' }));
    expect(await screen.findByRole('img', { name: 'Someone at the kitchen counter' })).toHaveAttribute(
      'src',
      '/kept/rear.jpg',
    );
  });

  it('keeps the plain message when the journal cites nothing on this camera then', async () => {
    showExpired([cited({ stage: 'clip', url: '/x.mp4', camera: 'cam1' })]);
    expect(
      await screen.findByText(
        `The video around ${hms(T)} has expired and was deleted. What was seen and heard in it is still kept.`,
      ),
    ).toBeInTheDocument();
    expect(document.querySelector('video')).toBeNull();
  });
});

describe('one camera large, the others as tiles (#327)', () => {
  it('shows the first camera with video large, names each camera, and says which tile has no video now', () => {
    show({ avail: ['cam1'] }, ['cam0', 'cam1']);
    const big = screen.getByLabelText('Rear camera video').parentElement as HTMLElement;
    expect(within(big).getByText('Rear camera')).toBeInTheDocument();
    expect(within(big).queryByRole('button', { name: /Show the .* large/ })).not.toBeInTheDocument();
    const tile = screen.getByRole('button', { name: 'Show the front camera large' }).parentElement as HTMLElement;
    expect(within(tile).getByText('Front camera')).toBeInTheDocument();
    expect(within(tile).getByText('No video now')).toBeInTheDocument();
  });

  it('swaps a tile into the large view, without remounting any video', async () => {
    show({ avail: ['cam0', 'cam1'] }, ['cam0', 'cam1']);
    const front = screen.getByLabelText('Front camera video');
    const rear = screen.getByLabelText('Rear camera video');
    await userEvent.click(screen.getByRole('button', { name: 'Show the rear camera large' }));
    expect(screen.getByRole('button', { name: 'Show the front camera large' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Show the rear camera large' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Front camera video')).toBe(front); // the same elements, so the engine keeps them
    expect(screen.getByLabelText('Rear camera video')).toBe(rear);
  });

  it('a new clip link supersedes an old manual camera choice, even for the same linked camera', async () => {
    const engine = show({ avail: ['cam0', 'cam1'] }, ['cam0', 'cam1']);
    act(() => {
      engine.start();
    });
    const followClip = () => {
      history.replaceState(null, '', `/review#cam=cam0&play=1&t=${T}`);
      window.dispatchEvent(new HashChangeEvent('hashchange'));
    };
    act(followClip);
    await userEvent.click(screen.getByRole('button', { name: 'Show the rear camera large' }));
    expect(screen.getByRole('button', { name: 'Show the front camera large' })).toBeInTheDocument();
    act(followClip);
    expect(screen.getByRole('button', { name: 'Show the rear camera large' })).toBeInTheDocument();
    // Manual selection still works after the link was followed.
    await userEvent.click(screen.getByRole('button', { name: 'Show the rear camera large' }));
    expect(screen.getByRole('button', { name: 'Show the front camera large' })).toBeInTheDocument();
    act(() => {
      engine.stop();
    });
    history.replaceState(null, '', '/');
  });

  it('captions the large camera with what it noticed, and who it saw', () => {
    show(
      {
        avail: ['cam0'],
        notes: { cams: { cam0: { ts: T, dt: 0, caption: 'Two people at the counter', faces: [{ name: 'Dana' }] } } },
      },
      ['cam0', 'cam1'],
    );
    const big = screen.getByLabelText('Front camera video').parentElement as HTMLElement;
    expect(within(big).getByText('Two people at the counter')).toBeInTheDocument();
    expect(within(big).getByText('Seen: Dana')).toBeInTheDocument();
  });
});

describe('event marks use the displayed camera', () => {
  const engines: ReviewEngine[] = [];
  const opened = (state: Partial<ReturnType<ReviewEngine['ui']['getState']>>) => {
    history.replaceState(null, '', '/review');
    const engine = show(state, ['cam0', 'cam1']);
    act(() => {
      engine.start();
      engine.t = T;
      // Model the loaded day after start clears the previous day's availability.
      engine.ui.setState(state);
    });
    engines.push(engine);
    return engine;
  };
  afterEach(() => {
    for (const engine of engines.splice(0)) engine.stop();
  });

  it.each([{ avail: ['cam0', 'cam1'] }, { avail: ['cam0', 'cam1'], camera: 'cam0', cameraIntent: 1 }])(
    'marks the camera swapped large, retaining it if the tile changes before mark out: %j',
    async state => {
      const engine = opened(state);
      await userEvent.click(screen.getByRole('button', { name: 'Show the rear camera large' }));
      fireEvent.keyDown(document.body, { key: 'i' });
      expect(engine.ui.getState().eventMarks).toEqual({ cam: 'cam1', in: T, out: null });
      await userEvent.click(screen.getByRole('button', { name: 'Show the front camera large' }));
      act(() => {
        engine.t = T + 10;
      });
      fireEvent.keyDown(document.body, { key: 'o' });
      expect(engine.ui.getState().eventSpan).toEqual({ cam: 'cam1', start: T, end: T + 10 });
    },
  );

  it('marks the available camera shown large before any manual selection', () => {
    const engine = opened({ avail: ['cam1'] });
    expect(screen.getByRole('button', { name: 'Show the front camera large' })).toBeInTheDocument();
    fireEvent.keyDown(document.body, { key: 'i' });
    act(() => {
      engine.t = T + 10;
    });
    fireEvent.keyDown(document.body, { key: 'o' });
    expect(engine.ui.getState().eventSpan).toEqual({ cam: 'cam1', start: T, end: T + 10 });
  });
});

describe('the controls', () => {
  it('play and pause through the engine', async () => {
    const engine = show({ playing: false });
    const toggle = vi.spyOn(engine, 'togglePlay').mockImplementation(() => {});
    await userEvent.click(within(video()).getByRole('button', { name: 'Play' }));
    expect(toggle).toHaveBeenCalled();
  });

  it('step the speed 1×, 1.5×, 2×, and round again', async () => {
    expect([1, 1.5, 2].map(nextSpeed)).toEqual([1.5, 2, 1]);
    const engine = show({});
    await userEvent.click(within(video()).getByRole('button', { name: 'Speed 1×. Change to 1.5×' }));
    expect(engine.ui.getState().speed).toBe(1.5);
    expect(within(video()).getByRole('button', { name: 'Speed 1.5×. Change to 2×' })).toHaveTextContent('1.5×');
  });

  it('skip quiet time with the checkbox', async () => {
    const engine = show({ skip: false });
    await userEvent.click(within(video()).getByRole('checkbox', { name: 'Skip quiet time' }));
    expect(engine.ui.getState().skip).toBe(true);
  });
});

describe('the caption (#379)', () => {
  const note = (over: { caption: string; unchanged?: boolean; since?: number }) => ({
    notes: { cams: { cam0: { ts: T, dt: 0, faces: null, ...over } } },
    avail: ['cam0'],
  });

  it('shows who was seen when there is no caption (the vision model was warming up)', () => {
    show({
      avail: ['cam0'],
      notes: {
        cams: { cam0: { ts: T, dt: 0, faces: [{ name: 'Dana' }], caption: null as unknown as string } },
      },
    });
    expect(screen.getByText('Seen: Dana')).toBeInTheDocument();
  });

  it('marks a caption carried forward with when it was new', () => {
    const since = new Date(2026, 9, 27, 18, 40).getTime() / 1000;
    show(note({ caption: 'Two people at the counter', unchanged: true, since }));
    expect(screen.getByText('Two people at the counter')).toBeInTheDocument();
    expect(screen.getByTitle('Unchanged since 18:40')).toBeInTheDocument();
  });

  it('keeps focus on the caption as it changes, and starts each new caption at two lines (#2537)', async () => {
    const engine = show(note({ caption: 'Two people at the counter' }));
    const button = screen.getByRole('button', { name: 'Two people at the counter' });
    await userEvent.click(button);
    expect(button).toHaveAttribute('aria-expanded', 'true');
    expect(button).toHaveFocus();
    act(() => engine.ui.setState(note({ caption: 'One person by the door' })));
    const next = screen.getByRole('button', { name: 'One person by the door' });
    expect(next).toBe(button); // the same element, so focus stays put during playback
    expect(next).toHaveFocus();
    expect(next).toHaveAttribute('aria-expanded', 'false');
  });

  it("puts the caption's notes on its own line, not below it", () => {
    show({
      ...note({ caption: 'Two people at the counter' }),
      notes: { cams: { cam0: { ts: T, dt: -12, faces: null, caption: 'Two people at the counter' } } },
    });
    const words = screen.getByRole('button', { name: 'Two people at the counter' });
    expect(words.parentElement).toContainElement(screen.getByText('(12 s before)'));
  });
});

describe('the boxes (#374)', () => {
  it('defaults to people and remembers independent pet and object marker choices', async () => {
    show({ avail: ['cam0'] });
    const group = screen.getByRole('group', { name: 'Review markers' });
    expect(within(group).getByRole('checkbox', { name: 'People' })).toBeChecked();
    expect(within(group).getByRole('checkbox', { name: 'Pets' })).not.toBeChecked();
    expect(within(group).getByRole('checkbox', { name: 'Objects' })).not.toBeChecked();
    await userEvent.click(within(group).getByRole('checkbox', { name: 'Pets' }));
    await userEvent.click(within(group).getByRole('checkbox', { name: 'Objects' }));
    expect(JSON.parse(localStorage.getItem('liberational.prefs') ?? '{}').reviewMarkers).toEqual([
      'people',
      'pets',
      'objects',
    ]);
    await userEvent.click(within(group).getByRole('checkbox', { name: 'People' }));
    expect(within(group).getByRole('checkbox', { name: 'Pets' })).toBeChecked();
    expect(document.querySelector('.bx-layer')).not.toBeNull();
  });
  it('offers Known only under Objects, on by default, and remembers turning it off', async () => {
    show({ avail: ['cam0'] });
    expect(screen.queryByRole('checkbox', { name: 'Known only' })).toBeNull();
    const group = screen.getByRole('group', { name: 'Review markers' });
    await userEvent.click(within(group).getByRole('checkbox', { name: 'Objects' }));
    const knownOnly = screen.getByRole('checkbox', { name: 'Known only' });
    expect(knownOnly).toBeChecked();
    await userEvent.click(knownOnly);
    expect(knownOnly).not.toBeChecked();
    expect(JSON.parse(localStorage.getItem('liberational.prefs') ?? '{}').reviewObjectsKnownOnly).toBe(false);
  });
  it('asks for the large camera’s boxes around the playhead, and none once hidden', async () => {
    const asked: string[] = [];
    server.use(
      http.get('/review/boxes', ({ request }) => {
        asked.push(new URL(request.url).search);
        return HttpResponse.json({ boxes: [] });
      }),
    );
    show({ avail: ['cam0'] });
    const start = Math.floor(T / 60) * 60;
    await vi.waitFor(() => expect(asked).toContain(`?cam=cam0&start=${start}&end=${start + 60}`));
    expect(document.querySelector('.bx-layer')).not.toBeNull();
    await userEvent.click(within(video()).getByRole('checkbox', { name: 'People' }));
    expect(document.querySelector('.bx-layer')).toBeNull();
  });

  it('draws no boxes over a camera with no video now', () => {
    show({ avail: [] });
    expect(document.querySelector('.bx-layer')).toBeNull();
  });
});

it('hides excluded camera tiles, chooses an included large view, and restores the same canvases', () => {
  const engine = show({ avail: ['cam0', 'cam1'] }, ['cam0', 'cam1']);
  const front = screen.getByRole('img', { name: 'Front camera video' });
  const rear = screen.getByRole('img', { name: 'Rear camera video' });
  act(() => engine.pickCameras(['cam1']));
  expect(screen.queryByRole('img', { name: 'Front camera video' })).not.toBeInTheDocument();
  expect(screen.getByRole('img', { name: 'Rear camera video' })).toBe(rear);
  expect(screen.queryByRole('button', { name: 'Show the rear camera large' })).not.toBeInTheDocument();
  act(() => engine.pickCameras([]));
  expect(screen.getByRole('img', { name: 'Front camera video' })).toBe(front);
});

it('mutes the room microphone without pausing playback', async () => {
  const engine = show({ playing: true });
  const audio = document.createElement('audio');
  engine.setAudio(audio);
  await userEvent.click(screen.getByRole('button', { name: 'Mute the room microphone' }));
  expect(audio.muted).toBe(true);
  expect(engine.ui.getState().playing).toBe(true);
  expect(screen.getByRole('button', { name: 'Unmute the room microphone' })).toHaveAttribute('aria-pressed', 'true');
  await userEvent.click(screen.getByRole('button', { name: 'Unmute the room microphone' }));
  expect(audio.muted).toBe(false);
  expect(engine.ui.getState().playing).toBe(true);
});
