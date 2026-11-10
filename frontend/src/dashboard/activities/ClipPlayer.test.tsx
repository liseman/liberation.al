import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { MakeVideo } from '../review/Player';
import type { RawSegment } from '../review/types';
import { ClipPlayer } from './ClipPlayer';
import { thumbWait } from './clipThumb';

const START = 1_793_000_000;
const asked: URL[] = [];
let answer: number;

beforeEach(() => {
  asked.length = 0;
  answer = 200;
  server.use(
    http.get('/people/video-thumb.jpg', ({ request }) => {
      asked.push(new URL(request.url));
      return answer === 200
        ? new HttpResponse(new Uint8Array([0xff, 0xd8, 0xff]), { headers: { 'Content-Type': 'image/jpeg' } })
        : new HttpResponse(null, { status: answer });
    }),
  );
});

const play = () => screen.getByRole('button', { name: /^Play squats at/ });
const still = () => play().querySelector('img');
const show = (loadSegments = vi.fn(async () => [])) =>
  renderPage(<ClipPlayer cam="living-room" start={START} end={START + 6} label="squats" loadSegments={loadSegments} />);

it("shows the clip's start frame on the poster before it's played", async () => {
  const loadSegments = vi.fn(async () => []);
  show(loadSegments);
  await waitFor(() => expect(still()).not.toBeNull());
  expect(asked[0]?.searchParams.get('cam')).toBe('living-room');
  expect(asked[0]?.searchParams.get('start')).toBe(`${START}.000`);
  expect(asked[0]?.searchParams.get('end')).toBe(`${START + 6}.000`);
  expect(play()).not.toHaveAttribute('aria-busy');
  expect(loadSegments).not.toHaveBeenCalled();
});

it('waits on a still that is still being cut, then shows it', async () => {
  answer = 202;
  show();
  await waitFor(() => expect(play()).toHaveAttribute('aria-busy', 'true'));
  expect(still()).toBeNull();
  answer = 200;
  await waitFor(() => expect(still()).not.toBeNull(), { timeout: 4000 });
  expect(play()).not.toHaveAttribute('aria-busy');
});

it('keeps the plain poster when no frame is kept', async () => {
  answer = 404;
  show();
  await waitFor(() => expect(asked).toHaveLength(1));
  await waitFor(() => expect(play()).not.toHaveAttribute('aria-busy'));
  expect(still()).toBeNull();
});

it('loads the video only when played', async () => {
  server.use(http.get('/ambient', () => HttpResponse.json({ devices: {} }))); // the player's sound meter
  const loadSegments = vi.fn(async () => []);
  show(loadSegments);
  await userEvent.click(play());
  expect(await screen.findByText('This recording is no longer kept.')).toBeInTheDocument();
  expect(loadSegments).toHaveBeenCalledTimes(1);
});

it('asks again after 2 s, then backs off to at most 30 s', () => {
  expect([1, 2, 3].map(thumbWait)).toEqual([2000, 3000, 4500]);
  expect(thumbWait(40)).toBe(30_000);
});

it("shapes the poster to the camera's picture, from its still", async () => {
  show();
  await waitFor(() => expect(still()).not.toBeNull());
  const img = still() as HTMLImageElement;
  Object.defineProperties(img, { naturalWidth: { value: 101 }, naturalHeight: { value: 180 } });
  fireEvent.load(img);
  expect(play().style.getPropertyValue('--clip-ratio')).toBe(String(101 / 180));
});

/** Real DOM media events, with only the decoder replaced (as VideoFloat's tests do). */
function media() {
  const element = document.createElement('video');
  let paused = true;
  let time = 0;
  Object.defineProperties(element, {
    readyState: { get: () => 4 },
    duration: { get: () => 10 },
    paused: { get: () => paused },
    currentTime: {
      get: () => time,
      set: (value: number) => {
        time = value;
        queueMicrotask(() => element.dispatchEvent(new Event('seeked')));
      },
    },
  });
  element.play = vi.fn(async () => {
    paused = false;
    element.dispatchEvent(new Event('playing'));
  });
  element.pause = vi.fn(() => {
    paused = true;
    element.dispatchEvent(new Event('pause'));
  });
  element.load = vi.fn();
  return Object.assign(element, { close: vi.fn() });
}

it('opens the playing clip bigger in a popup, and hands its time back to the card on close', async () => {
  server.use(
    http.get('/ambient', () => HttpResponse.json({ devices: {} })),
    http.get('/review/boxes', () => HttpResponse.json({ boxes: [] })), // the popup's player shows who's where
  );
  const segments: RawSegment[] = [
    { cam: 'living-room', start: START - 30, end: START + 30, url: 'gateway:1:0:60000000' },
  ];
  const videos: ReturnType<typeof media>[] = [];
  const make: MakeVideo = canvas => {
    const video = media();
    const toggle = video.toggleAttribute.bind(video);
    video.toggleAttribute = (name, force) => {
      canvas.toggleAttribute(name, force);
      return toggle(name, force);
    };
    videos.push(video);
    return video;
  };
  renderPage(
    <ClipPlayer
      cam="living-room"
      start={START}
      end={START + 6}
      label="squats"
      makeVideo={make}
      loadSegments={async () => segments}
    />,
  );
  await userEvent.click(play());
  const expand = await screen.findByRole('button', { name: 'Open squats bigger' });
  const card = videos.slice();
  await waitFor(() => expect(card.some(v => !v.paused)).toBe(true));
  await userEvent.click(expand);
  const popup = await screen.findByRole('dialog', { name: 'squats' });
  expect(card.every(v => v.paused)).toBe(true);
  expect(within(popup).getByRole('img', { name: /video$/ })).toBeInTheDocument();
  const big = videos.filter(v => !card.includes(v));
  await waitFor(() => expect(big.some(v => !v.paused)).toBe(true));
  // A click on the popup's own video leaves it playing where it is (#2302), not reopened at the paused card's time.
  await userEvent.click(within(popup).getByRole('img', { name: /video$/ }));
  await new Promise(resolve => setTimeout(resolve, 50));
  expect(big.some(v => !v.paused)).toBe(true);
  await userEvent.keyboard('{Escape}');
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  await waitFor(() => expect(card.some(v => !v.paused)).toBe(true));
});
