import { act, fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import type { MakeVideo } from '../../review/Player';
import type { RawSegment } from '../../review/types';
import type { CardVideo } from '../api';
import { VideoFloat } from './VideoFloat';

const START = new Date(2026, 9, 27, 10, 0).getTime() / 1000;
const CLIP: CardVideo = { cam: 'cam0', start: START, end: START + 10, duration_s: 10, thumb: '/thumb.jpg', what: null };
const SEGMENTS: RawSegment[] = [{ cam: 'cam0', start: START, end: START + 10, url: 'gateway:1:0:10000000' }];

/** Actual DOM media events, with only the browser decoder/playback boundary replaced. */
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
  const destroyed = vi.fn(() => {
    element.pause();
    element.removeAttribute('src');
  });
  return Object.assign(element, { close: destroyed, destroyed });
}

function show(load = vi.fn(async (_start: number, _end: number, _signal: AbortSignal) => SEGMENTS)) {
  const origin = document.createElement('button');
  origin.textContent = 'Original clip';
  document.body.append(origin);
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
  const closed = vi.fn();
  const rendered = renderPage(
    <VideoFloat video={CLIP} name="Dana" origin={origin} onClose={closed} makeVideo={make} loadSegments={load} />,
  );
  return { ...rendered, origin, videos, closed, load };
}

afterEach(() => {
  vi.useRealTimers();
  history.replaceState(null, '', '/');
  document.querySelectorAll('body > button').forEach(element => {
    element.remove();
  });
});

describe('floating People video', () => {
  it('opens the clip through Review playback and keeps People in the address bar', async () => {
    history.replaceState(null, '', '/people#Dana');
    const { videos, load } = show();
    await screen.findByRole('dialog', { name: 'Video of Dana' });
    await waitFor(() => expect(videos[0]?.paused).toBe(false));
    expect(load).toHaveBeenCalledWith(START, START + 10, expect.any(AbortSignal));
    expect(videos[0]?.src).toContain('gateway:1:0:10000000');
    expect(location.pathname + location.hash).toBe('/people#Dana');
    expect(screen.queryByText('Show boxes')).not.toBeInTheDocument();
    expect(screen.queryByText('Skip quiet time')).not.toBeInTheDocument();
  });

  it('selects the requested camera and its earliest retained segment when listing order differs', async () => {
    const load = vi.fn(async () => [
      { cam: 'cam1', start: START, end: START + 10, url: 'gateway:9:0:10000000' },
      { cam: 'cam0', start: START + 5, end: START + 10, url: 'gateway:2:5000000:10000000' },
      ...SEGMENTS,
    ]);
    const { videos } = show(load);
    await waitFor(() => expect(videos[0]?.paused).toBe(false));
    expect(videos[0]?.src).toContain('gateway:1:0:10000000');
  });

  it('keeps the poster through seek completion until the active canvas has a painted frame', async () => {
    const { videos } = show();
    await waitFor(() => expect(videos[0]?.paused).toBe(false));
    const dialog = screen.getByRole('dialog');
    const poster = dialog.querySelector('img');
    expect(poster).not.toHaveAttribute('data-ready');
    const standby = dialog.querySelector('canvas[data-standby]') as HTMLCanvasElement;
    await act(async () => {
      standby.width = 640;
      standby.height = 480;
    });
    expect(poster).not.toHaveAttribute('data-ready');
    const active = dialog.querySelector('canvas:not([data-standby])') as HTMLCanvasElement;
    await act(async () => {
      active.width = 640;
      active.height = 480;
    });
    expect(poster).toHaveAttribute('data-ready');
  });

  it('fades a failed poster away when the active canvas paints', async () => {
    const { videos } = show();
    await waitFor(() => expect(videos[0]?.paused).toBe(false));
    const dialog = screen.getByRole('dialog');
    fireEvent.error(dialog.querySelector('img') as HTMLImageElement);
    const poster = dialog.querySelector('[data-safe-image]') as HTMLElement;
    expect(poster).toBeInTheDocument();
    expect(poster).not.toHaveStyle({ opacity: '0' });
    const active = dialog.querySelector('canvas:not([data-standby])') as HTMLCanvasElement;
    await act(async () => {
      active.width = 640;
      active.height = 480;
    });
    expect(poster).toHaveStyle({ opacity: '0' });
    await userEvent.click(screen.getByRole('button', { name: 'Close video' }));
    expect(poster).toHaveStyle({ opacity: '1' });
  });

  it('keeps a delayed first frame visible as loading, then offers a retry that replaces stalled media', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { videos, load } = show();
    await act(async () => {});
    expect(screen.getByRole('status')).toHaveTextContent('Loading video');
    expect(screen.queryByRole('button', { name: 'Retry video' })).not.toBeInTheDocument();
    const stale = screen.getByRole('dialog').querySelector('canvas:not([data-standby])') as HTMLCanvasElement;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(screen.getByRole('status')).toHaveTextContent('Still loading video');
    fireEvent.click(screen.getByRole('button', { name: 'Retry video' }));
    await act(async () => {});
    expect(load).toHaveBeenCalledTimes(2);
    expect(load.mock.calls[0]?.[2].aborted).toBe(true);
    expect(videos[0]?.destroyed).toHaveBeenCalledOnce();
    await act(async () => {
      stale.width = 640;
      stale.height = 480;
    });
    expect(screen.getByRole('status')).toHaveTextContent('Loading video');
    const active = screen.getByRole('dialog').querySelector('canvas:not([data-standby])') as HTMLCanvasElement;
    await act(async () => {
      active.width = 640;
      active.height = 480;
    });
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(screen.queryByRole('button', { name: 'Retry video' })).not.toBeInTheDocument();
  });

  it('accepts a late first frame without forcing a retry', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { load } = show();
    await act(async () => {});
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(screen.getByRole('status')).toHaveTextContent('Still loading video');
    const active = screen.getByRole('dialog').querySelector('canvas:not([data-standby])') as HTMLCanvasElement;
    await act(async () => {
      active.width = 640;
      active.height = 480;
    });
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(load).toHaveBeenCalledOnce();
  });

  it('retries a slow listing and ignores the abandoned response', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    let finish: (segments: RawSegment[]) => void = () => {};
    const load = vi.fn((_start: number, _end: number, _signal: AbortSignal) => Promise.resolve(SEGMENTS));
    load.mockImplementationOnce(
      () =>
        new Promise<RawSegment[]>(resolve => {
          finish = resolve;
        }),
    );
    const { videos } = show(load);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    fireEvent.click(screen.getByRole('button', { name: 'Retry video' }));
    await act(async () => {});
    expect(load.mock.calls[0]?.[2].aborted).toBe(true);
    expect(videos[2]?.paused).toBe(false);
    await act(async () => {
      finish([]);
    });
    expect(screen.queryByText('This recording is no longer kept.')).not.toBeInTheDocument();
    expect(videos[2]?.paused).toBe(false);
  });

  it('opens Review at the current playback point with fresh play intent', async () => {
    const { videos, closed } = show();
    await waitFor(() => expect(videos[0]?.paused).toBe(false));
    act(() => {
      if (videos[0]) videos[0].currentTime = 5;
    });
    await waitFor(() =>
      expect(new URL((screen.getByRole('link', { name: 'Open in Review' }) as HTMLAnchorElement).href).hash).toContain(
        `t=${START + 5}`, // a whole second, written without decimals (#2391)
      ),
    );
    await userEvent.click(screen.getByRole('link', { name: 'Open in Review' }));
    const hash = new URLSearchParams(location.hash.slice(1));
    expect(hash.get('t')).toBe(String(START + 5));
    expect(hash.get('cam')).toBe('cam0');
    expect(hash.get('play')).toBe('1');
    expect(closed).toHaveBeenCalledOnce();
    expect(videos[0]?.paused).toBe(true);
  });

  it('aborts a pending listing on close and never applies its late result', async () => {
    let finish: (segments: RawSegment[]) => void = () => {};
    const load = vi.fn(
      (_start: number, _end: number, _signal: AbortSignal) =>
        new Promise<RawSegment[]>(resolve => {
          finish = resolve;
        }),
    );
    const { videos, closed, unmount } = show(load);
    await userEvent.click(screen.getByRole('button', { name: 'Close video' }));
    await waitFor(() => expect(closed).toHaveBeenCalledOnce());
    expect(load.mock.calls[0]?.[2].aborted).toBe(true);
    await act(async () => {
      finish(SEGMENTS);
    });
    for (const video of videos) expect(video.play).not.toHaveBeenCalled();
    unmount();
    for (const video of videos) expect(video.destroyed).toHaveBeenCalledOnce();
  });

  it('animates the actual portal dialog once, then returns toward its tile on close', async () => {
    const original = Element.prototype.animate;
    const targets: Element[] = [];
    const frames: Keyframe[][] = [];
    const cancelled: number[] = [];
    const animate = vi.fn(function (this: Element, keys: Keyframe[]) {
      targets.push(this);
      frames.push(keys);
      const index = targets.length;
      return {
        finished: Promise.resolve(),
        cancel: () => {
          cancelled.push(index);
        },
      } as unknown as Animation;
    });
    Object.defineProperty(Element.prototype, 'animate', { configurable: true, value: animate });
    try {
      const { closed } = show();
      const dialog = await screen.findByRole('dialog', { name: 'Video of Dana' });
      await waitFor(() => expect(animate).toHaveBeenCalledTimes(2));
      expect(targets).toContain(dialog);
      act(() => {
        window.dispatchEvent(new Event('resize'));
      });
      expect(animate).toHaveBeenCalledTimes(2);
      await userEvent.click(screen.getByRole('button', { name: 'Close video' }));
      await waitFor(() => expect(closed).toHaveBeenCalledOnce());
      expect(animate).toHaveBeenCalledTimes(4);
      expect(cancelled).toEqual([1, 2]); // opening effects yield to the matching final layout; closing effects hold
      for (const keyframe of frames.flat()) {
        expect(Object.keys(keyframe).every(key => ['transform', 'clipPath', 'opacity'].includes(key))).toBe(true);
      }
    } finally {
      if (original) Object.defineProperty(Element.prototype, 'animate', { configurable: true, value: original });
      else Reflect.deleteProperty(Element.prototype, 'animate');
    }
  });

  it('shows an explicit unavailable state when the recording is gone', async () => {
    show(vi.fn(async () => []));
    expect(await screen.findByText('This recording is no longer kept.')).toBeInTheDocument();
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });
});

beforeEach(() => server.use(http.get('/ambient', () => HttpResponse.json({ devices: {} }))));
