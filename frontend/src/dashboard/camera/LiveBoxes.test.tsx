import { act, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Box } from './boxes';
import { type FakeBox, fakeFrames, fakeGateway } from './testFakes';

/** A box's translate as numbers: jsdom writes the transform back in its own format ("translate(40px, 20px)"), so the
 *  tests read the position, not the string. */
function translated(el: HTMLElement | null | undefined): [number, number] | null {
  const m = el?.style.transform.match(/translate\(\s*(-?[\d.]+)px\s*,\s*(-?[\d.]+)px\s*\)/);
  return m ? [Number(m[1]), Number(m[2])] : null;
}

// boxes.ts and gateway.ts keep their sessions at module level, so each test loads fresh copies.
let LiveBoxes: typeof import('./LiveBoxes').LiveBoxes;
let BoxLayer: typeof import('./BoxLayer').BoxLayer;
let frames: ReturnType<typeof fakeFrames>;
let net: ReturnType<typeof fakeGateway>;
let now = 0;
const step = () => {
  now += 100;
  frames.step(now);
};
const flush = () => act(() => vi.advanceTimersByTimeAsync(0)); // a gateway connection opens on a resolved promise

beforeEach(async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
  frames = fakeFrames();
  now = performance.now();
  vi.resetModules();
  const gw = await import('./gateway');
  net = fakeGateway();
  gw.setGatewayForTests(net.connect);
  ({ LiveBoxes } = await import('./LiveBoxes'));
  ({ BoxLayer } = await import('./BoxLayer'));
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

/** A camera tile: the live canvas, and the boxes layer beside it. */
const Tile = ({ cam, mirror, skeletons }: { cam: string; mirror?: boolean; skeletons?: boolean }) => (
  <div data-testid={cam}>
    <canvas className="live-canvas" />
    <LiveBoxes cam={cam} mirror={mirror} skeletons={skeletons} />
  </div>
);
/** A person whose left shoulder and elbow are seen (one bone, two joints) and nothing else. */
const person = () => ({
  box: [0.2, 0.1, 0.6, 0.9, 0.9] as [number, number, number, number, number],
  joints: Array.from({ length: 26 }, (_, i) => (i === 5 ? [0.4, 0.3, 0.9] : i === 7 ? [0.35, 0.45, 0.8] : [-1, -1, 0])),
});
const face = (name: string): FakeBox => ({ xyxy: [0.1, 0.1, 0.2, 0.2], label: name, kind: 'face', id: '1' });
const us = () => Date.now() * 1000;
const labelsIn = (root: HTMLElement) => [...root.querySelectorAll('.bx .bx-new')].map(n => n.textContent);

describe('LiveBoxes', () => {
  it("opens a session per camera, each layer drawing only its own camera's boxes", async () => {
    const { getByTestId } = render(
      <>
        <Tile cam="cam0" />
        <Tile cam="cam1" />
      </>,
    );
    await flush();
    expect(net.conns).toHaveLength(2);
    const [c0, c1] = net.conns;
    c0?.hello();
    c0?.streams({ 'cam0.video': 1 });
    c1?.hello();
    c1?.streams({ 'cam1.video': 2 });
    expect(c0?.last).toEqual({ type: 'live', stream: 1, overlays: ['faces', 'objects', 'poses'] });
    c0?.overlay(us(), [face('Alice')]);
    step();
    expect(labelsIn(getByTestId('cam0'))).toEqual(['Alice']);
    expect(labelsIn(getByTestId('cam1'))).toEqual([]);
  });

  it("closes a camera's session only when its last layer goes, and opens a new one for the next", async () => {
    const { rerender, unmount } = render(
      <>
        <Tile cam="cam0" />
        <Tile cam="cam0" mirror />
      </>,
    );
    await flush();
    const [first] = net.conns;
    rerender(<Tile cam="cam0" />);
    expect(first?.closed).toBe(false);
    unmount();
    expect(first?.closed).toBe(true);
    render(<Tile cam="cam1" />);
    await flush();
    expect(net.open()).toHaveLength(1);
  });

  it('follows a change of camera', async () => {
    const { rerender, getByTestId } = render(<Tile cam="cam0" />);
    await flush();
    rerender(
      <div data-testid="cam0">
        <canvas className="live-canvas" />
        <LiveBoxes cam="cam1" />
      </div>,
    );
    await flush();
    expect(net.conns[0]?.closed).toBe(true);
    net.ready('cam1', 2).overlay(us(), [face('Bob')]);
    step();
    expect(labelsIn(getByTestId('cam0'))).toEqual(['Bob']);
  });

  it('opens the session again after it drops, while anyone still listens', async () => {
    const { unmount } = render(<Tile cam="cam0" />);
    await flush();
    net.conns[0]?.fail();
    await act(() => vi.advanceTimersByTimeAsync(999));
    expect(net.conns).toHaveLength(1);
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(net.conns).toHaveLength(2);
    unmount();
    expect(net.open()).toHaveLength(0);
  });

  it('applies mirror to boxes already shown', async () => {
    const { rerender, getByTestId } = render(<Tile cam="cam0" />);
    await flush();
    const cv = getByTestId('cam0').querySelector('canvas') as HTMLCanvasElement;
    [cv.width, cv.height] = [400, 200]; // a 400x200 picture shown at that size
    Object.defineProperty(cv, 'clientWidth', { value: 400 });
    Object.defineProperty(cv, 'clientHeight', { value: 200 });
    net.ready('cam0', 1).overlay(us(), [face('Alice')]);
    step();
    const box = getByTestId('cam0').querySelector<HTMLElement>('.bx');
    expect(translated(box)).toEqual([40, 20]);
    rerender(<Tile cam="cam0" mirror />);
    step();
    expect(translated(box)).toEqual([320, 20]); // x 0.1 to 0.2 becomes 0.8 to 0.9
    expect(labelsIn(getByTestId('cam0'))).toEqual(['Alice']);
  });

  it('stops drawing when unmounted', () => {
    const { unmount } = render(<Tile cam="cam0" />);
    expect(frames.pending).toBe(1);
    unmount();
    expect(frames.pending).toBe(0);
  });
});

/** A recorded picture with its boxes. */
const Picture = ({ boxes }: { boxes: Box[] }) => (
  <div data-testid="pic">
    <img src="/clip/frame.jpg" alt="" />
    <BoxLayer boxes={boxes} ts={Date.now() / 1000} />
  </div>
);

describe('BoxLayer', () => {
  const cup: Box = { id: 'o1', box: [0.1, 0.1, 0.5, 0.5], label: 'cup', kind: 'object' };

  it('draws the boxes it is given over a recorded picture, and fades out the ones dropped', () => {
    const { getByTestId, rerender } = render(<Picture boxes={[cup]} />);
    step();
    const layer = getByTestId('pic').querySelector<HTMLElement>('.bx-layer');
    expect(layer?.style.display).toBe('');
    expect(labelsIn(getByTestId('pic'))).toEqual(['cup']);
    rerender(<Picture boxes={[]} />);
    for (let i = 0; i < 6; i++) step();
    expect(labelsIn(getByTestId('pic'))).toEqual([]);
  });

  it('stops drawing when unmounted', () => {
    const { unmount } = render(<Picture boxes={[cup]} />);
    unmount();
    expect(frames.pending).toBe(0);
  });

  describe('skeletons (#520)', () => {
    const lives = () => net.conns.flatMap(c => c.sent.filter(r => r.type === 'live').map(r => r.overlays));

    it('keeps the gateway subscription stable while skeleton visibility changes', async () => {
      const { rerender } = render(<Tile cam="cam0" />);
      await flush();
      net.ready('cam0', 1);
      rerender(<Tile cam="cam0" skeletons />);
      rerender(<Tile cam="cam0" />);
      expect(lives()).toEqual([['faces', 'objects', 'poses']]);
    });

    it('asks for skeletons from the first live when the view starts with them on', async () => {
      render(<Tile cam="cam0" skeletons />);
      await flush();
      net.ready('cam0', 1);
      expect(lives()).toEqual([['faces', 'objects', 'poses']]);
    });

    it("doesn't draw skeletons on a view that has them off, when another view's request brings them", async () => {
      const { getAllByTestId } = render(
        <>
          <Tile cam="cam0" />
          <Tile cam="cam0" skeletons />
        </>,
      );
      await flush();
      net.ready('cam0', 1).overlay(us(), [face('Alice')], [person()]);
      const [off, on] = getAllByTestId('cam0');
      expect(off?.querySelector('svg.bx-skeletons')).toBeNull();
      expect(on?.querySelector('svg.bx-skeletons')).not.toBeNull();
    });

    it('draws the bones and joints of the skeletons, mirrored with the picture, and clears them when switched off', async () => {
      const { rerender, getByTestId } = render(<Tile cam="cam0" skeletons />);
      await flush();
      const c = net.ready('cam0', 1);
      c.overlay(us(), [face('Alice')], [person()]);
      step();
      const svg = getByTestId('cam0').querySelector('svg.bx-skeletons');
      expect(svg?.querySelectorAll('line:not(.bx-joint)')).toHaveLength(1);
      expect(svg?.querySelectorAll('line.bx-joint')).toHaveLength(2);
      expect(svg?.querySelector('g')?.getAttribute('transform')).toBeNull();
      rerender(<Tile cam="cam0" skeletons mirror />);
      await act(() => vi.advanceTimersByTimeAsync(100));
      c.overlay(us(), [face('Alice')], [person()]);
      expect(getByTestId('cam0').querySelector('svg.bx-skeletons g')?.getAttribute('transform')).toBe(
        'matrix(-1 0 0 1 1 0)',
      );
      rerender(<Tile cam="cam0" />);
      expect(getByTestId('cam0').querySelector('svg.bx-skeletons')).toBeNull();
    });

    it('fades skeletons away once none fresh arrive (stale, or nobody in view), rather than blinking (#592)', async () => {
      const { getByTestId } = render(<Tile cam="cam0" skeletons />);
      await flush();
      const c = net.ready('cam0', 1);
      c.overlay(us(), [face('Alice')], [person()]);
      expect(getByTestId('cam0').querySelector('svg.bx-skeletons')).not.toBeNull();
      for (let i = 0; i < 5; i++) step(); // shown for a moment, faded in
      await act(() => vi.advanceTimersByTimeAsync(3000)); // past the 2 s the latest skeletons are carried
      c.overlay(us(), [face('Alice')]); // a frame with no poses
      step();
      expect(getByTestId('cam0').querySelector('svg.bx-skeletons')).not.toBeNull(); // still there, fading
      for (let i = 0; i < 10; i++) step();
      expect(getByTestId('cam0').querySelector('svg.bx-skeletons')).toBeNull();
    });
  });
});
