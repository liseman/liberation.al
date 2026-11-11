import { act, render, screen } from '@testing-library/react';
import { createRef } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fakeBitmaps, fakeGateway } from './testFakes';

// mux.ts and gateway.ts keep their state at module level, so each test loads fresh copies.
let LiveCam: typeof import('./LiveCam').LiveCam;
let net: ReturnType<typeof fakeGateway>;
const settle = (ms = 0) => act(() => vi.advanceTimersByTimeAsync(ms));

beforeEach(async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
  vi.resetModules();
  const gw = await import('./gateway');
  net = fakeGateway();
  gw.setGatewayForTests(net.connect);
  ({ LiveCam } = await import('./LiveCam'));
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  delete (window as Window & { __PACK_DEMO__?: boolean }).__PACK_DEMO__;
});

const canvas = () => document.querySelector('canvas') as HTMLCanvasElement;
const note = () => screen.queryByRole('img', { name: 'The front door: no picture from this camera yet' });

describe('LiveCam', () => {
  it('exposes the displayed canvas and clears capture identity when switching cameras', async () => {
    fakeBitmaps();
    const ref = createRef<HTMLCanvasElement>();
    const { rerender, unmount } = render(<LiveCam cam="cam0" alt="The front door" canvasRef={ref} />);
    await settle();
    net.ready('cam0', 1).jpeg(1_234_567, new Uint8Array(3));
    await settle();
    const first = canvas();
    expect(ref.current).toBe(first);
    expect(first.dataset.capturedAt).toBe('1.234567');
    rerender(<LiveCam cam="cam1" alt="The back door" canvasRef={ref} />);
    await settle();
    expect(ref.current).toBe(canvas());
    expect(canvas().dataset.cam).toBe('cam1');
    expect(canvas().dataset.capturedAt).toBeUndefined();
    unmount();
    expect(ref.current).toBeNull();
  });

  it('is a labelled canvas for its camera', async () => {
    render(<LiveCam cam="cam0" alt="The front door" />);
    expect(screen.getAllByRole('img', { name: 'The front door' })).toContain(canvas());
    expect(canvas().dataset.cam).toBe('cam0');
    await settle();
  });

  it('says there is no picture yet until the first frame, then shows the canvas', async () => {
    fakeBitmaps();
    render(<LiveCam cam="cam0" alt="The front door" />);
    expect(canvas()).toHaveClass('live-canvas', 'off');
    expect(note()).toBeInTheDocument();
    await settle();
    net.ready('cam0', 1).jpeg(1, new Uint8Array(3));
    await settle();
    expect(canvas()).toHaveClass('live-canvas');
    expect(canvas()).not.toHaveClass('off');
    expect(note()).toBeNull();
  });

  it('keeps the last frame, dimmed, 4 s after frames stop, and livens it when they come back', async () => {
    fakeBitmaps();
    render(<LiveCam cam="cam0" alt="The front door" />);
    await settle();
    const c = net.ready('cam0', 1);
    c.jpeg(1, new Uint8Array(3));
    await settle(5000);
    expect(canvas()).toHaveClass('live-canvas', 'stale');
    expect(note()).toBeNull();
    c.jpeg(2, new Uint8Array(3));
    await settle();
    expect(canvas()).not.toHaveClass('stale');
  });

  it('shows a decode failure explicitly and restores the picture state after recovery', async () => {
    const { useLiveError } = await import('./mux');
    fakeBitmaps();
    const { unmount } = render(<LiveCam cam="rtsp-entry" alt="The entry camera" />);
    await settle();
    act(() => useLiveError.setState({ 'rtsp-entry': 'This browser cannot decode H.264 video.' }));
    expect(screen.getByRole('alert')).toHaveTextContent('This browser cannot decode H.264 video.');
    expect(screen.queryByText('No picture from this camera yet')).not.toBeInTheDocument();
    net.ready('rtsp-entry', 1).jpeg(1_234_567, new Uint8Array(3));
    await settle();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(canvas()).not.toHaveClass('off', 'stale');
    unmount();
  });

  it("lets go of the camera's session when unmounted", async () => {
    const { unmount } = render(<LiveCam cam="cam0" alt="The front door" />);
    await settle();
    unmount();
    expect(net.open()).toHaveLength(0);
    await settle(30_000);
    expect(net.conns).toHaveLength(1);
  });
});

/** A frame around one camera's canvas, following its shape (the hook comes from each test's fresh module). */
function Framed({ cam, hook }: { cam: string; hook: typeof import('./LiveCam').useCameraAspect }) {
  const ref = hook<HTMLDivElement>(cam);
  return (
    <div ref={ref} data-testid="frame">
      <canvas className="live-canvas" data-cam={cam} />
    </div>
  );
}

describe('useCameraAspect', () => {
  it("starts from the camera's shape last seen in this browser, then keeps the one its picture shows", async () => {
    const { useCameraAspect } = await import('./LiveCam');
    localStorage.setItem('liberational.cameraAspect', JSON.stringify({ desk: 0.75 }));
    render(<Framed cam="desk" hook={useCameraAspect} />);
    const frame = screen.getByTestId('frame');
    expect(frame.style.getPropertyValue('--camera-aspect')).toBe('0.75');
    expect(frame.dataset.orientation).toBe('portrait');
    // its first picture is 4:3: shown, and remembered for the next load
    act(() => {
      const c = frame.querySelector('canvas') as HTMLCanvasElement;
      c.width = 640;
      c.height = 480;
    });
    await settle();
    expect(frame.style.getPropertyValue('--camera-aspect')).toBe(String(640 / 480));
    expect(JSON.parse(localStorage.getItem('liberational.cameraAspect') ?? '{}')).toEqual({ desk: 640 / 480 });
  });
});

it('draws the demo still without connecting to a camera gateway', async () => {
  (window as Window & { __PACK_DEMO__?: boolean }).__PACK_DEMO__ = true;
  const picture = { onload: null as (() => void) | null, src: '', naturalWidth: 960, naturalHeight: 540 };
  function fakeImage() {
    return picture;
  }
  vi.stubGlobal('Image', fakeImage);
  const draw = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage: draw,
  } as unknown as CanvasRenderingContext2D);
  const ref = createRef<HTMLCanvasElement>();
  const { unmount } = render(<LiveCam cam="kitchen" alt="Kitchen" canvasRef={ref} />);
  expect(picture.src).toBe('/demo/media/kitchen.jpg');
  act(() => picture.onload?.());
  expect(canvas().width).toBe(960);
  expect(canvas().height).toBe(540);
  expect(draw).toHaveBeenCalledWith(picture, 0, 0);
  expect(screen.getByRole('img', { name: 'Kitchen, demo illustration' })).toBe(canvas());
  await settle();
  expect(net.conns).toHaveLength(0);
  unmount();
  expect(ref.current).toBeNull();
});
