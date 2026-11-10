// A camera, live: frames from its gateway session (mux.ts) drawn on a canvas. Until the first frame arrives, a note
// says there's no picture yet; when frames stop for 4 s, the last one stays, dimmed, until they come back.
import { attach, useLiveError, useLiveNow } from './mux';
import './camera.css';
import { type Ref, type RefObject, useCallback, useEffect, useLayoutEffect, useRef } from 'react';

const ASPECT_KEY = 'liberational.cameraAspect';

/** Each camera's last shape in this browser: a large camera on Live's panel grid is as wide as its height allows,
 *  so starting from 16:9 and then learning the camera's own shape moved it sideways on every load. */
function rememberedAspects(): Record<string, number> {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(ASPECT_KEY) ?? '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, number>) : {};
  } catch {
    return {}; // storage can be unavailable (private mode) or hold junk
  }
}

function rememberAspect(cam: string, aspect: number): void {
  try {
    const all = rememberedAspects();
    if (Math.abs((all[cam] ?? 0) - aspect) < 0.001) return;
    localStorage.setItem(ASPECT_KEY, JSON.stringify({ ...all, [cam]: aspect }));
  } catch {
    // not remembered: the next load starts from 16:9 again
  }
}

/** Follow displayed canvas dimensions, including a stream changing orientation after its first picture. The shape
 *  last seen in this browser is used until the first picture says otherwise. */
export function useCameraAspect<T extends HTMLElement>(cam: string): RefObject<T | null> {
  const ref = useRef<T>(null);
  useLayoutEffect(() => {
    const root = ref.current;
    if (!root) return;
    const show = (aspect: number) => {
      root.style.setProperty('--camera-aspect', String(aspect));
      root.dataset.orientation = aspect < 1 ? 'portrait' : 'landscape';
    };
    const known = rememberedAspects()[cam];
    if (typeof known === 'number' && Number.isFinite(known) && known > 0) show(known);
    const size = () => {
      const canvas = [...root.querySelectorAll<HTMLCanvasElement>('canvas.live-canvas, canvas.preview-canvas')].find(
        picture => picture.dataset.cam === cam,
      );
      if (!canvas?.width || !canvas.height) return;
      // No picture yet: a canvas starts at the HTML default 300 x 150, which is no camera's shape
      if (canvas.width === 300 && canvas.height === 150) return;
      show(canvas.width / canvas.height);
      rememberAspect(cam, canvas.width / canvas.height);
    };
    const observer = new MutationObserver(size);
    observer.observe(root, { attributes: true, attributeFilter: ['width', 'height'], childList: true, subtree: true });
    size();
    return () => observer.disconnect();
  }, [cam]);
  return ref;
}

/** The canvas, and the note in its place before any frame: a LiveBoxes beside them finds the picture in their
 *  parent. */
export function LiveCam({ cam, alt, canvasRef }: { cam: string; alt: string; canvasRef?: Ref<HTMLCanvasElement> }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const setCanvas = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      ref.current = canvas;
      if (typeof canvasRef === 'function') return canvasRef(canvas);
      if (canvasRef) canvasRef.current = canvas;
    },
    [canvasRef],
  );
  useEffect(() => attach(cam, ref.current), [cam]);
  // undefined before the camera's first frame, true while frames arrive, false once they've stopped.
  const live = useLiveNow(s => s[cam]);
  const error = useLiveError(s => s[cam]);
  const cls = live === undefined ? 'live-canvas off' : live && !error ? 'live-canvas' : 'live-canvas stale';
  return (
    <>
      <canvas key={cam} ref={setCanvas} className={cls} data-cam={cam} role="img" aria-label={alt} />
      {error && (
        <div className="snap-missing" role="alert">
          Video unavailable: {error}
        </div>
      )}
      {live === undefined && !error && (
        <div className="snap-missing" role="img" aria-label={`${alt}: no picture from this camera yet`}>
          No picture from this camera yet
        </div>
      )}
    </>
  );
}
