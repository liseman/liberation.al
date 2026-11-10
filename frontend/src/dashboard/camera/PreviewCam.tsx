import { useEffect, useRef, useState } from 'react';
import { reloadForNewBuild } from '../lib/staleChunk';
import type { PreviewState } from './preview';
import './preview.css';

/** An overview thumbnail only; selecting its surrounding camera control opens the full Live view. */
export function PreviewCam({ cam, alt }: { cam: string; alt: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<PreviewState>({ status: 'loading', capturedAt: null, error: null });
  useEffect(() => {
    const target = canvas.current;
    let closed = false;
    let off: (() => void) | undefined;
    // A single focused camera does not need the thumbnail scheduler or its read connection.
    void import('./preview')
      .then(module => {
        // Vite resolves a stale-build import with nothing once the page is reloading for the new build.
        if (!closed && target && module) off = module.attachPreview(cam, target, setState);
      })
      .catch(error => {
        // A tab left open across a dashboard rebuild asks for a chunk that's gone: reload once for the new build
        // instead of showing "Preview unavailable" until someone reloads by hand.
        if (reloadForNewBuild(error)) return;
        if (!closed) setState({ status: 'unavailable', capturedAt: null, error: 'Preview unavailable' });
      });
    return () => {
      closed = true;
      off?.();
    };
  }, [cam]);
  const hasPicture = state.capturedAt !== null;
  const age = state.capturedAt === null ? null : Math.max(0, Math.round(Date.now() / 1000 - state.capturedAt));
  return (
    <>
      <canvas
        key={cam}
        ref={canvas}
        data-cam={cam}
        data-status={state.status}
        className={`preview-canvas${hasPicture ? (state.status === 'ready' ? '' : ' stale') : ' off'}`}
        role="img"
        aria-label={alt}
      />
      {!hasPicture && (
        <span className="preview-missing">
          {state.status === 'loading' ? 'Loading preview…' : 'Preview unavailable'}
        </span>
      )}
      <span className="preview-status" aria-live="off">
        {hasPicture ? `${state.status === 'ready' ? 'Preview' : 'Older preview'} · ${age} s ago` : 'Preview'}
        {hasPicture && state.error && <span>Open camera for live video</span>}
      </span>
    </>
  );
}
