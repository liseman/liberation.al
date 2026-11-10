// A layer of boxes over the picture it sits beside (both inside a positioned parent), for pictures whose boxes come
// as a list, like a recording's sightings.
import { useEffect, useRef } from 'react';
import { type Box, type BoxPick, BoxRenderer } from './boxes';
import './camera.css';

export interface BoxLayerProps {
  boxes: readonly Box[] | null | undefined;
  /** When the picture was captured (wall seconds), if known. */
  ts?: number | null;
  mirror?: boolean;
  onPick?: ((pick: BoxPick) => void) | null;
  /** The pointer came onto a box (with it) or left it (null), for a mouse; touch gets onPick alone. */
  onHover?: ((pick: BoxPick | null) => void) | null;
  /** Pets can be clicked too. */
  pickPets?: boolean;
  /** Which faces can be clicked: only unknown ones (the default), or every face (Review's). */
  pickFaces?: 'unknown' | 'all';
  /** How long boxes stay with no update; Review keeps them while paused (Infinity). */
  staleS?: number;
}

export function BoxLayer({
  boxes,
  ts,
  mirror = false,
  onPick = null,
  onHover = null,
  pickPets = false,
  pickFaces = 'unknown',
  staleS,
}: BoxLayerProps) {
  const ref = useRef<HTMLDivElement>(null);
  const r = useRef<BoxRenderer | null>(null);
  const made = useRef({ pickFaces, pickPets, staleS }); // read when the renderer is made
  useEffect(() => {
    if (!ref.current) return;
    const renderer = new BoxRenderer(ref.current, made.current);
    r.current = renderer;
    return () => {
      renderer.close();
      r.current = null;
    };
  }, []);
  useEffect(() => {
    if (!r.current) return;
    r.current.mirror = mirror;
    r.current.onPick = onPick;
    r.current.onHover = onHover;
    r.current.wake(); // drawn again with them
  }, [mirror, onPick, onHover]);
  useEffect(() => r.current?.push(boxes ?? [], ts), [boxes, ts]);
  return <div className="bx-layer" ref={ref} />;
}
