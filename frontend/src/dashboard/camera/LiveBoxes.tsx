// The live boxes for one camera, over the live picture beside this layer, from the one shared boxes stream.
import { useEffect, useRef } from 'react';
import { type Box, type BoxPick, BoxRenderer, liveBoxes, subscribe, wantSkeletons } from './boxes';
import './camera.css';

export interface LiveBoxesProps {
  cam: string;
  mirror?: boolean;
  onPick?: ((pick: BoxPick) => void) | null;
  /** Change a box before it is drawn (Live's large view labels an unknown face "Who's this?", #352). With it, every
   *  face is pickable, not only unknown ones. */
  relabel?: ((box: Box) => Box) | null;
  /** Draw the people's skeletons too (#520): vision then poses every frame it analyzes. */
  skeletons?: boolean;
  pickObjects?: boolean;
}

export function LiveBoxes({
  cam,
  mirror = false,
  onPick = null,
  relabel = null,
  skeletons = false,
  pickObjects = false,
}: LiveBoxesProps) {
  const ref = useRef<HTMLDivElement>(null);
  const r = useRef<BoxRenderer | null>(null);
  const opts = useRef({ mirror, onPick }); // the latest, for a renderer made when cam changes
  const change = useRef(relabel);
  change.current = relabel;
  const drawSkeletons = useRef(skeletons);
  drawSkeletons.current = skeletons;
  // Before the boxes stream is subscribed, so a first mount opens it once, already asking for skeletons.
  useEffect(() => {
    if (!skeletons) {
      r.current?.clearSkeletons(); // switched off: gone at once, not faded
      return;
    }
    return wantSkeletons(cam);
  }, [cam, skeletons]);
  useEffect(() => {
    if (!ref.current) return;
    const renderer = new BoxRenderer(ref.current, {
      ...opts.current,
      pickFaces: change.current ? 'all' : 'unknown',
      pickObjects,
    });
    r.current = renderer;
    let poseTs: number | undefined;
    const off = subscribe(cam, d => {
      const boxes = liveBoxes(d);
      renderer.push(change.current ? boxes.map(change.current) : boxes, d.ts);
      if (d.pose_ts !== poseTs) {
        poseTs = d.pose_ts; // skeletons change only when a pose turn lands, not with every frame
        // Another view's request can bring this camera's skeletons; a view with them off still doesn't draw them.
        renderer.skeletons(drawSkeletons.current ? (d.poses ?? null) : null);
      }
    });
    return () => {
      off();
      renderer.close();
      r.current = null;
    };
  }, [cam, pickObjects]);
  useEffect(() => {
    opts.current = { mirror, onPick };
    if (!r.current) return;
    r.current.mirror = mirror;
    r.current.onPick = onPick;
    r.current.wake(); // drawn again with them
  }, [mirror, onPick]);
  return <div className="bx-layer" ref={ref} />;
}
