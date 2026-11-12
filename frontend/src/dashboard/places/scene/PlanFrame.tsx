// A place's plan in Pascal Editor's 2D editor, in an iframe on the editor's own page (plan-editor/index.html,
// src/plan-editor/). The iframe is what keeps the editor out of every other page: nothing of it is imported here, so it
// is fetched only when this is shown, and its styles stay inside its own document. The page itself also knows the 3D
// and split views (`PlanView`, protocol.ts), which nothing at this end asks for.
import { useEffect, useRef, useState } from 'react';
import { editorUrl, type FromEditor, type Open, ours, type Scene } from '../../../plan-editor/protocol';
import { useFillsPage } from '../../lib/fill';
import { BlockSkeleton } from '../../lib/loading';
import classes from '../places.module.css';

export function PlanFrame({
  scene,
  scheme,
  cameras,
  look = false,
  onChange,
}: {
  /** The scene to open. A later one is ignored: the editor holds the plan from then on, and remounting opens another. */
  scene: Scene;
  scheme: 'light' | 'dark';
  /** The house's cameras, for the editor to offer the ones with no spot on the plan yet. */
  cameras?: { id: string; name: string }[];
  /** Open the scene to look at only: the editor shows it with nothing that changes it. */
  look?: boolean;
  /** The scene after each edit. */
  onChange?: (scene: Scene) => void;
}) {
  const frame = useRef<HTMLIFrameElement>(null);
  // The editor takes what's left of the window under where it starts (lib/fill.ts measures that), down to the window's
  // bottom edge: data-fills-page has the page give up the room it kept under it there.
  const fillsStyle = useFillsPage(frame);
  const first = useRef(scene);
  const known = useRef(cameras);
  known.current = cameras;
  const changed = useRef(onChange);
  changed.current = onChange;
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const listen = (event: MessageEvent) => {
      if (event.source !== frame.current?.contentWindow) return;
      const message = ours<FromEditor>(event, ['plan-editor:ready', 'plan-editor:changed']);
      if (message?.type === 'plan-editor:ready') {
        const open: Open = { type: 'plan-editor:open', scene: first.current, cameras: known.current };
        frame.current?.contentWindow?.postMessage(open, location.origin);
        setReady(true);
      } else if (message?.type === 'plan-editor:changed') changed.current?.(message.scene);
    };
    addEventListener('message', listen);
    return () => removeEventListener('message', listen);
  }, []);
  return (
    <>
      {!ready && <BlockSkeleton label="the plan editor" />}
      <iframe
        ref={frame}
        className={classes.planFrame}
        data-fills-page
        style={fillsStyle}
        title="Plan editor"
        src={editorUrl(import.meta.env.BASE_URL, '2d', scheme, look)}
      />
    </>
  );
}
