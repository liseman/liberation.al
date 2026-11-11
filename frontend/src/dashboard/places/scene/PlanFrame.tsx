// A place's plan in Pascal Editor's 2D editor, in an iframe on the editor's own page (plan-editor/index.html,
// src/plan-editor/). The iframe is what keeps the editor out of every other page: nothing of it is imported here, so it
// is fetched only when this is shown, and its styles stay inside its own document. The page itself also knows the 3D
// and split views (`PlanView`, protocol.ts), which nothing at this end asks for.
import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { editorUrl, type FromEditor, type Open, ours, type Scene } from '../../../plan-editor/protocol';
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
  // The editor takes what's left of the window under where it starts, as the 3D view does (CloudView.tsx): measured,
  // since what's above it (the tabs, the place's header, Save, and any message) is the page's to decide, and measured
  // again after each render (a message above it comes and goes) and when the window changes.
  const [top, setTop] = useState(0);
  const measure = () => {
    const el = frame.current;
    if (el) setTop(Math.round(el.getBoundingClientRect().top + window.scrollY));
  };
  useLayoutEffect(measure);
  // biome-ignore lint/correctness/useExhaustiveDependencies: measure reads the frame's ref, not a value of a render
  useEffect(() => {
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);
  return (
    <>
      {!ready && <BlockSkeleton label="the plan editor" />}
      <iframe
        ref={frame}
        className={classes.planFrame}
        style={{ '--plan-top': `${top}px` } as CSSProperties}
        title="Plan editor"
        src={editorUrl(import.meta.env.BASE_URL, '2d', scheme, look)}
      />
    </>
  );
}
