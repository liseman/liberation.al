// A place's plan in Pascal Editor: its 2D editor or its 3D view, in an iframe on the editor's own page
// (plan-editor/index.html, src/plan-editor/). The iframe is what keeps the editor out of every other page: nothing of it
// is imported here, so it is fetched only when this is shown, and its styles stay inside its own document.
import { Text } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { editorUrl, type FromEditor, type Open, ours, type PlanView, type Scene } from '../../../plan-editor/protocol';
import classes from '../places.module.css';

export function PlanFrame({
  scene,
  view,
  scheme,
  cameras,
  onChange,
}: {
  /** The scene to open. A later one is ignored: the editor holds the plan from then on, and remounting opens another. */
  scene: Scene;
  view: PlanView;
  scheme: 'light' | 'dark';
  /** The house's cameras, for the editor to offer the ones with no spot on the plan yet. */
  cameras?: { id: string; name: string }[];
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
  return (
    <>
      {!ready && (
        <Text size="sm" c="dimmed">
          Loading the plan editor…
        </Text>
      )}
      <iframe
        ref={frame}
        className={classes.planFrame}
        title={view === '3d' ? 'The plan in 3D' : 'Plan editor'}
        src={editorUrl(import.meta.env.BASE_URL, view, scheme)}
      />
    </>
  );
}
