// The plan editor's page: Pascal Editor (its 2D editor and its 3D view) on one scene, in an iframe of the Places
// page. It is a page of its own so that the editor's megabytes load only when someone opens it and its styles stay
// its own. The host gives it a scene and hears each change (protocol.ts). Nothing here reaches another origin:
// assets, decoders, and icons are the dashboard's own (vite.pascal.ts, public/pascal/).
import './plan-editor.css';
import icons from 'virtual:pascal-icons';
import { addCollection } from '@iconify/react';
import { emitter, loadPlugin, useScene } from '@pascal-app/core';
import { Editor, selectDefaultBuildingAndLevel, useAudio, useEditor } from '@pascal-app/editor';
import { registerBundledLocales } from '@pascal-app/editor/locales';
import { builtinPlugin } from '@pascal-app/nodes';
import { configureViewer, useViewer } from '@pascal-app/viewer';
import { useGLTF } from '@react-three/drei';
import { createRoot } from 'react-dom/client';
import { BuildTab, icon } from './BuildTab';
import { CameraMarks } from './CameraMarks';
import { CamerasTab } from './CamerasTab';
import { type Changed, fromHash, type Open, ours, type Ready, type Scene } from './protocol';
import { useSpotsUi } from './spotsUi';

const assets = `${location.origin}${import.meta.env.BASE_URL}pascal`;
const { view, scheme, look } = fromHash(location.hash);

// The editor's chrome follows a `dark` class, and its menus are added to <body>.
document.documentElement.classList.toggle('dark', scheme === 'dark');
document.body.classList.toggle('dark', scheme === 'dark');

for (const collection of icons) addCollection(collection);
configureViewer({
  assetsBaseUrl: assets,
  storageUrl: assets,
  decoders: { dracoPath: `${assets}/decoders/draco/`, basisPath: `${assets}/decoders/basis/` },
});
useGLTF.setDecoderPath(`${assets}/decoders/draco/`);
registerBundledLocales();
useViewer.getState().setUnit('metric');
// the plan's canvas and the 3D scene follow the dashboard's scheme: the editor keeps that apart from its chrome
useViewer.getState().setSceneTheme(scheme === 'dark' ? 'twilight' : 'studio');
useAudio.getState().setMuted(true); // its interface sounds are files we don't ship
await loadPlugin(builtinPlugin);

/** The scene the host opens this page on: asked for once the page is up, and given by message. */
const opened = new Promise<Scene>(resolve => {
  const listen = (event: MessageEvent) => {
    const message = ours<Open>(event, ['plan-editor:open']);
    if (!message) return;
    removeEventListener('message', listen);
    useSpotsUi.getState().setCameras(message.cameras ?? []);
    resolve(message.scene);
  };
  addEventListener('message', listen);
});
const tell = (message: Ready | Changed) => parent.postMessage(message, location.origin);
tell({ type: 'plan-editor:ready' });

const tabIcon = (name: string) => <img alt="" className="h-8 w-8 object-contain" src={icon(name)} />;
const TABS = [
  // the editor's own panel of levels and rooms
  { id: 'site', label: 'Plan', icon: tabIcon('floorplan'), component: () => null },
  // a scene opened to look at has nothing that changes it: no Build panel and no Cameras panel
  ...(look
    ? []
    : [
        { id: 'build', label: 'Build', icon: tabIcon('wall'), component: BuildTab },
        { id: 'cameras', label: 'Cameras', icon: tabIcon('topview'), component: CamerasTab },
      ]),
];

/** The box around the plan's walls and rooms (metres), or null with neither. */
function planBounds(): { min: [number, number, number]; max: [number, number, number] } | null {
  const xs: number[] = [];
  const zs: number[] = [];
  let height = 0;
  for (const node of Object.values(useScene.getState().nodes) as unknown as Record<string, unknown>[]) {
    const points = node.type === 'wall' ? [node.start, node.end] : node.type === 'zone' ? node.polygon : null;
    if (typeof node.height === 'number') height = Math.max(height, node.height);
    for (const p of Array.isArray(points) ? points : [])
      if (Array.isArray(p) && typeof p[0] === 'number' && typeof p[1] === 'number') {
        xs.push(p[0]);
        zs.push(p[1]);
      }
  }
  if (!xs.length) return null;
  return { min: [Math.min(...xs), 0, Math.min(...zs)], max: [Math.max(...xs), height || 2.4, Math.max(...zs)] };
}

/** How long after a scene opens the views have laid themselves out and can be asked to frame the plan. */
const FRAME_AFTER_MS = 400;

/** Run `then` once the editor holds a scene: it sets its own layout and selection as it loads one, so what this
 *  page wants (the storey to work on, and the view asked for) goes in after. */
function onceLoaded(then: () => void): void {
  if (useScene.getState().rootNodeIds.length) {
    then();
    return;
  }
  const stop = useScene.subscribe(state => {
    if (!state.rootNodeIds.length) return;
    stop();
    then();
  });
}

function show(): void {
  const editor = useEditor.getState();
  editor.setPhase('building');
  selectDefaultBuildingAndLevel();
  if (view === 'split') {
    editor.showView('2d');
    editor.showView('3d', { beside: true });
  } else editor.showView(view);
  // Fill the view with the plan itself: the editor's own first fit is to the whole site, which leaves a home small.
  setTimeout(() => {
    const bounds = planBounds();
    if (bounds) emitter.emit('camera-controls:frame', { bounds });
  }, FRAME_AFTER_MS);
}

function App() {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Editor
        layoutVersion="v2"
        projectId="place-plan"
        sidebarTabs={TABS}
        floorplanSceneSlot={<CameraMarks locked={look} />}
        isVersionPreviewMode={look}
        onLoad={async () => {
          const scene = await opened;
          onceLoaded(show);
          return scene;
        }}
        onSave={async scene => tell({ type: 'plan-editor:changed', scene })}
      />
    </div>
  );
}

createRoot(document.getElementById('plan-editor') as HTMLElement).render(<App />);
