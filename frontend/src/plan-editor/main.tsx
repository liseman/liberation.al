// The plan editor's page: Pascal Editor (its 2D editor and its 3D view) on one scene, in an iframe of the Places
// page. It is a page of its own so that the editor's megabytes load only when someone opens it and its styles stay
// its own. The host gives it a scene and hears each change (protocol.ts). Nothing here reaches another origin:
// assets, decoders, and icons are the dashboard's own (vite.pascal.ts, public/pascal/).
import './plan-editor.css';
import icons from 'virtual:pascal-icons';
import { addCollection } from '@iconify/react';
import { loadPlugin, useScene } from '@pascal-app/core';
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
const { view, scheme } = fromHash(location.hash);

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
  { id: 'build', label: 'Build', icon: tabIcon('wall'), component: BuildTab },
  { id: 'cameras', label: 'Cameras', icon: tabIcon('topview'), component: CamerasTab },
];

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
}

function App() {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Editor
        layoutVersion="v2"
        projectId="place-plan"
        sidebarTabs={TABS}
        floorplanSceneSlot={<CameraMarks />}
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
