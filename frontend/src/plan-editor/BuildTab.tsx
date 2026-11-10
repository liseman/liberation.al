// The editor's Build panel: the tools a home's plan needs and no others. Pascal's own host app has a panel with
// several dozen (roofs, ducts, stairs, fences); this one draws rooms and walls and puts doors and windows in them,
// from the same parts (`BuildPanelRoomsSection`, `BuildToolTile`).
import {
  BuildPanelRoomsSection,
  BuildPanelSection,
  BuildToolGrid,
  BuildToolTile,
  selectWallDrawVariant,
  useEditor,
  useWallDrawVariant,
} from '@pascal-app/editor';
import { useViewer } from '@pascal-app/viewer';

/** One of the editor's own icons, where the dashboard serves them (public/pascal/icons/). */
export const icon = (name: string): string => `${import.meta.env.BASE_URL}pascal/icons/${name}.webp`;

/** The tools that go on a wall, by their node kind. */
export const OPENINGS = [
  { kind: 'door', label: 'Door', iconSrc: icon('door') },
  { kind: 'window', label: 'Window', iconSrc: icon('window') },
] as const;

/** Start drawing with the tool for `kind`, as the editor's own panel does: nothing selected, in build mode. */
export function startTool(kind: string): void {
  const editor = useEditor.getState();
  useViewer.getState().setSelection({ selectedIds: [], zoneId: null });
  editor.setPhase('building');
  editor.setCatalogCategory(null);
  editor.setToolDefaults(kind, null);
  editor.setMode('build');
  editor.setTool(kind);
}

export function BuildTab() {
  const tool = useEditor(s => (s.mode === 'build' ? s.tool : null));
  const variant = useWallDrawVariant();
  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-3">
      <BuildPanelRoomsSection
        activeVariant={tool === 'wall' ? variant : null}
        onSelect={next => {
          selectWallDrawVariant(next);
          if (useEditor.getState().tool !== 'wall' || useEditor.getState().mode !== 'build') startTool('wall');
        }}
      />
      <BuildPanelSection id="add" title="Add to rooms">
        <BuildToolGrid columns={4}>
          {OPENINGS.map(o => (
            <BuildToolTile
              key={o.kind}
              active={tool === o.kind}
              iconSrc={o.iconSrc}
              label={o.label}
              onClick={() => startTool(o.kind)}
            />
          ))}
        </BuildToolGrid>
      </BuildPanelSection>
    </div>
  );
}
