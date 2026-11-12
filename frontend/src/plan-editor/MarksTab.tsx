// The editor's Marks panel: the objects and unexplored areas on the plan, a field for the selected one's label, and a
// way to add one or remove it. Each change is one write to the scene (useAreas.ts), so it undoes and saves like any
// other edit. An object's label is what the Places page shows for it; an unexplored area's name is its label too.
import { type Area, type AreaKind, box, newId } from './areas';
import { isSelected, kindName, kindWord, markName, useAreasUi } from './areasUi';
import { middle } from './spots';
import { useAreas } from './useAreas';

const INPUT =
  'h-8 w-full rounded-md border border-border bg-background px-2 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring';
const BUTTON =
  'rounded-md border border-border px-2 py-1 text-xs hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50';

/** One kind's section: its list of records, and the way to add another. */
function Section({ kind, title, hint }: { kind: AreaKind; title: string; hint: string }) {
  const { scene, areas, set } = useAreas();
  const { selected, adding, select, setAdding } = useAreasUi();
  const found = areas[kind];
  const centre = middle(scene);
  return (
    <section className="flex flex-col gap-2">
      <h3 className="font-medium text-foreground text-xs">{title}</h3>
      {found.length === 0 ? (
        <p className="text-[11px] text-muted-foreground">{hint}</p>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {found.map(area => {
            const on = isSelected(selected, kind, area.id);
            return (
              <button
                key={area.id}
                type="button"
                aria-pressed={on}
                className={`${BUTTON} ${on ? 'border-transparent bg-primary text-primary-foreground hover:bg-primary' : ''}`}
                onClick={() => select(kind, on ? null : area.id)}
              >
                {markName(area)}
              </button>
            );
          })}
        </div>
      )}
      <button type="button" className={`${BUTTON} w-fit`} onClick={() => setAdding(kind)}>
        Add {kind === 'object' ? 'an object' : 'an unexplored area'}
      </button>
      {adding === kind && (
        <p className="text-[11px] text-muted-foreground">
          Click the plan where it goes (2D view, select tool), or put it in the middle.
        </p>
      )}
      {/* a way to add without finding the plan first: a new one in the middle of the plan */}
      {adding === kind && (
        <button
          type="button"
          className={`${BUTTON} w-fit`}
          disabled={!centre}
          onClick={() => {
            if (!centre) return;
            const area: Area = { id: newId(kind), kind, label: '', poly: box(centre) };
            set(kind, area.id, area);
            select(kind, area.id);
            setAdding(null);
          }}
        >
          Put it in the middle
        </button>
      )}
    </section>
  );
}

export function MarksTab() {
  const { areas, set } = useAreas();
  const { selected, select } = useAreasUi();
  const area = selected ? areas[selected.kind].find(a => a.id === selected.id) : undefined;
  const change = (patch: Partial<Area>) => {
    if (area) set(area.kind, area.id, { ...area, ...patch });
  };
  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-3" data-surface="marks-tab">
      <Section kind="object" title="Objects on the plan" hint="No objects yet. Add one and drag it into place." />
      <Section
        kind="unexplored"
        title="Unexplored areas"
        hint="No unexplored areas yet. Mark a part of the house the cameras haven't seen."
      />
      {selected && area && (
        <section className="flex flex-col gap-2 border-border/60 border-t pt-3">
          <h3 className="font-medium text-foreground text-xs">{kindName(selected.kind)}</h3>
          <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
            Label
            <input
              className={INPUT}
              type="text"
              placeholder="What is it? (sofa, bookcase)"
              value={area.label}
              onChange={event => change({ label: event.target.value })}
              onKeyDown={event => event.stopPropagation()} // the editor's own shortcuts are single keys
            />
          </label>
          <button
            type="button"
            className={`${BUTTON} w-fit text-destructive`}
            onClick={() => {
              set(area.kind, area.id, null);
              select(null);
            }}
          >
            Remove this {kindWord(area.kind)}
          </button>
        </section>
      )}
      <p className="text-[11px] text-muted-foreground">
        An object is a labelled box on the plan. An unexplored area is a shape the house hasn't seen yet. Drag either to
        move it; drag a corner to reshape it. They are saved with the plan.
      </p>
    </div>
  );
}
