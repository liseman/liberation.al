// The editor's Cameras panel: the camera spots on the plan, a form for the one selected (where it is, how high, which
// way it looks, and how wide it sees), and the house's cameras that have no spot yet, to add. Each change is one
// write to the scene (useSpots.ts). People are placed from a saved spot (spatial/spots.py), so a facing matters: a
// spot without one places nobody.
import { useEffect, useState } from 'react';
import { middle, type Spot } from './spots';
import { cameraName, useSpotsUi } from './spotsUi';
import { useSpots } from './useSpots';

const INPUT =
  'h-8 w-full rounded-md border border-border bg-background px-2 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring';
const BUTTON =
  'rounded-md border border-border px-2 py-1 text-xs hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50';

/** A number field that writes when it is left or Enter is pressed. Empty clears an optional number, and puts a
 *  required one back as it was. */
function Field({
  label,
  value,
  step,
  optional,
  onSet,
}: {
  label: string;
  value: number | undefined;
  step: number;
  optional?: boolean;
  onSet: (value: number | undefined) => void;
}) {
  const [text, setText] = useState(value === undefined ? '' : String(value));
  useEffect(() => setText(value === undefined ? '' : String(value)), [value]);
  const commit = () => {
    const n = text.trim() === '' ? undefined : Number(text);
    if (n !== undefined && !Number.isFinite(n)) return setText(value === undefined ? '' : String(value));
    if (n === undefined && !optional) return setText(String(value ?? ''));
    if (n !== value) onSet(n);
  };
  return (
    <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
      {label}
      <input
        className={INPUT}
        type="number"
        inputMode="decimal"
        step={step}
        value={text}
        placeholder={optional ? 'not set' : undefined}
        onChange={event => setText(event.target.value)}
        onBlur={commit}
        onKeyDown={event => {
          if (event.key === 'Enter') commit();
          event.stopPropagation(); // the editor's own shortcuts are single keys
        }}
      />
    </label>
  );
}

export function CamerasTab() {
  const { scene, spots, set } = useSpots();
  const { selected, adding, cameras, select, setAdding } = useSpotsUi();
  const spot = selected ? spots[selected] : undefined;
  const unplaced = cameras.filter(camera => !(camera.id in spots));
  const change = (patch: Partial<Spot>) => {
    if (selected && spot) set(selected, { ...spot, ...patch });
  };
  const centre = middle(scene);
  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-3" data-surface="cameras-tab">
      <section className="flex flex-col gap-2">
        <h3 className="font-medium text-foreground text-xs">Cameras on the plan</h3>
        {Object.keys(spots).length === 0 &&
          (cameras.length > 0 ? (
            <p className="text-[11px] text-muted-foreground">
              No camera has a spot yet. Pick a camera below, then click the plan where it is.
            </p>
          ) : (
            <p className="text-[11px] text-muted-foreground">
              No cameras are set up yet. Add one in{' '}
              <a className="underline" href="/devices" target="_top">
                Devices
              </a>
              .
            </p>
          ))}
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(spots).map(([id, s]) => (
            <button
              key={id}
              type="button"
              aria-pressed={id === selected}
              className={`${BUTTON} ${id === selected ? 'bg-primary text-primary-foreground hover:bg-primary' : ''}`}
              onClick={() => select(id === selected ? null : id)}
            >
              {cameraName(cameras, id)}
              {s.yaw === undefined ? ' (no facing)' : ''}
            </button>
          ))}
        </div>
      </section>
      {selected && spot && (
        <section className="flex flex-col gap-2 border-border/60 border-t pt-3">
          <h3 className="font-medium text-foreground text-xs">{cameraName(cameras, selected)}</h3>
          <div className="grid grid-cols-2 gap-2">
            <Field
              label="X, m"
              value={spot.at[0]}
              step={0.05}
              onSet={v => change({ at: [v ?? spot.at[0], spot.at[1]] })}
            />
            <Field
              label="Y, m"
              value={spot.at[1]}
              step={0.05}
              onSet={v => change({ at: [spot.at[0], v ?? spot.at[1]] })}
            />
            <Field label="Height, m" value={spot.height} step={0.05} optional onSet={height => change({ height })} />
            <Field
              label="Facing, degrees (0 is up)"
              value={spot.yaw}
              step={5}
              optional
              onSet={yaw => change({ yaw })}
            />
            <Field label="Field of view, degrees" value={spot.fov} step={5} optional onSet={fov => change({ fov })} />
          </div>
          {spot.yaw === undefined && (
            <p className="text-[11px] text-muted-foreground">
              It places nobody until it has a facing: drag the handle on the plan, or type one (0 is up the page, 90 to
              the right).
            </p>
          )}
          <button
            type="button"
            className={`${BUTTON} w-fit`}
            onClick={() => {
              set(selected, null);
              select(null);
            }}
          >
            Remove this spot
          </button>
        </section>
      )}
      {unplaced.length > 0 && (
        <section className="flex flex-col gap-2 border-border/60 border-t pt-3">
          <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
            Add a camera
            <select className={INPUT} value={adding ?? ''} onChange={event => setAdding(event.target.value || null)}>
              <option value="">Pick one</option>
              {unplaced.map(camera => (
                <option key={camera.id} value={camera.id}>
                  {camera.name}
                </option>
              ))}
            </select>
          </label>
          {adding && (
            <>
              <p className="text-[11px] text-muted-foreground">
                Click the plan where {cameraName(cameras, adding)} is (in the 2D view, with the select tool).
              </p>
              <button
                type="button"
                className={`${BUTTON} w-fit`}
                disabled={!centre}
                onClick={() => {
                  if (!centre) return;
                  set(adding, { at: centre });
                  select(adding);
                  setAdding(null);
                }}
              >
                Put it in the middle
              </button>
            </>
          )}
        </section>
      )}
      <p className="text-[11px] text-muted-foreground">
        People are placed from a saved spot at the next hourly positions run.
      </p>
    </div>
  );
}
