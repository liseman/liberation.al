// Known › Places (#2796): the indoor places the house knows (GET /places, memory/places.py), or the example plans
// while it knows none, and one place's plan, drawn by the vendored floorplan library (src/vendor/floorplan), with where people and pets
// went on it over time (paths/, #2831). A link can open a place at a moment: /places?place=<id>&t=<epoch>. Places is
// places and plans only: what happened where belongs to the journal.
import { Badge, NavLink, Paper, SegmentedControl, Text, useComputedColorScheme } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { lazy, Suspense, useMemo, useState } from 'react';
import type { FloorplanResult, RenderOptions } from '../../vendor/floorplan/index.js';
import { ago, plural } from '../lib/format';
import { PageHeader, PageLoading } from '../lib/page';
import { useRoute } from '../router';
import { type PlaceSummary, placesQuery } from './api';
import { OwnerBar, type Showing } from './edit/OwnerBar';
import { PlanEditor } from './edit/PlanEditor';
import { EXAMPLE_PLACES, type Place } from './examples';
import { PlacePaths } from './paths/PlacePaths';
import classes from './places.module.css';
import { parsePlaceDsl } from './plan/dsl';
import type { Extensions } from './plan/ext';
import { drawPlace, looks, type PlaceDrawing } from './plan/render';

// The 3D view (#2876) is its own chunk: three.js loads only when someone opens it.
const CloudView = lazy(() => import('./cloud/CloudView'));

/** How a plan is drawn on the page: the app's own light or dark, on the card it sits in (--lib-card), with the app's
 *  primary colour as its accent (what the legend's swatches use too). The paths are placed with the same options. */
export const renderOptions = (scheme: 'light' | 'dark'): RenderOptions => ({
  theme: scheme,
  colors: { bg: 'var(--lib-card)', accent: 'var(--mantine-primary-color-filled)' },
});

/** A plan drawn for the page, extensions included. */
export function drawPlan(plan: Record<string, unknown>, scheme: 'light' | 'dark'): PlaceDrawing {
  return drawPlace(plan, renderOptions(scheme));
}

/** A place from GET /places as the tab lists it: its DSL plan read into a document (or why it couldn't be). */
export function fromServer(p: PlaceSummary): Place {
  const out: Place = {
    id: p.id,
    name: p.name,
    lastVisit: p.last_visit,
    lastActivity: p.last_activity ?? null,
    plan: null,
    example: false,
  };
  if (!p.plan) return out;
  out.origin = { source: p.plan_source ?? 'machine', at: p.revision?.at ?? null, machineNewer: p.machine_newer };
  out.planText = p.plan;
  out.machinePlanText = p.machine_plan;
  out.planRevision = p.plan_revision;
  try {
    out.plan = parsePlaceDsl(p.plan);
  } catch (e) {
    out.planError = e instanceof Error ? e.message : String(e);
  }
  return out;
}

/** Whose plan this is, for its heading: a first real plan is a rough machine baseline, so it says so. */
export function originText(place: Place): string | null {
  const o = place.origin;
  if (!o) return null;
  if (o.source === 'owner')
    return o.machineNewer ? 'Your corrected plan (a newer machine plan exists)' : 'Your corrected plan';
  const when = o.at ? `, ${new Date(o.at * 1000).toLocaleDateString([], { month: 'short', day: 'numeric' })}` : '';
  return `Machine-made plan${when}`;
}

/** The place and moment a link asked for (/places?place=&t=), read from the address. */
export function linkedFrom(search: string): { place: string | null; t: number | null } {
  const q = new URLSearchParams(search);
  const t = Number(q.get('t'));
  return { place: q.get('place'), t: q.has('t') && Number.isFinite(t) && t > 0 ? t : null };
}

/** "Revision 2, from 2 visits": which derivation of the place this is, when the plan says. */
export function revisionText(ext: Extensions): string | null {
  const r = ext.revision;
  return r ? `Revision ${r.n}, from ${plural(r.visits.length, 'visit', 'visits')}` : null;
}

/** The legend's entries: one per kind of mark this plan's overlay draws. */
export function legend(ext: Extensions): { key: string; label: string }[] {
  const marks = [...Object.values(ext.provenance), ...Object.values(ext.objects).filter(o => o.source)].map(p =>
    looks({ source: p.source ?? 'machine', confidence: p.confidence }),
  );
  const out: { key: string; label: string }[] = [];
  if (marks.includes('owner')) out.push({ key: 'owner', label: 'You corrected' });
  if (marks.includes('unsure')) out.push({ key: 'unsure', label: 'Machine, unsure' });
  if (Object.values(ext.provenance).some(p => p.partial)) out.push({ key: 'partial', label: 'Partly seen' });
  if (Object.keys(ext.unexplored).length) out.push({ key: 'unexplored', label: 'Not seen yet' });
  if (Object.keys(ext.cameras).length) out.push({ key: 'camera', label: 'Camera' });
  return out;
}

/** "6 rooms · 62.4 m²": what a plan holds, from its schedule (interior area on wall centrelines). */
export function planSummary(result: FloorplanResult): string {
  const { rooms, interiorArea } = result.schedule;
  return `${plural(rooms.length, 'room', 'rooms')} · ${interiorArea.toFixed(1)} m²`;
}

/** When a place was last visited, for its list row. */
/** When the place was last in use: the owner's visit (the rig was there), or, when newer, anyone its fixed cameras saw
 *  (#2905: a place only cameras watch has no visits, and read "No visits yet" with hundreds of tracks on its plan). */
export const lastVisit = (place: Place): string => {
  const visit = place.lastVisit ?? null;
  const activity = place.lastActivity ?? null;
  if (activity != null && (visit == null || activity > visit)) return `Activity ${ago(activity)}`;
  return visit == null ? 'Nothing seen yet' : `Visited ${ago(visit)}`;
};

function drawOrNull(place: Place, scheme: 'light' | 'dark'): PlaceDrawing | null {
  if (!place.plan) return null;
  try {
    return drawPlan(place.plan, scheme);
  } catch {
    return null; // a plan the library can't read; the view says so
  }
}

export function PlacesTab({ examples = EXAMPLE_PLACES }: { examples?: Place[] }) {
  const scheme = useComputedColorScheme('light');
  const server = useQuery(placesQuery());
  // The examples are the empty state: they show only while no place has a plan yet (or the list couldn't load), after
  // any places still waiting for their first.
  const real = server.data?.places ?? [];
  const showingExamples = !real.some(p => p.plan);
  const places = useMemo(
    () => [...real.map(fromServer), ...(showingExamples ? examples : [])],
    [real, examples, showingExamples],
  );
  // A link opens a place at a moment; reading the address on each navigation follows a link while the tab is open.
  const route = useRoute(s => s);
  // biome-ignore lint/correctness/useExhaustiveDependencies: a navigation (route) is what re-reads the address
  const linked = useMemo(() => linkedFrom(location.search), [route]);
  const [picked, setPicked] = useState<string | undefined>();
  const selected = picked ?? linked.place ?? places[0]?.id;
  const place = places.find(p => p.id === selected) ?? places[0];
  const drawn = useMemo(() => new Map(places.map(p => [p.id, drawOrNull(p, scheme)])), [places, scheme]);
  // Correcting a plan, and which of the two plans shows while the owner's holds over a newer machine one.
  const [editing, setEditing] = useState<string | null>(null);
  const [showing, setShowing] = useState<Showing>('mine');
  // The plan, or the place's map in 3D (real places only: the examples have no map).
  const [view, setView] = useState<'plan' | '3d'>('plan');
  const machine = useMemo(() => {
    if (!place?.machinePlanText || showing !== 'machine') return null;
    try {
      return drawPlan(parsePlaceDsl(place.machinePlanText), scheme);
    } catch {
      return null;
    }
  }, [place, showing, scheme]);
  const drawing = place ? drawn.get(place.id) : null;
  const result = drawing?.result;
  return (
    <section aria-labelledby="places-h">
      <PageHeader
        title="Places"
        titleId="places-h"
        lede="Floorplans of the places you've been, and where people and pets went in them."
      />
      {server.isPending ? (
        <PageLoading label="places" />
      ) : (
        <>
          {server.isError && (
            <Text size="sm" c="red" mb="sm">
              Couldn't load the places the house knows. These are example plans.
            </Text>
          )}
          {showingExamples && !server.isError && (
            <Text size="sm" c="dimmed" mb="sm">
              No place has a floorplan yet, so the example plans are shown too. A plan appears once a place has been
              mapped.
            </Text>
          )}
          <div className={classes.layout}>
            <nav aria-label="Places">
              <ul className={classes.list}>
                {places.map(p => {
                  const r = drawn.get(p.id)?.result;
                  return (
                    <li key={p.id}>
                      <NavLink
                        component="button"
                        type="button"
                        active={p.id === place?.id}
                        aria-current={p.id === place?.id ? 'true' : undefined}
                        onClick={() => {
                          setPicked(p.id);
                          setEditing(null);
                          setShowing('mine');
                        }}
                        label={
                          <span className={classes.name}>
                            {p.name}
                            {p.example && (
                              <Badge size="xs" variant="light" tt="none" fw={500}>
                                Example
                              </Badge>
                            )}
                          </span>
                        }
                        description={`${r ? `${planSummary(r)} · ` : p.plan ? '' : 'No plan yet · '}${lastVisit(p)}`}
                      />
                    </li>
                  );
                })}
              </ul>
            </nav>
            {place && (
              <Paper withBorder radius="md" className={classes.plan}>
                <div className={classes.planHead}>
                  <Text fw={600} component="h2" className={classes.planTitle}>
                    {place.name}
                  </Text>
                  {result && (
                    <Text size="sm" c="dimmed">
                      {[originText(place), planSummary(result), drawing && revisionText(drawing.ext)]
                        .filter(Boolean)
                        .join(' · ')}
                    </Text>
                  )}
                  {!place.example && editing !== place.id && (
                    <SegmentedControl
                      size="xs"
                      value={view}
                      onChange={v => setView(v as 'plan' | '3d')}
                      data={[
                        { value: 'plan', label: 'Plan' },
                        { value: '3d', label: '3D' },
                      ]}
                      aria-label="Show the place as"
                    />
                  )}
                </div>
                {!place.example && view === '3d' && editing !== place.id ? (
                  <Suspense
                    fallback={
                      <Text size="sm" c="dimmed">
                        Loading the 3D view…
                      </Text>
                    }
                  >
                    <CloudView key={place.id} place={place.id} />
                  </Suspense>
                ) : (
                  <>
                    {!place.example && place.planText && editing !== place.id && (
                      <OwnerBar
                        key={place.id}
                        place={place}
                        showing={showing}
                        onShow={setShowing}
                        onEdit={() => setEditing(place.id)}
                      />
                    )}
                    {editing === place.id && place.planText ? (
                      <PlanEditor
                        key={place.planText}
                        place={place.id}
                        text={place.planText}
                        frame={place.planRevision ?? null}
                        options={renderOptions(scheme)}
                        onDone={() => setEditing(null)}
                      />
                    ) : showing === 'machine' && machine ? (
                      <>
                        <Text size="sm" c="dimmed" mb="xs">
                          The newest machine plan, to compare. Yours holds until you revert to it.
                        </Text>
                        <div
                          className={classes.drawing}
                          // biome-ignore lint/security/noDangerouslySetInnerHtml: the library's own SVG, which escapes every label
                          dangerouslySetInnerHTML={{ __html: machine.svg }}
                        />
                      </>
                    ) : drawing && result ? (
                      <>
                        <PlacePaths
                          key={`${place.id}-${linked.t ?? ''}`}
                          place={place.id}
                          example={place.example}
                          svg={drawing.svg}
                          result={result}
                          options={renderOptions(scheme)}
                          linked={place.id === (linked.place ?? place.id) ? linked.t : null}
                        />
                        {legend(drawing.ext).length > 0 && (
                          <ul className={classes.legend} aria-label="Key">
                            {legend(drawing.ext).map(l => (
                              <li key={l.key}>
                                <span className={classes.swatch} data-mark={l.key} aria-hidden />
                                {l.label}
                              </li>
                            ))}
                          </ul>
                        )}
                      </>
                    ) : (
                      <>
                        <Text size="sm" c="dimmed">
                          {place.planError
                            ? `This place's plan couldn't be read: ${place.planError}`
                            : place.plan
                              ? "This place's plan couldn't be drawn."
                              : 'No floorplan yet. It appears once the place has been mapped. Until then, paths are drawn on plain floor.'}
                        </Text>
                        {!place.plan && !place.example && (
                          // A lone camera's people are placed on its floor without a plan (#3157, #3169)
                          <PlacePaths
                            key={`${place.id}-${linked.t ?? ''}`}
                            place={place.id}
                            example={false}
                            options={renderOptions(scheme)}
                            linked={place.id === (linked.place ?? place.id) ? linked.t : null}
                          />
                        )}
                      </>
                    )}
                  </>
                )}
              </Paper>
            )}
          </div>
        </>
      )}
    </section>
  );
}
