// Known › Places: the indoor places the house knows (GET /places, memory/places.py), or the example plans
// while it knows none, and one place's plan, drawn by the vendored floorplan library (src/vendor/floorplan), with where people and pets
// went on it over time (paths/). A link can open a place at a moment: /places?place=<id>&t=<epoch>. Places is
// places and plans only: what happened where belongs to the journal.
import { Badge, Button, Group, NavLink, Paper, SegmentedControl, Text, useComputedColorScheme } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { lazy, Suspense, useMemo, useState } from 'react';
import type { Scene } from '../../plan-editor/protocol';
import type { FloorplanResult, RenderOptions } from '../../vendor/floorplan/index.js';
import { confirm } from '../lib/confirm';
import { ago, plural } from '../lib/format';
import { BlockSkeleton } from '../lib/loading';
import { PageHeader, PageLoading } from '../lib/page';
import { useRoute } from '../router';
import { type PlaceSummary, placesQuery } from './api';
import { OwnerBar, type Showing } from './edit/OwnerBar';
import { EXAMPLE_PLACES, type Place } from './examples';
import { LearningBadge, LearningPanel, useLearning } from './learning';
import { PlacePaths } from './paths/PlacePaths';
import classes from './places.module.css';
import { parsePlaceDsl } from './plan/dsl';
import type { Extensions } from './plan/ext';
import { drawPlace, looks, type PlaceDrawing } from './plan/render';
import { drawScene, type SceneDrawing } from './scene/draw';
import { PlaceEditor } from './scene/PlaceEditor';

// The 3D view is its own chunk: the viewer and its engine load only when someone opens it.
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
    // a plan stored as a Pascal scene is JSON; one still in the old format is the floorplan library's DSL
    if (p.plan.trimStart().startsWith('{')) out.scene = sceneOf(p.plan);
    else out.plan = parsePlaceDsl(p.plan);
  } catch (e) {
    out.planError = e instanceof Error ? e.message : String(e);
  }
  return out;
}

/** How a place shows, for looking: its plan with paths, or the place's map in 3D. */
export type View = 'plan' | 'map';

/** A list row's "6 rooms · 62.4 m² · ", from whichever way the plan was drawn, or null with no drawing. */
function summaryOf(result: FloorplanResult | undefined, scened: SceneDrawing | null | undefined): string | null {
  if (result) return `${planSummary(result)} · `;
  return scened ? `${sceneSummary(scened)} · ` : null;
}

/** A stored plan's text read as a Pascal scene. Throws when it isn't one. */
export function sceneOf(text: string): Scene {
  const scene = JSON.parse(text) as Partial<Scene> | null;
  if (!scene || typeof scene.nodes !== 'object' || scene.nodes === null || !Array.isArray(scene.rootNodeIds))
    throw new Error('the plan is not a scene');
  return scene as Scene;
}

/** "6 rooms · 62.4 m²" for a plan drawn from a scene (floor area on wall centrelines). */
export const sceneSummary = (drawing: SceneDrawing): string =>
  `${plural(drawing.rooms, 'room', 'rooms')} · ${drawing.area.toFixed(1)} m²`;

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
 *  (a place only cameras watch has no visits, and read "No visits yet" with hundreds of tracks on its plan). */
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
  // What "Learn surroundings" is doing now: a badge on each place, and a panel under the list (learning.tsx).
  const learning = useLearning();
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
  // Which of the two plans shows while the owner's holds over a newer machine one.
  const [showing, setShowing] = useState<Showing>('mine');
  // How the place shows, for looking: its plan with paths, or its map in 3D (real places only: the examples have no
  // map). Neither can change the plan.
  const [view, setView] = useState<View>('plan');
  // The plan editor (scene/PlaceEditor.tsx), opened by Edit plan: the one way into changing a plan. `unsaved` is
  // whether it holds changes that leaving it would drop.
  const [editing, setEditing] = useState(false);
  const [unsaved, setUnsaved] = useState(false);
  /** Leave the editor (to look, or for another place), asking first when that drops unsaved changes. */
  const leave = async (then: () => void) => {
    if (
      editing &&
      unsaved &&
      !(await confirm({
        title: 'Discard your unsaved changes?',
        body: "The changes to this plan that you haven't saved will be lost.",
        confirm: 'Discard changes',
        danger: true,
        id: 'plan-leave',
      }))
    )
      return;
    setEditing(false);
    setUnsaved(false);
    then();
  };
  // A plan stored as a Pascal scene is drawn from it (scene/draw.ts); one in the old format by the floorplan library.
  const sceneDrawn = useMemo(
    () => new Map(places.map(p => [p.id, p.scene && !p.plan ? drawScene(p.scene, p.name) : null])),
    [places],
  );
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
  const scened = place ? (sceneDrawn.get(place.id) ?? null) : null;
  // The editor opens on a place's own plan, in either format, and on the example that has a scene.
  const editable = !!place && (place.example ? !!place.scene : !!(place.plan || place.scene));
  const views: { value: View; label: string }[] =
    place && !place.example
      ? [
          { value: 'plan', label: 'Plan' },
          { value: 'map', label: '3D map' },
        ]
      : [];
  const shown: View = views.some(v => v.value === view) ? view : 'plan';
  const facts = place
    ? [
        originText(place),
        result ? planSummary(result) : scened ? sceneSummary(scened) : null,
        drawing ? revisionText(drawing.ext) : null,
      ].filter(Boolean)
    : [];
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
            <div className={classes.side}>
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
                          onClick={() =>
                            leave(() => {
                              setPicked(p.id);
                              setView('plan');
                              setShowing('mine');
                            })
                          }
                          label={
                            <span className={classes.name}>
                              {p.name}
                              {p.example && (
                                <Badge size="xs" variant="light" tt="none" fw={500}>
                                  Example
                                </Badge>
                              )}
                              {!p.example && <LearningBadge status={learning} place={p.id} />}
                            </span>
                          }
                          description={`${summaryOf(r, sceneDrawn.get(p.id)) ?? (p.plan || p.scene ? '' : 'No plan yet · ')}${lastVisit(p)}`}
                        />
                      </li>
                    );
                  })}
                </ul>
              </nav>
              <LearningPanel status={learning} />
            </div>
            {place && (
              <Paper withBorder radius="md" className={classes.plan}>
                <div className={classes.planHead}>
                  <div className={classes.planNamed}>
                    <Text fw={600} component="h2" className={classes.planTitle}>
                      <span className={classes.name}>
                        {place.name}
                        {!place.example && <LearningBadge status={learning} place={place.id} />}
                      </span>
                    </Text>
                    {facts.length > 0 && (
                      <Text size="sm" c="dimmed">
                        {facts.join(' · ')}
                      </Text>
                    )}
                  </div>
                  <Group gap="xs" wrap="wrap">
                    {views.length > 1 && (
                      <SegmentedControl
                        size="xs"
                        value={editing ? '' : shown}
                        onChange={v => leave(() => setView(v as View))}
                        data={views}
                        aria-label="Show the place as"
                      />
                    )}
                    {editable &&
                      (editing ? (
                        <Button size="compact-sm" variant="default" onClick={() => leave(() => {})}>
                          Close editor
                        </Button>
                      ) : (
                        <Button
                          size="compact-sm"
                          variant="light"
                          disabled={showing === 'machine'}
                          onClick={() => setEditing(true)}
                        >
                          Edit plan
                        </Button>
                      ))}
                  </Group>
                </div>
                {editable && editing ? (
                  <PlaceEditor
                    key={place.id}
                    place={place}
                    view="2d"
                    scheme={scheme}
                    onUnsaved={setUnsaved}
                    onSaved={() => {
                      setEditing(false);
                      setUnsaved(false);
                      setView('plan');
                    }}
                  />
                ) : !place.example && shown === 'map' ? (
                  <Suspense fallback={<BlockSkeleton label="the 3D view" />}>
                    <CloudView key={place.id} place={place.id} />
                  </Suspense>
                ) : (
                  <>
                    {!place.example && place.planText && (
                      <OwnerBar key={place.id} place={place} showing={showing} onShow={setShowing} />
                    )}
                    {showing === 'machine' && machine ? (
                      <>
                        <Text size="sm" c="dimmed" mb="xs">
                          The newest machine plan, to compare. Yours holds unless you take this one.
                        </Text>
                        <div
                          className={classes.drawing}
                          // biome-ignore lint/security/noDangerouslySetInnerHtml: the library's own SVG, which escapes every label
                          dangerouslySetInnerHTML={{ __html: machine.svg }}
                        />
                      </>
                    ) : scened ? (
                      <>
                        <PlacePaths
                          key={`${place.id}-${linked.t ?? ''}`}
                          place={place.id}
                          example={place.example}
                          svg={scened.svg}
                          drawn={scened.surface}
                          options={renderOptions(scheme)}
                          linked={place.id === (linked.place ?? place.id) ? linked.t : null}
                        />
                        {(Object.keys(scened.cameras).length > 0 || scened.unexplored) && (
                          <ul className={classes.legend} aria-label="Key">
                            {scened.unexplored && (
                              <li>
                                <span className={classes.swatch} data-mark="unexplored" aria-hidden />
                                Not seen yet
                              </li>
                            )}
                            {Object.keys(scened.cameras).length > 0 && (
                              <li>
                                <span className={classes.swatch} data-mark="camera" aria-hidden />
                                Camera
                              </li>
                            )}
                          </ul>
                        )}
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
                            : place.plan || place.scene
                              ? "This place's plan couldn't be drawn."
                              : 'No floorplan yet. It appears once the place has been mapped. Until then, paths are drawn on plain floor.'}
                        </Text>
                        {!place.plan && !place.scene && !place.example && (
                          // A lone camera's people are placed on its floor without a plan
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
