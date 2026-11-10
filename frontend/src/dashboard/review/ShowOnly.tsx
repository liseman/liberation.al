// Review's "Show only" (R3, #328): the day's people (each with their colour), unknown faces, things, and events, as
// chips. Picking people shows only their speech and presence on the timeline at full strength; picking anything
// marks its time, and playback can skip the rest. Replaces Chips.tsx. Each group lists its 10 most common at first,
// with "+N more" for the rest (#2454): a busy day had about 250 things, and the panel ran thousands of pixels tall.
import { Button, Chip as MantineChip, TextInput, Tooltip, UnstyledButton } from '@mantine/core';
import { type CSSProperties, memo, useId, useMemo, useState } from 'react';
import { EMPTY_SPEAKERS } from '../lib/speakers';
import { useLive } from '../live/hooks';
import { cameraName } from '../people/profile/role';
import { shell } from '../shell';
import { chipColor, chipTitle } from './chipStyle';
import { useEngine, useReview } from './context';
import { cameraChips, personColor, UNKNOWN_COLOR, UNKNOWN_FACES } from './lanes';
import { SkipQuiet, VideoMarkers } from './Player';
import classes from './showOnly.module.css';
import { facetedSpans, nonCooccurring } from './spans';
import type { Chip } from './types';

/** The groups, in order, and which chips each holds. */
const GROUPS: { name: string; has: (c: Chip) => boolean }[] = [
  { name: 'People', has: c => c.kind === 'person' },
  { name: 'Things', has: c => c.kind === 'object' },
  { name: 'Events', has: c => c.kind === 'event' },
];

/** A label as people read it: "sewing_machine" is "sewing machine" (#2454 follow-up). */
const spaced = (label: string): string => label.replace(/_+/g, ' ');

/** How many chips a group lists before "+N more". */
export const GROUP_CAP = 10;

/** The chips a group lists: all of them when open or searching, else its GROUP_CAP most common plus anything picked,
 *  in the group's own order. */
export function capped(chips: readonly Chip[], picked: ReadonlySet<string>, all: boolean): Chip[] {
  if (all || chips.length <= GROUP_CAP) return [...chips];
  const top = new Set([...chips].sort((a, b) => b.count - a.count).slice(0, GROUP_CAP));
  return chips.filter(c => top.has(c) || picked.has(c.id));
}

/** video: also hold the video's markers and Skip quiet time (Review's page, whose control bar leaves them out). */
// Memoized (#2522): Review re-renders on every query and status change while it plays, and redrawing a few hundred
// chips with their tooltips each time was Review's largest render cost. Its props (the engine's chip list) only
// change with the day's data; the selection, colours, and cameras it reads from the engine's store.
export const ShowOnly = memo(function ShowOnly({ chips, video = false }: { chips: Chip[]; video?: boolean }) {
  const engine = useEngine();
  const sel = useReview(s => s.sel);
  const colors = useReview(s => s.colors);
  const cameras = useReview(s => s.cameras);
  const filterStatus = useReview(s => s.filterStatus);
  const speakers = useLive('speakers') ?? EMPTY_SPEAKERS;
  const [q, setQ] = useState('');
  const [open, setOpen] = useState<ReadonlySet<string>>(() => new Set()); // the groups showing all their chips
  const uid = useId(); // group label ids: unique per Show only, and free of the spaces in the names
  const query = q.trim().toLowerCase();
  const any = sel.size > 0 || cameras.length > 0;

  const color = (c: Chip) =>
    c.id === UNKNOWN_FACES
      ? UNKNOWN_COLOR
      : c.kind === 'person'
        ? personColor(c.label, speakers)
        : (colors[c.id] ?? chipColor(c.id));
  const availability = useMemo(() => {
    const dimmed = nonCooccurring(cameraChips(chips, cameras), sel, chips);
    const cameraChoices = new Set(
      engine.cams.filter(
        cam =>
          !cameras.includes(cam) && sel.size > 0 && facetedSpans(cameraChips(chips, [cam]), sel, chips)?.length === 0,
      ),
    );
    return { dimmed, cameraChoices };
  }, [chips, sel, cameras, engine.cams]);
  const views = shell().views;
  const descriptions = useMemo(() => {
    const selected = GROUPS.map(group => chips.filter(c => sel.has(c.id) && group.has(c)));
    const camera = selected
      .map(group => group.map(c => c.label).join(' or '))
      .filter(Boolean)
      .join(' and ');
    const cameraNames = cameras.map(cam => cameraName(cam, views[cam])).join(' or ');
    const byKind = new Map<string, string>();
    for (const kind of new Set(chips.map(c => c.kind))) {
      const labels = selected
        .map(group =>
          group
            .filter(c => c.kind !== kind)
            .map(c => c.label)
            .join(' or '),
        )
        .filter(Boolean);
      if (cameraNames) labels.push(cameraNames);
      byKind.set(kind, labels.join(' and '));
    }
    return { camera, byKind };
  }, [chips, sel, cameras, views]);
  const tip = (c: Chip) =>
    availability.dimmed.has(c.id)
      ? `${chipTitle(c)}. Never at the same time as ${descriptions.byKind.get(c.kind)} in the loaded data. You can still choose it.`
      : chipTitle(c);
  const groups = GROUPS.map(g => {
    const all = chips.filter(c => g.has(c) && (!query || c.label.toLowerCase().includes(query)));
    const shown = capped(all, sel, !!query || open.has(g.name));
    return { name: g.name, chips: shown, more: all.length - shown.length, long: all.length > GROUP_CAP };
  }).filter(g => g.chips.length);
  const toggleGroup = (name: string) =>
    setOpen(prev => {
      const next = new Set(prev);
      if (!next.delete(name)) next.add(name);
      return next;
    });
  const moreButton = (g: (typeof groups)[number]) =>
    !query &&
    g.long && (
      <Button
        variant="subtle"
        size="compact-sm"
        className={classes.more}
        aria-expanded={g.more === 0}
        onClick={() => toggleGroup(g.name)}
      >
        {g.more > 0 ? `+${g.more} more` : 'Show fewer'}
      </Button>
    );

  return (
    <fieldset className={classes.showOnly}>
      <legend className={classes.legend}>Show only</legend>
      <div className={classes.filters}>
        {/* biome-ignore lint/a11y/useSemanticElements: camera toggles grouped inside the Show only fieldset */}
        <div className={classes.group} role="group" aria-labelledby={`${uid}-cameras`}>
          <h3 className={classes.groupName} id={`${uid}-cameras`}>
            Cameras
          </h3>
          <MantineChip.Group multiple value={cameras} onChange={value => engine.pickCameras(value)}>
            <div className={classes.groupChips}>
              {engine.cams.map(cam => (
                <Tooltip
                  key={cam}
                  label={
                    availability.cameraChoices.has(cam)
                      ? `Never at the same time as ${descriptions.camera} in the loaded data. You can still choose it.`
                      : cameraName(cam, shell().views[cam])
                  }
                  events={{ hover: true, focus: true, touch: false }}
                >
                  <span
                    className={classes.cameraChoice}
                    data-unrelated={availability.cameraChoices.has(cam) || undefined}
                  >
                    <MantineChip value={cam} size="sm" classNames={{ label: classes.cameraChip }}>
                      {cameraName(cam, shell().views[cam])}
                    </MantineChip>
                  </span>
                </Tooltip>
              ))}
            </div>
          </MantineChip.Group>
        </div>
        {chips.length > 12 && (
          <TextInput
            className={classes.search}
            size="sm"
            label="Find a filter"
            type="search"
            value={q}
            onChange={e => setQ(e.currentTarget.value)}
            placeholder="Find"
            aria-label="Find people, things, and events"
          />
        )}
      </div>
      <div className={classes.list}>
        {chips.length === 0 && (
          <span className={classes.empty}>No people, things, or events found for this day yet.</span>
        )}
        {chips.length > 0 && groups.length === 0 && (
          <span className={classes.empty}>{`Nothing matches “${q.trim()}”.`}</span>
        )}
        {groups.map((g, i) => (
          // biome-ignore lint/a11y/useSemanticElements: a group of toggle buttons inside the Show only fieldset
          <div key={g.name} className={classes.group} role="group" aria-labelledby={`${uid}-g${i}`}>
            <h3 className={classes.groupName} id={`${uid}-g${i}`}>
              {g.name}
            </h3>
            {g.name === 'People' ? (
              // biome-ignore lint/a11y/noNoninteractiveTabindex: scroll long filter lists with the keyboard
              <div className={classes.groupChips} tabIndex={0}>
                {g.chips.map(c => (
                  <Tooltip key={c.id} label={tip(c)} events={{ hover: true, focus: true, touch: false }}>
                    <UnstyledButton
                      type="button"
                      className={classes.chip}
                      data-review-keys
                      data-unrelated={availability.dimmed.has(c.id) || undefined}
                      aria-pressed={sel.has(c.id)}
                      style={{ '--so-color': color(c) } as CSSProperties}
                      onClick={() => engine.toggleChip(c.id)}
                    >
                      <i className={classes.dot} />
                      {spaced(c.label)}
                      <span className={classes.count}>{c.count}</span>
                    </UnstyledButton>
                  </Tooltip>
                ))}
                {moreButton(g)}
              </div>
            ) : (
              // Things and events are rounded pills, like the cameras, and wrap instead of scrolling (#2287).
              <div className={classes.pills} data-review-keys>
                {g.chips.map(c => (
                  <Tooltip key={c.id} label={tip(c)} events={{ hover: true, focus: true, touch: false }}>
                    <span className={classes.cameraChoice}>
                      <MantineChip
                        size="sm"
                        radius="xl"
                        checked={sel.has(c.id)}
                        onChange={() => engine.toggleChip(c.id)}
                        data-unrelated={availability.dimmed.has(c.id) || undefined}
                        classNames={{ label: classes.cameraChip }}
                        style={{ '--so-color': color(c) } as CSSProperties}
                      >
                        <i className={classes.dot} />
                        {spaced(c.label)}
                        <span className={classes.count}>{c.count}</span>
                      </MantineChip>
                    </span>
                  </Tooltip>
                ))}
                {moreButton(g)}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className={classes.controls}>
        <span className={classes.selection} role="status">
          {any
            ? `${sel.size} moment ${sel.size === 1 ? 'filter' : 'filters'} · ${cameras.length ? `${cameras.length} of ${engine.cams.length} cameras` : 'all cameras'}`
            : 'All moments · all cameras'}
          {filterStatus === 'empty' && ' · Nothing overlaps in the loaded moments.'}
          {filterStatus === 'unknown' && ' · Timing unavailable for these filters; Skip stays neutral.'}
        </span>
        {video && (
          <>
            <SkipQuiet />
            <span className={classes.markers}>
              <span className={classes.markersName}>Boxes on the video</span>
              <VideoMarkers />
            </span>
          </>
        )}
        <Button
          variant="default"
          size="compact-sm"
          className={classes.showAll}
          onClick={() => {
            engine.clearChips();
            engine.pickCameras([]);
            engine.pickPerson('');
          }}
          disabled={!any}
        >
          Show all
        </Button>
      </div>
    </fieldset>
  );
});
