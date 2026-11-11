// The 3D view's per-camera boxes (#409): one a camera, with how many points (or splats) it gave, that shows or hides
// what came from it, and under each its kept views by when they were taken, so an earlier pass can be switched off
// on its own. All and None set every box at once. A map made before maps said where their points came from shows the
// cameras' boxes unavailable, with why.
import { Button, Checkbox, Group, Stack, Text } from '@mantine/core';
import { Disclosure } from '../../lib/Disclosure';
import { momentWords, plural } from '../../lib/format';
import type { Sources } from '../api';
import classes from '../places.module.css';
import { cameraState, countText, toggled } from './sources';

export function SourceFilter({
  sources,
  cameras,
  hidden,
  onChange,
  noun,
  splatUntold,
}: {
  /** Where the map's points came from; null for a map that doesn't say. */
  sources: Sources | null | undefined;
  /** The cameras registered in the map, for the unavailable boxes of a map that doesn't say. */
  cameras: readonly string[];
  hidden: ReadonlySet<string>;
  onChange: (hidden: Set<string>) => void;
  /** What's counted: the points (Points) or the splat's Gaussians (Splat). */
  noun: 'points' | 'splats';
  /** The splat drawn doesn't say which view each of its Gaussians came from: the boxes change only the points. */
  splatUntold: boolean;
}) {
  if (!sources?.views.length) {
    if (!cameras.length) return null;
    return (
      <Stack gap={4} mt="xs" role="group" aria-label="Show what each camera added">
        <Group gap="md">
          {cameras.map(name => (
            <Checkbox key={name} size="xs" label={name} checked disabled readOnly />
          ))}
        </Group>
        <Text size="xs" c="dimmed">
          Showing each camera's part isn't available for this map yet. The map gains it the next time it's rebuilt.
        </Text>
      </Stack>
    );
  }
  const { views } = sources;
  const all = views.map(v => v.id);
  const count = (points: number, gaussians: number | null) =>
    countText(noun === 'points' ? points : (gaussians ?? null), noun);
  return (
    <Stack gap={4} mt="xs" role="group" aria-label="Show what each camera added">
      <Group gap="xs" wrap="wrap">
        <Text size="xs" c="dimmed">
          Show what each camera added
        </Text>
        <Button size="compact-xs" variant="subtle" onClick={() => onChange(new Set())} disabled={hidden.size === 0}>
          All
        </Button>
        <Button
          size="compact-xs"
          variant="subtle"
          onClick={() => onChange(new Set(all))}
          disabled={all.every(id => hidden.has(id))}
        >
          None
        </Button>
      </Group>
      <div className={classes.sourceList}>
        {sources.cameras.map(camera => {
          const own = camera.views.flatMap(i => (views[i] ? [views[i]] : []));
          const state = cameraState(own, hidden);
          const counted = count(camera.points, camera.gaussians);
          return (
            <div key={camera.name} className={classes.sourceCamera}>
              <Checkbox
                size="xs"
                label={counted ? `${camera.name} · ${counted}` : camera.name}
                checked={state === 'on'}
                indeterminate={state === 'mixed'}
                onChange={() =>
                  onChange(
                    toggled(
                      hidden,
                      own.map(v => v.id),
                      state !== 'on',
                    ),
                  )
                }
              />
              {own.length > 1 && (
                <Disclosure
                  lazy
                  summary={
                    <Text size="xs" c="dimmed" span>
                      {plural(own.length, 'view', 'views')}
                    </Text>
                  }
                  className={classes.sourceViews}
                >
                  <Stack gap={4} pl="lg" pt={4}>
                    {own.map(v => {
                      const when = v.t != null ? momentWords(v.t) : 'An earlier view';
                      const n = count(v.points, v.gaussians);
                      return (
                        <Checkbox
                          key={v.id}
                          size="xs"
                          label={n ? `${when} · ${n}` : when}
                          checked={!hidden.has(v.id)}
                          onChange={e => onChange(toggled(hidden, [v.id], e.currentTarget.checked))}
                        />
                      );
                    })}
                  </Stack>
                </Disclosure>
              )}
            </div>
          );
        })}
      </div>
      {splatUntold && (
        <Text size="xs" c="dimmed">
          This splat doesn't say which camera each part of it came from, so the boxes change only the Points view.
        </Text>
      )}
    </Stack>
  );
}
