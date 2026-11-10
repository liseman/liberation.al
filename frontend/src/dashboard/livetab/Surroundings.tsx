// Live's "Learn surroundings" switch (#255): whether the cameras' pictures may refine a place's map, and which
// cameras are sampled. GET/POST /surroundings (dashboard/surroundings.py). The settings are followed live, so a
// change here restarts nothing. Any viewer sees the switch; only the machine Liberational runs on can change it.
import { Button, Checkbox, Popover, Stack, Text } from '@mantine/core';
import { IconChevronDown } from '@tabler/icons-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { messageOf, request } from '../../shared/http';
import { viewName } from '../review/views';

export interface SurroundingsCamera {
  name: string;
  sampled: boolean; // learning takes pictures from it while on
  default: boolean; // the one used when none is chosen
}

export interface SurroundingsView {
  control: boolean; // this browser may change it (the dashboard's machine)
  on: boolean;
  cameras: SurroundingsCamera[];
  chosen: string[]; // the choice as saved; empty means the default camera
}

export interface SurroundingsChange {
  on?: boolean;
  cameras?: string[];
}

export const SURROUNDINGS_KEY = ['surroundings'] as const;
const REFRESH_MS = 15_000; // another browser, or Settings, may change it

const ELSEWHERE = 'Only the machine Liberational runs on can change this.';

/** The cameras sampled after `name` is switched: taking the last one away leaves none chosen, which is the default. */
export function toggled(view: SurroundingsView, name: string): string[] {
  const sampled = view.cameras.filter(c => c.sampled).map(c => c.name);
  return sampled.includes(name) ? sampled.filter(c => c !== name) : [...sampled, name];
}

export function LearnSurroundings() {
  const client = useQueryClient();
  const view = useQuery({
    queryKey: SURROUNDINGS_KEY,
    queryFn: ({ signal }) => request<SurroundingsView>('/surroundings', undefined, signal, 'background'),
    refetchInterval: REFRESH_MS,
  }).data;
  const change = useMutation({
    mutationFn: (body: SurroundingsChange) => request<SurroundingsView>('/surroundings', body),
    onSuccess: next => client.setQueryData(SURROUNDINGS_KEY, next),
  });
  if (!view?.cameras.length) return null;
  const sampled = view.cameras.filter(c => c.sampled);
  const from = sampled.length === 1 ? viewName(sampled[0]?.name ?? '') : `${sampled.length} cameras`;
  const title = !view.control
    ? ELSEWHERE
    : view.on
      ? `Sampling pictures from ${from} to improve the map of this place. Click to stop.`
      : 'Sample pictures from the cameras and keep the ones that improve the map of this place.';
  return (
    <div className="surroundings">
      <Button.Group>
        <Button
          size="xs"
          variant={view.on ? 'filled' : 'default'}
          aria-pressed={view.on}
          disabled={!view.control}
          loading={change.isPending}
          title={title}
          onClick={() => change.mutate({ on: !view.on })}
        >
          Learn surroundings
        </Button>
        <Popover position="bottom-end" withinPortal width={260}>
          <Popover.Target>
            <Button
              size="xs"
              variant={view.on ? 'filled' : 'default'}
              px={6}
              aria-label={`Cameras to learn surroundings from: ${from}`}
            >
              <IconChevronDown size={14} aria-hidden />
            </Button>
          </Popover.Target>
          <Popover.Dropdown>
            <Stack gap="xs" role="group" aria-label="Cameras to learn surroundings from">
              {view.cameras.map(c => (
                <Checkbox
                  key={c.name}
                  size="xs"
                  label={c.default ? `${viewName(c.name)} (default)` : viewName(c.name)}
                  checked={c.sampled}
                  disabled={!view.control}
                  onChange={() => change.mutate({ cameras: toggled(view, c.name) })}
                />
              ))}
              <Text size="xs" c="dimmed">
                {view.control ? 'With none chosen, the default camera is used.' : ELSEWHERE}
              </Text>
            </Stack>
          </Popover.Dropdown>
        </Popover>
      </Button.Group>
      {change.isError && (
        <Text role="alert" size="xs" c="var(--lib-bad)">
          {`Couldn't change it: ${messageOf(change.error)}`}
        </Text>
      )}
    </div>
  );
}
