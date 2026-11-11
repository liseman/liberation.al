// Live's "Learn surroundings" switch (#255): whether pictures are collected from the cameras (for the 3D map, which
// is being rebuilt with a new method), and which
// cameras are sampled. GET/POST /surroundings (dashboard/surroundings.py). The settings are followed live, so a
// change here restarts nothing. Any viewer sees the switch; only the machine Liberational runs on can change it.
//
// The switch and each camera in its menu save at once: the change shows while the request is out (with a spinner on
// what is saving), and a failure puts the saved state back and says why in a toast. One save of each kind is out at a
// time, so a rollback never undoes a later change.
import { Anchor, Button, Loader, Menu, Switch, Text } from '@mantine/core';
import { IconCamera } from '@tabler/icons-react';
import { type QueryClient, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf, request } from '../../shared/http';
import { Toast } from '../lib/Toast';
import { viewName } from '../review/views';
import { linkClick } from '../router';

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

/** How many cameras are sampled, for the menu button and the on-state hint. */
export function sampledLabel(view: SurroundingsView): string {
  const sampled = view.cameras.filter(c => c.sampled);
  return sampled.length === 1 ? viewName(sampled[0]?.name ?? '') : `${sampled.length} cameras`;
}

/** The dimmed line beside the switch while this browser may change it. */
export function surroundingsHint(view: SurroundingsView): { off: string; on: string } {
  const from = sampledLabel(view);
  return {
    off: "Collects the cameras' pictures, and keeps a point cloud of what each one sees",
    on: `Collecting pictures from ${from}. See them in Places`,
  };
}

/** The cameras sampled after `name` is switched: taking the last one away leaves none chosen, which is the default. */
export function toggled(view: SurroundingsView, name: string): string[] {
  const sampled = view.cameras.filter(c => c.sampled).map(c => c.name);
  return sampled.includes(name) ? sampled.filter(c => c !== name) : [...sampled, name];
}

/** The view as the server will answer `change`, to show while it is saving. */
export function applied(view: SurroundingsView, change: SurroundingsChange): SurroundingsView {
  const chosen = change.cameras ?? view.chosen;
  return {
    ...view,
    on: change.on ?? view.on,
    chosen,
    cameras: view.cameras.map(c => ({ ...c, sampled: chosen.length ? chosen.includes(c.name) : c.default })),
  };
}

/** A save that shows its change at once and puts the saved view back, with a toast, if it fails. */
function useSave(client: QueryClient, onFail: (message: string) => void) {
  return useMutation({
    mutationFn: (body: SurroundingsChange) => request<SurroundingsView>('/surroundings', body),
    onMutate: async (body: SurroundingsChange) => {
      await client.cancelQueries({ queryKey: SURROUNDINGS_KEY });
      const before = client.getQueryData<SurroundingsView>(SURROUNDINGS_KEY);
      if (before) client.setQueryData(SURROUNDINGS_KEY, applied(before, body));
      return { before };
    },
    onError: (error, _body, context) => {
      if (context?.before) client.setQueryData(SURROUNDINGS_KEY, context.before);
      onFail(`Couldn't change it: ${messageOf(error)}`);
    },
    onSuccess: next => client.setQueryData(SURROUNDINGS_KEY, next),
  });
}

export function LearnSurroundings() {
  const client = useQueryClient();
  const view = useQuery({
    queryKey: SURROUNDINGS_KEY,
    queryFn: ({ signal }) => request<SurroundingsView>('/surroundings', undefined, signal, 'background'),
    refetchInterval: REFRESH_MS,
  }).data;
  const [fail, setFail] = useState<string | null>(null);
  const power = useSave(client, setFail);
  const cameras = useSave(client, setFail);
  const [toggling, setToggling] = useState<string | null>(null);
  if (!view?.cameras.length) return null;
  const from = sampledLabel(view);
  const hints = surroundingsHint(view);
  const saving = cameras.isPending ? toggling : null;
  return (
    <div className="surroundings">
      <div className="surroundings-row">
        <Switch
          size="sm"
          label="Learn surroundings"
          checked={view.on}
          disabled={!view.control}
          aria-busy={power.isPending}
          thumbIcon={power.isPending ? <Loader size={8} color="gray" aria-hidden /> : undefined}
          onChange={e => {
            if (!power.isPending) power.mutate({ on: e.currentTarget.checked });
          }}
        />
        <Menu position="bottom-end" withinPortal closeOnItemClick={false} disabled={!view.control}>
          <Menu.Target>
            <Button
              variant="subtle"
              color="gray"
              size="compact-sm"
              disabled={!view.control}
              leftSection={<IconCamera size={16} aria-hidden />}
              aria-label="Cameras to learn from"
            >
              {from}
            </Button>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Label>Cameras to learn from</Menu.Label>
            {view.cameras.map(c => {
              const busy = saving === c.name;
              return (
                <Menu.CheckboxItem
                  key={c.name}
                  checked={c.sampled}
                  disabled={cameras.isPending}
                  aria-busy={busy}
                  rightSection={
                    busy ? (
                      <Loader size={12} aria-hidden />
                    ) : c.default ? (
                      <Text size="xs" c="dimmed">
                        Used when none is chosen
                      </Text>
                    ) : undefined
                  }
                  onChange={() => {
                    setToggling(c.name);
                    cameras.mutate({ cameras: toggled(view, c.name) });
                  }}
                >
                  {viewName(c.name)}
                </Menu.CheckboxItem>
              );
            })}
            <Text size="xs" c="dimmed" px="sm" py={4} maw={240}>
              With none chosen, the default camera is used.
            </Text>
          </Menu.Dropdown>
        </Menu>
      </div>
      {view.control ? (
        <Text size="xs" c="dimmed" className="surroundings-hint">
          {view.on ? (
            <>
              {`Collecting pictures from ${from}. See them in `}
              <Anchor href="/places" size="xs" onClick={linkClick}>
                Places
              </Anchor>
            </>
          ) : (
            hints.off
          )}
        </Text>
      ) : (
        <Text size="xs" c="dimmed">
          {ELSEWHERE}
        </Text>
      )}
      <Toast text={fail} error label="Learn surroundings" onClose={() => setFail(null)} />
    </div>
  );
}
