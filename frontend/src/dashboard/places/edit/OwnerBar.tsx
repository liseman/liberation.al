// Above a real place's plan: whose plan it is and what the owner can do about it. Edit plan opens the
// editor on the plan that holds. Revert plan puts back the previous owner plan in the place's history, written as a
// new record (so the one it replaces stays in the history, and a second Revert puts that back); with no earlier owner
// plan it offers the machine's, when there is one, and is hidden when there's neither, so a place is never left with
// no plan. When a machine plan was made after the owner's, the two can be compared.
import { Badge, Button, Group, Modal, SegmentedControl, Text } from '@mantine/core';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf } from '../../../shared/http';
import { type PlanHistory, planHistoryQuery, restorePlan, savePlan } from '../api';
import type { Place } from '../examples';
import classes from '../places.module.css';

export type Showing = 'mine' | 'machine';

/** What Revert plan goes back to: an earlier owner plan, the machine's plan, or nothing (the button is hidden). */
export type RevertTarget =
  | { kind: 'owner'; record: string; rooms: number; at: number }
  | { kind: 'machine'; rooms: number; at: number | null }
  | null;

/** The rooms in a plan's DSL text: its `room` lines. */
const roomCount = (text: string) => text.split('\n').filter(l => /^\s*room\s/.test(l)).length;

/** Where Revert plan goes for `place`, given its plan history (undefined while it's loading or couldn't be read). */
export function revertTarget(place: Place, history: PlanHistory | undefined): RevertTarget {
  const prev = history?.previous;
  if (prev) return { kind: 'owner', record: prev.id, rooms: prev.rooms, at: prev.at };
  if (place.origin?.source === 'owner' && place.machinePlanText)
    return { kind: 'machine', rooms: roomCount(place.machinePlanText), at: place.origin.at };
  return null;
}

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
const when = (at: number) =>
  new Date(at * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

export function OwnerBar({
  place,
  showing,
  onShow,
  onEdit,
}: {
  place: Place;
  showing: Showing;
  onShow: (s: Showing) => void;
  onEdit: () => void;
}) {
  const client = useQueryClient();
  const [confirm, setConfirm] = useState(false);
  const real = !place.example && !!place.origin && !!place.planText;
  const history = useQuery({ ...planHistoryQuery(place.id), enabled: real });
  const target = revertTarget(place, history.data);
  const revert = useMutation({
    mutationFn: () => (target?.kind === 'owner' ? restorePlan(place.id, target.record) : savePlan(place.id, '')),
    onSuccess: () => {
      setConfirm(false);
      onShow('mine');
      client.invalidateQueries({ queryKey: ['places'] }); // the history too: its key is under 'places'
    },
  });
  const o = place.origin;
  if (!real || !o) return null;
  const mine = o.source === 'owner';
  return (
    <div className={classes.ownerBar}>
      <Group gap="xs">
        {mine && o.machineNewer && (
          <Badge variant="light" color="yellow" tt="none" fw={500} c="var(--lib-warn-text)">
            A newer machine plan exists
          </Badge>
        )}
        {mine && place.machinePlanText && (
          <SegmentedControl
            size="xs"
            value={showing}
            onChange={v => onShow(v as Showing)}
            data={[
              { value: 'mine', label: 'Your plan' },
              { value: 'machine', label: "The machine's" },
            ]}
            aria-label="Which plan to show"
          />
        )}
      </Group>
      <Group gap="xs">
        <Button size="compact-sm" variant="light" onClick={onEdit} disabled={showing === 'machine'}>
          Edit plan
        </Button>
        {target && (
          <Button size="compact-sm" variant="subtle" color="red" onClick={() => setConfirm(true)}>
            {target.kind === 'owner' ? 'Revert plan' : "Revert to the machine's plan"}
          </Button>
        )}
      </Group>
      <Modal
        opened={confirm && !!target}
        onClose={() => setConfirm(false)}
        title={target?.kind === 'machine' ? "Revert to the machine's plan?" : 'Revert to your earlier plan?'}
        centered
      >
        {target && (
          <Text size="sm">
            {target.kind === 'owner'
              ? `This puts back your plan for ${place.name} saved ${when(target.at)}, with ${plural(target.rooms, 'room', 'rooms')}.`
              : `This puts back the machine's plan for ${place.name}${target.at ? ` made ${when(target.at)}` : ''}, with ${plural(target.rooms, 'room', 'rooms')}.`}{' '}
            {mine
              ? 'The plan you have now stays in the history, and Revert plan can bring it back.'
              : "The machine's plan is kept, as it always is."}
          </Text>
        )}
        {revert.isError && (
          <Text size="sm" c="red" mt="xs">
            Couldn't revert it: {messageOf(revert.error)}
          </Text>
        )}
        <Group justify="flex-end" mt="md" gap="xs">
          <Button variant="default" onClick={() => setConfirm(false)}>
            Keep this plan
          </Button>
          <Button color="red" loading={revert.isPending} onClick={() => revert.mutate()}>
            Revert
          </Button>
        </Group>
      </Modal>
    </div>
  );
}
