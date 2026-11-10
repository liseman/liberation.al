// Above a real place's plan, when the machine has a plan of its own beside the owner's: the two can be compared
// (the page's Edit plan button is off while the machine's shows: you edit your own), and while the machine's
// is showing it can be taken instead: that withdraws the owner's plan, as one more version in the plan's history
// (scene/Versions.tsx), from which any earlier version can be restored.
import { Badge, Button, Group, SegmentedControl } from '@mantine/core';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { messageOf } from '../../shared/http';
import { confirm } from '../lib/confirm';
import { StatusLine } from '../lib/message';
import { savePlan } from './api';
import type { Place } from './examples';
import classes from './places.module.css';

export type Showing = 'mine' | 'machine';

export function OwnerBar({ place, showing, onShow }: { place: Place; showing: Showing; onShow: (s: Showing) => void }) {
  const client = useQueryClient();
  const withdraw = useMutation({
    mutationFn: () => savePlan(place.id, ''),
    onSuccess: () => {
      onShow('mine');
      client.invalidateQueries({ queryKey: ['places'] }); // the history too: its key is under 'places'
    },
  });
  const o = place.origin;
  if (place.example || !o || !place.scene) return null;
  const mine = o.source === 'owner';
  // nothing to compare, nothing to say: the bar is only for a plan the machine has its own version of
  if (!mine || (!o.machineNewer && !place.machineScene)) return null;
  return (
    <div className={classes.ownerBar}>
      <Group gap="xs">
        {mine && o.machineNewer && (
          <Badge variant="light" color="yellow" tt="none" fw={500} c="var(--lib-warn-text)">
            A newer machine plan exists
          </Badge>
        )}
        {mine && place.machineScene && (
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
        {mine && showing === 'machine' && (
          <Button
            size="compact-sm"
            variant="subtle"
            loading={withdraw.isPending}
            onClick={async () => {
              const yes = await confirm({
                title: "Use the machine's plan?",
                body: `Your plan for ${place.name} stays in its versions, and you can restore it from there.`,
                confirm: "Use the machine's plan",
                id: 'plan-withdraw',
              });
              if (yes) withdraw.mutate();
            }}
          >
            Use the machine's plan
          </Button>
        )}
      </Group>
      {withdraw.isError && <StatusLine msg={{ err: `Couldn't change the plan: ${messageOf(withdraw.error)}` }} />}
    </div>
  );
}
