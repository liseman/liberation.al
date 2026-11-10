// A place's plan in the new plan editor (or its 3D view), with Save. The editor works on a Pascal scene: the
// place's own when its plan is one, or its DSL plan converted by the server on the way out (GET /places/plan/scene),
// in which case saving is what turns the place's plan into a scene. Saving writes the owner's plan, as correcting it
// always has (POST /places/plan), so every save is a version in the place's plan history. An example place has
// nothing to save to.
import { Alert, Button, Group, Text } from '@mantine/core';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import type { PlanView, Scene } from '../../../plan-editor/protocol';
import { messageOf } from '../../../shared/http';
import { planSceneQuery, savePlan } from '../api';
import type { Place } from '../examples';
import { PlanFrame } from './PlanFrame';

/** How many rooms a scene has: a plan with none can't be saved. */
export const roomsIn = (scene: Scene): number => Object.values(scene.nodes).filter(n => n.type === 'zone').length;

export function PlaceEditor({
  place,
  view,
  scheme,
  onSaved,
}: {
  place: Place;
  view: PlanView;
  scheme: 'light' | 'dark';
  onSaved?: () => void;
}) {
  const client = useQueryClient();
  const stored = useQuery({ ...planSceneQuery(place.id), enabled: !place.example });
  const scene = place.example ? place.scene : stored.data?.scene;
  const [edited, setEdited] = useState<Scene | null>(null);
  // a new key makes the frame again, on the stored plan: that is what Discard does
  const [opened, setOpened] = useState(0);
  const save = useMutation({
    mutationFn: (next: Scene) => savePlan(place.id, JSON.stringify(next), stored.data?.base),
    onSuccess: () => {
      setEdited(null);
      client.invalidateQueries({ queryKey: ['places'] }); // the listing, the history, and this plan's scene
      onSaved?.();
    },
  });
  if (!place.example && stored.isPending)
    return (
      <Text size="sm" c="dimmed">
        Loading the plan…
      </Text>
    );
  if (!scene)
    return (
      <Alert color="red" variant="light">
        {stored.error ? messageOf(stored.error) : "This place's plan can't be opened in the editor."}
      </Alert>
    );
  const left = stored.data?.left ?? [];
  const empty = edited !== null && roomsIn(edited) === 0;
  // a plan still in the old format can be saved as it opened: that is what stores it in the new one
  const toSave = edited ?? (stored.data?.converted ? scene : null);
  return (
    <>
      <Group justify="space-between" gap="xs" mb="xs" wrap="wrap">
        <Text size="sm" c="dimmed">
          {place.example
            ? 'An example flat. Nothing drawn here is saved.'
            : stored.data?.converted
              ? 'This plan is stored in the old format. Saving it here stores it in the new one.'
              : edited
                ? 'Changes not saved yet.'
                : view === '3d'
                  ? 'The plan in 3D.'
                  : 'Draw rooms and walls, and put doors and windows in them.'}
        </Text>
        {!place.example && (
          <Group gap="xs">
            <Button
              size="xs"
              variant="default"
              disabled={!edited || save.isPending}
              onClick={() => {
                setEdited(null);
                setOpened(n => n + 1);
              }}
            >
              Discard changes
            </Button>
            <Button
              size="xs"
              disabled={!toSave || empty}
              loading={save.isPending}
              onClick={() => toSave && save.mutate(toSave)}
            >
              Save plan
            </Button>
          </Group>
        )}
      </Group>
      {left.length > 0 && (
        <Text size="xs" c="dimmed" mb="xs">
          {left.length === 1 ? 'One line' : `${left.length} lines`} of the old plan have no place in the new format and
          are left out when it is saved: {left.join('; ')}
        </Text>
      )}
      {empty && (
        <Alert color="yellow" variant="light" mb="xs">
          A plan needs at least one room before it can be saved.
        </Alert>
      )}
      {save.isError && (
        <Alert color="red" variant="light" mb="xs">
          Couldn't save the plan: {messageOf(save.error)}
        </Alert>
      )}
      <PlanFrame key={`${place.id}-${view}-${opened}`} scene={scene} view={view} scheme={scheme} onChange={setEdited} />
    </>
  );
}
