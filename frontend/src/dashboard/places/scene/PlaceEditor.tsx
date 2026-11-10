// A place's plan in the new plan editor (or its 3D view), with Save. The editor works on a Pascal scene: the
// place's own when its plan is one, or its DSL plan converted by the server on the way out (GET /places/plan/scene),
// in which case saving is what turns the place's plan into a scene. Saving writes the owner's plan, as correcting it
// always has (POST /places/plan), so every save is a version in the place's plan history. Those versions open from
// a Versions button beside Save (Versions.tsx), so the editor keeps the card's full width: picking one opens it in
// the editor to look at only, with nothing that changes it, and Restore saves it again on top. An example place has
// nothing to save to and no versions.
import { Alert, Badge, Button, Group, Popover, Text } from '@mantine/core';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import type { PlanView, Scene } from '../../../plan-editor/protocol';
import { messageOf } from '../../../shared/http';
import { confirm } from '../../lib/confirm';
import { momentWords } from '../../lib/format';
import { BlockSkeleton } from '../../lib/loading';
import { shell } from '../../shell';
import { type PlanVersion, planSceneQuery, restorePlan, savePlan } from '../api';
import type { Place } from '../examples';
import classes from '../places.module.css';
import { PlanFrame } from './PlanFrame';
import { Versions } from './Versions';

/** The house's cameras as the editor's Cameras panel lists them: each with the name the page shows. */
export const houseCameras = (): { id: string; name: string }[] =>
  shell().cams.map(id => ({ id, name: shell().views[id] ?? id }));

/** How many rooms a scene has: a plan with none can't be saved. */
export const roomsIn = (scene: Scene): number => Object.values(scene.nodes).filter(n => n.type === 'zone').length;

export function PlaceEditor({
  place,
  view: shown,
  scheme,
  onSaved,
  onUnsaved,
}: {
  place: Place;
  view: PlanView;
  scheme: 'light' | 'dark';
  onSaved?: () => void;
  /** Whether the editor holds changes that haven't been saved, each time that changes. */
  onUnsaved?: (unsaved: boolean) => void;
}) {
  const client = useQueryClient();
  // An earlier version being looked at (Versions.tsx), or null for the plan as it is now.
  const [viewing, setViewing] = useState<PlanVersion | null>(null);
  const [listed, setListed] = useState(false);
  const stored = useQuery({ ...planSceneQuery(place.id, viewing?.id), enabled: !place.example });
  const scene = place.example ? place.scene : stored.data?.scene;
  const [edited, setEdited] = useState<Scene | null>(null);
  const dirty = edited !== null;
  // biome-ignore lint/correctness/useExhaustiveDependencies: told when it changes, whoever is listening
  useEffect(() => onUnsaved?.(dirty), [dirty]);
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
  const restore = useMutation({
    mutationFn: (version: PlanVersion) => restorePlan(place.id, version.id),
    onSuccess: () => {
      setViewing(null);
      setEdited(null);
      setOpened(n => n + 1);
      client.invalidateQueries({ queryKey: ['places'] });
    },
  });
  /** Look at another version (or back at the current plan), asking first when that would drop unsaved changes. */
  const view = async (version: PlanVersion | null) => {
    if (version?.id === viewing?.id) return;
    if (
      edited &&
      !(await confirm({
        title: 'Discard your unsaved changes?',
        body: "The changes to this plan that you haven't saved will be lost.",
        confirm: 'Discard changes',
        danger: true,
        id: 'plan-versions',
      }))
    )
      return;
    setEdited(null);
    restore.reset();
    setViewing(version);
  };
  if (!place.example && stored.isPending) return <BlockSkeleton label="the plan" />;
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
  // The plan's versions, in a popover from a button beside Save: the editor keeps the card's full width.
  const versions = !place.example && (
    <Popover opened={listed} onChange={setListed} position="bottom-end" width={300} shadow="md" withinPortal>
      <Popover.Target>
        <Button size="xs" variant="default" aria-expanded={listed} onClick={() => setListed(open => !open)}>
          Versions
        </Button>
      </Popover.Target>
      <Popover.Dropdown>
        <Versions
          place={place.id}
          viewing={viewing?.id ?? null}
          onView={version => {
            setListed(false);
            view(version);
          }}
        />
      </Popover.Dropdown>
    </Popover>
  );
  return (
    <>
      <Group justify="space-between" gap="xs" mb="xs" wrap="wrap">
        <Text size="sm" c="dimmed">
          {place.example ? (
            'An example flat. Nothing drawn here is saved.'
          ) : viewing ? (
            <>
              <Badge variant="light" color="gray" tt="none" fw={500} mr={6}>
                Earlier version
              </Badge>
              Saved {momentWords(viewing.at)}. Looking at it changes nothing.
            </>
          ) : stored.data?.converted ? (
            'Saving here lets you edit this plan in 3D.'
          ) : shown === '3d' ? (
            'The plan in 3D.'
          ) : (
            'Draw rooms and walls, and put doors and windows in them.'
          )}
        </Text>
        {viewing ? (
          <Group gap="xs" className={classes.editorActions}>
            {versions}
            <Button size="xs" variant="default" disabled={restore.isPending} onClick={() => view(null)}>
              Back to the current plan
            </Button>
            <Button size="xs" loading={restore.isPending} onClick={() => restore.mutate(viewing)}>
              Restore this version
            </Button>
          </Group>
        ) : (
          !place.example && (
            <Group gap="xs" className={classes.editorActions}>
              {edited && (
                <Text size="sm" c="var(--lib-warn-text)" role="status">
                  Unsaved changes
                </Text>
              )}
              {versions}
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
          )
        )}
      </Group>
      {left.length > 0 && !viewing && (
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
      {restore.isError && (
        <Alert color="red" variant="light" mb="xs">
          Couldn't restore that version: {messageOf(restore.error)}
        </Alert>
      )}
      <PlanFrame
        key={`${place.id}-${shown}-${viewing?.id ?? 'now'}-${opened}`}
        scene={scene}
        view={shown}
        scheme={scheme}
        cameras={place.example ? [] : houseCameras()}
        look={viewing !== null}
        onChange={viewing ? undefined : setEdited}
      />
    </>
  );
}
