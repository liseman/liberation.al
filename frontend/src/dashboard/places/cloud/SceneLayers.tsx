// The cameras of a scene of all cameras' clouds (#445), under the 3D view: a row a camera, made to be scanned. A box
// that shows or hides its points, its colour (its outline in the scene, and its points' tint when coloured by camera),
// its name, its state in a word (Placed, Adjusted, Moved, Not placed, No spot), and at the right what can be done:
// look from it, move its spot to where its pictures put it, or set its spot in the plan editor. The sentence behind
// the word, how its cloud was levelled and squared to the plan, what its pair's pictures said, when its picture was
// taken, and how many of its points are left out open under the row on demand.
// Moving a spot changes the owner's plan, so it asks first and says how to take it back.
import { ActionIcon, Badge, Button, Checkbox, Collapse, ColorSwatch, Group, Stack, Text } from '@mantine/core';
import { IconChevronDown } from '@tabler/icons-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf } from '../../../shared/http';
import { confirm } from '../../lib/confirm';
import { Toast } from '../../lib/Toast';
import { viewName } from '../../review/views';
import { acceptSpot, type ScenePair, type ScenePlaced, undoSpot } from '../api';
import { tintCss } from './layers';
import {
  APPROXIMATE,
  bendLine,
  fitLine,
  levelLine,
  moveSpotQuestion,
  pairLine,
  pictureLine,
  placedLine,
  placedState,
  SCENE_LIMITS,
  squareLine,
} from './status';

/** A spot that was moved, and the plan version to put back to undo it. */
interface Moved {
  camera: string;
  previous: string | null;
}

export function SceneLayers({
  place,
  cameras,
  pairs,
  hidden,
  onToggle,
  outside,
  inside,
  onLookFrom,
  onEditPlan,
}: {
  place: string;
  cameras: readonly ScenePlaced[];
  pairs: readonly ScenePair[];
  /** The cameras whose points are hidden. */
  hidden: ReadonlySet<string>;
  onToggle: (camera: string, show: boolean) => void;
  /** How many of each camera's points lie outside the plan, by its place in `cameras`. */
  outside: readonly number[];
  /** Whether the points outside the plan are left out of the view. */
  inside: boolean;
  /** Fly the view to where a camera stands. */
  onLookFrom: (camera: string) => void;
  /** Open the plan editor, where a camera's spot is set; absent where the plan can't be edited from here. */
  onEditPlan?: () => void;
}) {
  const client = useQueryClient();
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set());
  const [moved, setMoved] = useState<Moved | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const refresh = () => client.invalidateQueries({ queryKey: ['places'] });
  const accept = useMutation({
    mutationFn: (camera: string) => acceptSpot(place, camera),
    onSuccess: (reply, camera) => {
      setMoved({ camera, previous: reply.previous });
      return refresh();
    },
    onError: error => setFailed(`Couldn't move the spot: ${messageOf(error)}`),
  });
  const undo = useMutation({
    mutationFn: (was: Moved) => undoSpot(place, was.camera, was.previous ?? ''),
    onSuccess: () => {
      setMoved(null);
      return refresh();
    },
    onError: error => {
      setMoved(null);
      setFailed(`Couldn't put the spot back: ${messageOf(error)}`);
    },
  });
  const moveSpot = async (camera: ScenePlaced) => {
    const asked = moveSpotQuestion(camera, viewName(camera.camera));
    if (await confirm({ ...asked, confirm: 'Move its spot', id: `move-spot-${camera.camera}` }))
      accept.mutate(camera.camera);
  };
  const toggle = (camera: string) =>
    setOpen(was => {
      const next = new Set(was);
      if (!next.delete(camera)) next.add(camera);
      return next;
    });
  return (
    <Stack gap="xs" mt="sm">
      <Text size="xs" c="dimmed">
        {SCENE_LIMITS}
      </Text>
      <Stack gap={4} component="ul" m={0} p={0} style={{ listStyle: 'none' }} aria-label="Cameras in the scene">
        {cameras.map((camera, index) => {
          const name = viewName(camera.camera);
          const state = placedState(camera.how);
          const shown = open.has(camera.camera);
          const lines = [
            camera.fit ? fitLine(camera.fit) : null,
            squareLine(camera),
            levelLine(camera.level, camera.tilt_deg, camera.roll_deg),
            bendLine(camera),
            ...pairs
              .filter(pair => pair.a === camera.camera || pair.b === camera.camera)
              .map(pair => pairLine(pair, viewName)),
            pictureLine(camera),
          ].filter((line): line is string => !!line);
          return (
            <li key={camera.camera}>
              <Group gap="xs" wrap="nowrap" align="center">
                <Checkbox
                  size="xs"
                  checked={camera.placed && !hidden.has(camera.camera)}
                  disabled={!camera.placed}
                  onChange={event => onToggle(camera.camera, event.currentTarget.checked)}
                  aria-label={`Show ${name}'s points`}
                />
                <ColorSwatch color={tintCss(index)} size={12} aria-hidden withShadow={false} />
                <Text size="sm" truncate flex={1} miw={0}>
                  {name}
                </Text>
                <Badge
                  variant="light"
                  tt="none"
                  fw={500}
                  color={state.tone === 'warn' ? 'yellow' : 'gray'}
                  c={state.tone === 'warn' ? 'var(--lib-warn-text)' : undefined}
                  flex="0 0 auto"
                >
                  {state.label}
                </Badge>
                {camera.suggest ? (
                  <Button
                    size="compact-xs"
                    variant="light"
                    flex="0 0 auto"
                    loading={accept.isPending && accept.variables === camera.camera}
                    onClick={() => void moveSpot(camera)}
                  >
                    Move its spot
                  </Button>
                ) : !camera.placed && onEditPlan ? (
                  <Button size="compact-xs" variant="subtle" flex="0 0 auto" onClick={onEditPlan}>
                    Set its spot
                  </Button>
                ) : null}
                {camera.placed && (
                  <Button
                    size="compact-xs"
                    variant="subtle"
                    flex="0 0 auto"
                    visibleFrom="xs"
                    onClick={() => onLookFrom(camera.camera)}
                  >
                    Look from
                  </Button>
                )}
                <ActionIcon
                  size="sm"
                  variant="subtle"
                  color="gray"
                  aria-expanded={shown}
                  aria-label={`Details of ${name}`}
                  onClick={() => toggle(camera.camera)}
                >
                  <IconChevronDown
                    size={14}
                    style={{ transform: shown ? 'rotate(180deg)' : undefined, transition: 'transform 120ms' }}
                  />
                </ActionIcon>
              </Group>
              <Collapse expanded={shown}>
                <Stack gap={2} pl={44} pt={2} pb="xs">
                  <Text size="xs" c="dimmed">
                    {placedLine(camera, viewName)}
                  </Text>
                  {lines.map(line => (
                    <Text key={line} size="xs" c="dimmed">
                      {line}
                    </Text>
                  ))}
                  {camera.placed && (
                    <Text size="xs" c="dimmed">
                      {camera.fitted ? '' : APPROXIMATE}
                      {outside[index]
                        ? ` ${outside[index]?.toLocaleString()} of its points fall outside your plan's rooms${inside ? ' and are left out' : ''}.`
                        : ''}
                    </Text>
                  )}
                  {camera.placed && (
                    <Button
                      size="compact-xs"
                      variant="subtle"
                      hiddenFrom="xs"
                      w="fit-content"
                      onClick={() => onLookFrom(camera.camera)}
                    >
                      Look from
                    </Button>
                  )}
                </Stack>
              </Collapse>
            </li>
          );
        })}
      </Stack>
      <Toast
        text={moved ? `Moved ${viewName(moved.camera)}'s spot on your plan.` : failed}
        error={!moved && !!failed}
        action={
          moved?.previous ? (
            <Button size="compact-xs" variant="subtle" disabled={undo.isPending} onClick={() => undo.mutate(moved)}>
              Undo
            </Button>
          ) : undefined
        }
        onClose={() => {
          setMoved(null);
          setFailed(null);
        }}
      />
    </Stack>
  );
}
