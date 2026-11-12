// The cameras of a scene of all cameras' clouds (#445), under the 3D view: a row a camera, made to be scanned. A box
// that shows or hides its points, its colour (its outline in the scene, and its points' tint when coloured by camera),
// its name, its state in a word (Placed, Adjusted, Following, Tracking, Moved, Not placed, No spot), and at the right what can be done:
// look from it, or set its spot in the plan editor. The sentence behind the word, how its cloud was levelled and
// squared to the plan, what its pair's pictures said, when its picture was taken, and how many of its points are left
// out open under the row on demand.
// A camera that was turned or carried is placed again by itself (#509): nothing is asked. Its detail says what was
// done and offers the one way back, to the plan's spot, which changes nothing of the owner's plan and can be undone.
import { ActionIcon, Badge, Box, Button, Checkbox, Collapse, ColorSwatch, Group, Stack, Text } from '@mantine/core';
import { IconChevronDown } from '@tabler/icons-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf } from '../../../shared/http';
import { Toast } from '../../lib/Toast';
import { viewName } from '../../review/views';
import { followCamera, type ScenePair, type ScenePlaced } from '../api';
import { tintCss } from './layers';
import {
  APPROXIMATE,
  addedNote,
  bendLine,
  coveredLine,
  fitLine,
  levelLine,
  mapLine,
  pairLine,
  pictureLine,
  placedLine,
  placedState,
  SCENE_LIMITS,
  squareLine,
  trackLine,
  wayBack,
} from './status';

/** A way back that was taken: the camera, and which way it went. */
interface Went {
  camera: string;
  use: 'spot' | 'cloud';
}

export function SceneLayers({
  place,
  cameras,
  pairs,
  covered,
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
  /** How much of the plan's floor the place's map covers so far. */
  covered?: { floor_m2: number; plan_m2: number } | null;
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
  const [went, setWent] = useState<Went | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const back = useMutation({
    mutationFn: (to: Went) => followCamera(place, to.camera, to.use),
    onSuccess: (_, to) => {
      setFailed(null);
      setWent(to);
      return client.invalidateQueries({ queryKey: ['places'] });
    },
    onError: (error, to) => {
      setWent(null);
      setFailed(
        `Couldn't ${to.use === 'spot' ? "put it back on the plan's spot" : 'follow it again'}: ${messageOf(error)}`,
      );
    },
  });
  const toggle = (camera: string) =>
    setOpen(was => {
      const next = new Set(was);
      if (!next.delete(camera)) next.add(camera);
      return next;
    });
  return (
    <Stack gap="xs" mt="sm">
      <Text size="xs" c="dimmed">
        {[SCENE_LIMITS, coveredLine(covered)].filter(Boolean).join(' ')}
      </Text>
      <Stack gap={4} component="ul" m={0} p={0} style={{ listStyle: 'none' }} aria-label="Cameras in the scene">
        {cameras.map((camera, index) => {
          const name = viewName(camera.camera);
          const leftOut = camera.map?.state === 'refused' && camera.map.why === 'apart';
          const state = placedState(camera.how, camera.track?.state, leftOut);
          const tracked = trackLine(camera);
          const added = addedNote(camera);
          const shown = open.has(camera.camera);
          const way = wayBack(camera);
          const lines = [
            mapLine(camera),
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
                {added && (
                  <Text size="xs" c="dimmed" flex="0 0 auto" aria-label={added.label}>
                    <Box component="span" visibleFrom="xs" aria-hidden>
                      {added.wide}
                    </Box>
                    <Box component="span" hiddenFrom="xs" aria-hidden>
                      {added.narrow}
                    </Box>
                  </Text>
                )}
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
                {camera.how === 'lost' && way ? (
                  <Button
                    size="compact-xs"
                    variant="subtle"
                    flex="0 0 auto"
                    loading={back.isPending && back.variables?.camera === camera.camera}
                    onClick={() => back.mutate({ camera: camera.camera, use: way.use })}
                  >
                    {way.label}
                  </Button>
                ) : !camera.placed && camera.how !== 'lost' && onEditPlan ? (
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
                  {tracked && (
                    <Text size="xs" c="dimmed">
                      {tracked}
                    </Text>
                  )}
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
                  {way && camera.how !== 'lost' && (
                    <Button
                      size="compact-xs"
                      variant="subtle"
                      w="fit-content"
                      px={0}
                      loading={back.isPending && back.variables?.camera === camera.camera}
                      onClick={() => back.mutate({ camera: camera.camera, use: way.use })}
                    >
                      {way.label}
                    </Button>
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
      <Toast<Went>
        message={
          went
            ? went.use === 'spot'
              ? { text: `${viewName(went.camera)} is back on your plan's spot.`, undo: went }
              : { text: `${viewName(went.camera)} is followed again.` }
            : failed
              ? { err: failed }
              : null
        }
        onUndo={was => back.mutate({ camera: was.camera, use: 'cloud' })}
        busy={back.isPending}
        onClose={() => {
          setWent(null);
          setFailed(null);
        }}
      />
    </Stack>
  );
}
