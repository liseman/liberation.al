// The cameras of a scene of all cameras' clouds (#445), under the 3D view: a row a camera, with a box that shows
// or hides its points, its colour (the tint its points take when coloured by camera), and how it is placed or why
// it isn't in the scene. A camera that moved offers the spot its pictures measured, saved only when the owner takes
// it. Then what each overlapping pair's pictures said, and what a scene like this is and isn't.
import { Button, Checkbox, ColorSwatch, Group, Stack, Text } from '@mantine/core';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { messageOf } from '../../../shared/http';
import { viewName } from '../../review/views';
import { acceptSpot, type ScenePair, type ScenePlaced } from '../api';
import { tintCss } from './layers';
import { pairLine, placedLine, SCENE_LIMITS } from './status';

export function SceneLayers({
  place,
  cameras,
  pairs,
  hidden,
  onToggle,
  outside,
  inside,
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
}) {
  const client = useQueryClient();
  const accept = useMutation({
    mutationFn: (camera: string) => acceptSpot(place, camera),
    onSuccess: () => client.invalidateQueries({ queryKey: ['places'] }),
  });
  const lines = pairs.map(pair => pairLine(pair, viewName)).filter((line): line is string => !!line);
  const apart = pairs.length > 0 && pairs.every(pair => pair.state === 'apart');
  return (
    <Stack gap="xs" mt="sm">
      <Stack gap={6} component="ul" m={0} p={0} style={{ listStyle: 'none' }} aria-label="Cameras in the scene">
        {cameras.map((camera, index) => (
          <Group key={camera.camera} component="li" gap="sm" align="flex-start" wrap="nowrap">
            <Checkbox
              size="xs"
              mt={2}
              checked={camera.placed && !hidden.has(camera.camera)}
              disabled={!camera.placed}
              onChange={event => onToggle(camera.camera, event.currentTarget.checked)}
              aria-label={`Show ${viewName(camera.camera)}'s points`}
            />
            <ColorSwatch color={tintCss(index)} size={12} mt={5} aria-hidden withShadow={false} />
            <Text size="xs" flex={1}>
              <Text span fw={600} inherit>
                {viewName(camera.camera)}
              </Text>
              {'. '}
              <Text span c="dimmed" inherit>
                {placedLine(camera, viewName)}
                {camera.placed && outside[index]
                  ? ` ${outside[index]?.toLocaleString()} of its points fall outside your plan's rooms${inside ? ' and are left out' : ''}.`
                  : ''}
              </Text>
              {camera.suggest && (
                <>
                  {' '}
                  <Button
                    size="compact-xs"
                    variant="light"
                    loading={accept.isPending && accept.variables === camera.camera}
                    onClick={() => accept.mutate(camera.camera)}
                  >
                    Use this as its spot
                  </Button>
                </>
              )}
            </Text>
          </Group>
        ))}
      </Stack>
      {accept.isError && (
        <Text size="xs" c="red" role="alert">
          Couldn't save the spot: {messageOf(accept.error)}
        </Text>
      )}
      {lines.map(line => (
        <Text key={line} size="xs" c="dimmed">
          {line}
        </Text>
      ))}
      {apart && (
        <Text size="xs" c="dimmed">
          No two of these cameras see the same space, so each is placed from its spot alone.
        </Text>
      )}
      <Text size="xs" c="dimmed">
        {SCENE_LIMITS}
      </Text>
    </Stack>
  );
}
