// The one round play/stop control for a clip (step 5): Who's this?'s voices, a person's lines and voice
// picks, and the clip picker all use it, so playing anything looks and reads the same.
import { ActionIcon } from '@mantine/core';
import { IconPlayerPlayFilled, IconPlayerStopFilled } from '@tabler/icons-react';
import { play, usePlayer } from './player';

const SIZES = { xs: 22, sm: 26, lg: 48 } as const;

export function PlayToggle({
  playing,
  label,
  onClick,
  size = 'sm',
  className,
}: {
  playing: boolean;
  label: string; // its whole accessible name, as the caller words it ("Play Face 5", "Stop: Hello")
  onClick: () => void;
  size?: keyof typeof SIZES;
  className?: string;
}) {
  const Icon = playing ? IconPlayerStopFilled : IconPlayerPlayFilled;
  return (
    <ActionIcon
      variant={playing ? 'filled' : 'default'}
      radius="xl"
      size={SIZES[size]}
      className={className}
      data-on={playing || undefined}
      aria-label={label}
      onClick={onClick}
    >
      <Icon size={Math.round(SIZES[size] * 0.45)} aria-hidden />
    </ActionIcon>
  );
}

/** A play/stop button for one clip through the shared player, so a second press stops it. `label` is what plays, after
 * "Play " or "Stop "; `playKey` is what the player calls the clip while it plays. */
export function PlayClip({
  url,
  playKey,
  ts = null,
  label,
  size,
  className,
}: {
  url: string;
  playKey: string;
  ts?: number | null;
  label: string;
  size?: keyof typeof SIZES;
  className?: string;
}) {
  const playing = usePlayer(s => s.key === playKey);
  return (
    <PlayToggle
      playing={playing}
      size={size}
      className={className}
      label={`${playing ? 'Stop' : 'Play'} ${label}`}
      onClick={() => play(url, playKey, ts ?? null)}
    />
  );
}
