// The top bar's status (#2453): one pill that reads as state first ("Listening · 6 cameras") and opens the privacy
// menu second. Its microphone and camera glyphs keep their colours (on, late, off, or paused), and the live feed's
// glyph joins them only when it's down. The full words are the pill's accessible name and the menu's first lines.
// On this machine the menu pauses capture for privacy (#2260): all of a kind with one item, or each device on its own.
import { Menu, UnstyledButton } from '@mantine/core';
import {
  IconCameraFilled,
  IconCameraOff,
  IconChevronDown,
  IconMicrophoneFilled,
  IconMicrophoneOff,
  IconPlugConnectedX,
  type TablerIcon,
} from '@tabler/icons-react';
import classes from './app.module.css';
import { useLive } from './live/hooks';
import { useLiveStatus } from './live/status';
import { canChange, type PrivacyKind, type PrivacySummary, summary, usePrivacy, usePrivacyChange } from './privacy';
import { viewName } from './review/views';
import { useRoute } from './router';
import { useActiveCameras } from './shell';
import { camStatus, camText, camTrouble, type Health, micStatus, micText } from './status';

type IconState = Health | 'bad' | 'paused';

const COLOR: Record<IconState, string> = {
  on: 'var(--lib-ok)',
  some: 'var(--lib-warn)',
  stale: 'var(--lib-warn)',
  off: 'var(--mantine-color-dimmed)',
  bad: 'var(--lib-bad)',
  paused: 'var(--lib-bad)',
};

/** The worst of the states shown, for the pill's dot. */
const RANK: IconState[] = ['bad', 'paused', 'stale', 'some', 'off', 'on'];
const worst = (states: IconState[]): IconState => RANK.find(r => states.includes(r)) ?? 'off';

const NOUN: Record<PrivacyKind, [string, string]> = {
  camera: ['camera', 'cameras'],
  microphone: ['microphone', 'microphones'],
};

const nameOf = (kind: PrivacyKind, name: string): string => (kind === 'camera' ? viewName(name) : name);

/** The paused part of the words: "Paused for privacy", or which are paused when only some are. */
export function pausedText(kind: PrivacyKind, s: PrivacySummary): string {
  const waiting = s.pending ? ' (applying…)' : '';
  if (s.all) return `Paused for privacy${waiting}`;
  return `${s.paused.map(n => nameOf(kind, n)).join(', ')} paused for privacy${waiting}`;
}

function Glyph({
  icon: Icon,
  state,
  testId,
  label,
}: {
  icon: TablerIcon;
  state: IconState;
  testId: string;
  label: string;
}) {
  return (
    <span
      className={classes.glyph}
      role="img"
      aria-label={label}
      data-testid={testId}
      data-state={state}
      style={{ color: COLOR[state] }}
    >
      <Icon size={16} stroke={1.75} aria-hidden />
    </span>
  );
}

/** One kind's lines in the menu: what it's doing, then pause or resume all of them, then each on its own. */
function KindItems({
  kind,
  s,
  text,
  control,
  change,
}: {
  kind: PrivacyKind;
  s: PrivacySummary;
  text: string;
  control: boolean;
  /** The pill's own mutation, so a failure lands where the pill shows it. */
  change: ReturnType<typeof usePrivacyChange>;
}) {
  const [one, many] = NOUN[kind];
  // Some paused: the main item pauses the rest, the safe way. All paused: it resumes them.
  const next = !s.all;
  const plural = s.devices.length > 1 ? `all ${many}` : `the ${one}`;
  const On = kind === 'camera' ? IconCameraFilled : IconMicrophoneFilled;
  const Off = kind === 'camera' ? IconCameraOff : IconMicrophoneOff;
  return (
    <>
      <Menu.Label className={classes.statusLine}>{text}</Menu.Label>
      {control && (
        <Menu.Item
          leftSection={next ? <Off size={16} aria-hidden /> : <On size={16} aria-hidden />}
          onClick={() => change.mutate({ kind, paused: next })}
        >
          {`${next ? 'Pause' : 'Resume'} ${plural}`}
        </Menu.Item>
      )}
      {control &&
        s.devices.length > 1 &&
        s.devices.map(d => (
          <Menu.Item
            key={d.name}
            pl={36}
            onClick={() => change.mutate({ kind, paused: !d.paused, device: d.name })}
            rightSection={d.paused ? <Off size={14} color="var(--lib-bad)" aria-hidden /> : null}
          >
            {`${d.paused ? 'Resume' : 'Pause'} ${nameOf(kind, d.name)}`}
          </Menu.Item>
        ))}
    </>
  );
}

/** The short words on the pill: the microphone, then the cameras. */
function shortWords(
  mic: ReturnType<typeof micStatus>,
  mics: PrivacySummary,
  cam: ReturnType<typeof camStatus>,
  cameras: PrivacySummary,
  total: number,
): string {
  const m = mics.all
    ? 'Mic paused'
    : mic.state === 'on'
      ? 'Listening'
      : mic.state === 'stale'
        ? 'Mic late'
        : 'Not listening';
  let c: string;
  if (cameras.all) c = 'cameras paused';
  else if (cam.state === 'some' || cameras.paused.length) c = `${cam.live.length} of ${total} cameras`;
  else if (cam.state === 'on') c = `${total} ${total === 1 ? 'camera' : 'cameras'}`;
  else c = 'cameras off';
  return `${m} · ${c}`;
}

export function StatusPill() {
  const tab = useRoute(s => s.tab);
  const ambient = useLive('ambient');
  // Outside Live, scene is only header status and must not occupy the page-data slot.
  const scene = useLive('scene', tab === 'live' ? 'page' : 'background');
  const vision = useLive('vision');
  const now = useLiveStatus(s => s.now);
  const conn = useLiveStatus(s => s.conn);
  const cams = useActiveCameras();
  const privacy = usePrivacy();
  const change = usePrivacyChange();
  const mics = summary(privacy, 'microphone');
  const cameras = summary(privacy, 'camera');
  const mic = micStatus(ambient, now);
  const cam = camStatus(scene, cams, now, vision, cameras.paused);

  const micState: IconState = mics.paused.length ? 'paused' : mic.state;
  const camState: IconState = cameras.paused.length ? 'paused' : cam.state;
  const micWords = mics.paused.length ? pausedText('microphone', mics) : micText(mic);
  const camWords = cameras.paused.length ? pausedText('camera', cameras) : camText(cam, cams, camTrouble(vision, now));
  const feedDown = conn.state === 'bad';
  const feedWords = conn.text === 'polling' ? 'Live updates off: checking every 3 s' : 'Reconnecting to live updates';
  const short = feedDown ? 'Reconnecting' : shortWords(mic, mics, cam, cameras, cams.length);

  const micControl = canChange(privacy, 'microphone');
  const camControl = canChange(privacy, 'camera');
  const control = micControl || camControl;
  const words = [feedDown ? feedWords : null, `Microphone: ${micWords}`, `Cameras: ${camWords}`]
    .filter(Boolean)
    .join('. ');
  const error = change.error instanceof Error ? change.error.message : '';
  // A pause or resume that failed turns the dot red and leads the words, so it shows without opening the menu.
  const dot = error ? 'bad' : worst(feedDown ? ['bad', micState, camState] : [micState, camState]);

  return (
    <Menu position="bottom-end" withinPortal width={280}>
      <Menu.Target>
        <UnstyledButton
          className={classes.pill}
          // A failed change leads, then its visible words (label in name), then the detail behind them.
          aria-label={`${error ? `Couldn't change privacy: ${error}. ` : ''}${short}. ${words}.${control ? ' Open to pause or resume.' : ''}`}
          data-testid="status"
          data-state={dot}
          data-pending={mics.pending || cameras.pending || undefined}
        >
          <span className={classes.dot} style={{ background: COLOR[dot] }} aria-hidden />
          {feedDown && <Glyph icon={IconPlugConnectedX} state="bad" testId="status-feed" label={feedWords} />}
          <Glyph
            icon={micState === 'paused' || mic.state === 'off' ? IconMicrophoneOff : IconMicrophoneFilled}
            state={micState}
            testId="status-mic"
            label={micWords}
          />
          <Glyph
            icon={camState === 'paused' || cam.state === 'off' ? IconCameraOff : IconCameraFilled}
            state={camState}
            testId="status-cams"
            label={camWords}
          />
          <span className={classes.pillText} aria-hidden>
            {short}
          </span>
          <IconChevronDown size={14} aria-hidden className={classes.pillChevron} />
        </UnstyledButton>
      </Menu.Target>
      <Menu.Dropdown aria-label="Status and privacy">
        {feedDown && <Menu.Label className={classes.statusLine}>{feedWords}</Menu.Label>}
        <KindItems kind="microphone" s={mics} text={micWords} control={micControl} change={change} />
        <Menu.Divider />
        <KindItems kind="camera" s={cameras} text={camWords} control={camControl} change={change} />
        {!privacy?.control && (
          <>
            <Menu.Divider />
            <Menu.Label>Pausing works only on the machine Liberational runs on.</Menu.Label>
          </>
        )}
        {error && (
          <Menu.Label c="var(--lib-bad)" role="alert">
            {error}
          </Menu.Label>
        )}
      </Menu.Dropdown>
    </Menu>
  );
}
