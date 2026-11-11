// Live: the cameras and the conversation as they happen. Historical people, clips, and the day's timeline are in
// Review. With no cameras, no microphone, and no recent conversation, it shows setup instead.
import './live.css';
import { VisuallyHidden } from '@mantine/core';
import { IconMicrophoneOff } from '@tabler/icons-react';
import { type MouseEventHandler, useRef } from 'react';
import { useGatewayLiveCams } from '../camera/gateway';
import { Waveform } from '../camera/Waveform';
import { usePrefs } from '../lib/prefs';
import { useLive } from '../live/hooks';
import { useLiveStatus } from '../live/status';
import { linkClick } from '../router';
import { useActiveCameras } from '../shell';
import { CAM_FRESH_S, camStatus, camTrouble, micStatus } from '../status';
import { GettingStarted } from './GettingStarted';
import { Happening } from './HappeningPanel';
import { HereNow } from './HereNow';
import { Mics } from './Mics';
import type { LiveScene } from './Rail';
import './rail.css';
import { Setup, StartButton } from './Setup';
import { Stage } from './Stage';
import { Transcript } from './Transcript';

/** The microphone off, as a state of the conversation header's mic line: it takes that line's place
 *  instead of a notice above the page, which pushed everything down when it came and up when it went. */
function MicOff() {
  return (
    <section className="mics mics-compact" aria-label="Microphones">
      <span className="mic-chip mic-compact mic-off" role="status">
        <IconMicrophoneOff size={14} aria-hidden="true" />
        Not listening
      </span>
      <StartButton name="listen" label="Start listening" size="compact-xs" />
    </section>
  );
}

export function LiveTab() {
  const naming = useRef<(() => boolean) | null>(null);
  const nameFace: MouseEventHandler<HTMLAnchorElement> = event => {
    if (
      !event.defaultPrevented &&
      event.button === 0 &&
      !event.metaKey &&
      !event.ctrlKey &&
      !event.shiftKey &&
      !event.altKey &&
      naming.current?.()
    ) {
      event.preventDefault();
      return;
    }
    linkClick(event);
  };
  const ambient = useLive('ambient');
  const sceneRead = useLive('scene');
  const scene = (sceneRead ?? []) as LiveScene[];
  const vision = useLive('vision');
  const transcript = useLive('transcript');
  const now = useLiveStatus(s => s.now);
  const waveform = usePrefs(s => s.waveform === true);
  const listening = micStatus(ambient, now).state === 'on';
  const activeCams = useActiveCameras();
  const v1 = camStatus(scene, activeCams, now);
  // Continuum's word on which cameras have live video, alongside v1's captions while both run
  const viaGateway = useGatewayLiveCams(CAM_FRESH_S);
  const cams = { ...v1, live: [...new Set([...v1.live, ...viaGateway.filter(c => activeCams.includes(c))])] };
  const trouble = camTrouble(vision, now); // a camera starting or being reopened: say so, not "off"
  const last: Record<string, LiveScene> = Object.create(null);
  for (const e of scene) if (!last[e.cam] || e.ts >= (last[e.cam]?.ts ?? 0)) last[e.cam] = e;

  // Until the microphone's first status read answers, nothing is known to be off: saying "isn't listening yet" (the
  // Setup page) or "not listening" (the notice) for that moment drew a page and then pulled it away (layout shift).
  const known = ambient !== undefined;
  if (known && !listening && !cams.live.length && !trouble && !transcript?.entries.length) return <Setup />;
  return (
    <>
      {listening && waveform && <Waveform />}
      <div className="live-tab">
        {/* The tab bar already says this is Live; the heading is for screen readers and focus. */}
        <VisuallyHidden component="h1">Live</VisuallyHidden>
        {!(window as Window & { __PACK_DEMO__?: boolean }).__PACK_DEMO__ && <GettingStarted onNameFace={nameFace} />}
        {/* Wide screens: the cameras on the left with who's here under them, and the conversation and what's
            happening in a full-height right rail. Narrow ones stack cameras, conversation, then
            who's here. */}
        <div className="live-main">
          <Stage last={last} live={cams.live} trouble={trouble} naming={naming} sceneKnown={sceneRead !== undefined} />
          <div className="live-side">
            <Transcript
              listening={listening}
              meter={
                listening ? (
                  <Mics compact />
                ) : known ? (
                  <MicOff />
                ) : (
                  <section className="mics mics-compact" aria-hidden="true" />
                )
              }
            />
            {/* Not a landmark of its own: the card inside is the "What's happening" region (one name each) */}
            <div className="rail">
              <Happening />
            </div>
          </div>
          {/* The card inside is the "Here now" region */}
          <div className="live-here">
            <HereNow onNameFace={nameFace} />
          </div>
        </div>
      </div>
    </>
  );
}
