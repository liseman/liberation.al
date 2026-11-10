// Live's microphones (#353): one chip per microphone the microphone service is listening to, with its level moving
// live, a mark where speech has to reach (its sensitivity's threshold, #331), and whether the listen worker detects a sound candidate now.
import { VisuallyHidden } from '@mantine/core';
import { IconMicrophoneFilled, IconMicrophoneOff } from '@tabler/icons-react';
import { useQueryClient } from '@tanstack/react-query';
import { type CSSProperties, useEffect, useRef, useState } from 'react';
import { useLive } from '../live/hooks';
import type { AmbientDevice } from '../live/types';
import { summary, usePrivacy } from '../privacy';
import { DEVICES_KEY, type Devices, gainText } from '../sources/api';
import { levelWidth, subscribeLevels, useLiveLevels } from '../sources/levels';
import './mics.css';

/** A short name for a microphone's device name: the Sources picker's when it has one, else the PipeWire or ALSA
 *  name tidied ("alsa_input.usb-Elgato_Systems_Elgato_Wave_3_A017A41910AAFN-00.mono-fallback" is "Elgato Wave 3"). */
export function micName(raw: string, known: ReadonlyMap<string, string> = new Map()): string {
  const given = known.get(raw);
  if (given && given !== raw) return given;
  const m = raw.match(/^alsa_input\.(.+?)(?:\.[a-z][\w-]*)?$/);
  if (!m?.[1]) return raw; // not a PipeWire input name (a Mac's "MacBook Pro Microphone"): it reads fine as it is
  const id = m[1];
  if (id.startsWith('pci-') || id.startsWith('platform-')) return 'Built-in microphone';
  const words = id
    .replace(/^usb-/, '')
    .replace(/-\d+$/, '') // the interface number
    .split(/[_\s]+/)
    .filter(Boolean);
  // udev names a USB device vendor_model_serial: drop the last word when it looks like a serial number.
  const last = words.at(-1);
  if (words.length > 2 && last && last.length >= 8 && /\d/.test(last) && last === last.toUpperCase()) words.pop();
  const first = words[0];
  const again = first ? words.indexOf(first, 1) : -1; // "Elgato Systems Elgato Wave 3": the vendor, then the product
  const name = (again > 0 ? words.slice(again) : words).join(' ');
  return name || raw;
}

function Chip({
  slug,
  device,
  name,
  compact = false,
}: {
  slug: string;
  device: AmbientDevice;
  name: string;
  compact?: boolean;
}) {
  const speech = useSpeech(slug);
  const healthy = device.healthy !== false;
  const level = typeof device.level_dbfs === 'number' ? device.level_dbfs : -120;
  const threshold = typeof device.threshold_dbfs === 'number' ? device.threshold_dbfs : null;
  const state = !healthy ? 'Not responding' : speech == null ? 'Unavailable' : speech ? 'Sound detected' : 'Quiet';
  // Compact, in the conversation's header (#2456): what it is doing in words, with the name and gain on hover.
  // No levels yet, or the feed stalled (useSpeech's undefined): say so rather than quiet (#2486).
  const said = !healthy
    ? 'Not responding'
    : speech == null
      ? 'Listening · level unavailable'
      : speech
        ? 'Listening · sound'
        : 'Listening · quiet';
  const detail = [device.name, device.gain?.auto && device.gain.db != null ? gainText(device.gain) : null]
    .filter(Boolean)
    .join(' · ');
  if (compact)
    return (
      <div className="mic-chip mic-compact" title={`${name}${detail ? ` (${detail})` : ''}`}>
        <IconMicrophoneFilled size={14} stroke={2} aria-hidden="true" />
        <VisuallyHidden>
          <meter
            min={0}
            max={100}
            value={Math.round(Number.parseFloat(levelWidth(level)))}
            aria-label={`${name} level`}
          />
        </VisuallyHidden>
        <span className="mic-meter" aria-hidden="true" data-level-key={slug}>
          <b style={{ '--level': levelWidth(level) } as CSSProperties} />
          {threshold != null && <s style={{ left: levelWidth(threshold) }} data-testid="mic-threshold" />}
          <i />
        </span>
        <span className={`mic-state${speech && healthy ? ' speech' : ''}${healthy ? '' : ' bad'}`}>
          {said}
          <VisuallyHidden>{`, ${name}`}</VisuallyHidden>
        </span>
      </div>
    );
  return (
    <div className="mic-chip">
      <IconMicrophoneFilled size={16} stroke={2} aria-hidden="true" />
      <span className="mic-name" title={device.name}>
        {name}
      </span>
      {/* The level for screen readers; the bar beside it moves live for everyone else. */}
      <VisuallyHidden>
        <meter
          min={0}
          max={100}
          value={Math.round(Number.parseFloat(levelWidth(level)))}
          aria-label={`${name} level`}
        />
      </VisuallyHidden>
      <span
        className="mic-meter"
        aria-hidden="true"
        data-level-key={slug}
        title={threshold != null ? `Speech has to pass the mark (${threshold.toFixed(0)} dBFS)` : undefined}
      >
        <b style={{ '--level': levelWidth(level) } as CSSProperties} />
        {threshold != null && <s style={{ left: levelWidth(threshold) }} data-testid="mic-threshold" />}
        <i />
      </span>
      {device.gain?.auto && device.gain.db != null && (
        <span className="mic-gain" title={gainText(device.gain)}>
          {`${device.gain.db > 0 ? '+' : ''}${device.gain.db.toFixed(1)} dB`}
        </span>
      )}
      <span className={`mic-state${speech && healthy ? ' speech' : ''}${healthy ? '' : ' bad'}`}>{state}</span>
    </div>
  );
}

/** The listen worker's energy-gate state; a missing flag is unknown rather than silence. */
function useSpeech(slug: string): boolean | undefined {
  const [speech, setSpeech] = useState<boolean>();
  useEffect(() => {
    let stale: ReturnType<typeof setTimeout> | undefined;
    const off = subscribeLevels(levels => {
      clearTimeout(stale);
      setSpeech(levels[slug]?.speech);
      // A connection can stall before EventSource reports an error. Do not keep its last activity state forever.
      stale = setTimeout(() => setSpeech(undefined), 3000);
    });
    return () => {
      clearTimeout(stale);
      off();
    };
  }, [slug]);
  return speech;
}

/** compact: one short line per microphone ("Listening · quiet"), for the conversation's header. */
export function Mics({ compact = false }: { compact?: boolean } = {}) {
  const ambient = useLive('ambient');
  const client = useQueryClient();
  const root = useRef<HTMLElement>(null);
  useLiveLevels(root); // the bars move with the live levels between status updates
  const devices = Object.entries((ambient?.devices ?? {}) as Record<string, AmbientDevice>);
  const picker = client.getQueryData<Devices>(DEVICES_KEY); // names the Sources drawer found, if it has been opened
  const known = new Map((picker?.mics ?? []).map(m => [m.id, m.name]));
  const paused = summary(usePrivacy(), 'microphone').paused;
  // The header line keeps its place while no microphone has reported yet (#2485).
  if (!devices.length && !paused.length)
    return compact ? <section className="mics mics-compact" aria-hidden="true" /> : null;
  return (
    <section className={`mics${compact ? ' mics-compact' : ''}`} aria-label="Microphones" ref={root}>
      {devices.map(([slug, d]) => (
        <Chip key={slug} slug={slug} device={d} name={micName(d.name, known)} compact={compact} />
      ))}
      {paused.map(name => (
        <span key={`paused-${name}`} className="mic-chip mic-paused" role="status">
          <IconMicrophoneOff size={14} aria-hidden="true" />
          {`${name}: Paused for privacy`}
        </span>
      ))}
    </section>
  );
}
