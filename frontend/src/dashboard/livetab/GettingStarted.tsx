// Live's "Getting started" (L5, #296): the six things a fresh install needs, each with its state and, while open, a
// button to the control that does it. A progress bar and a tick per finished step; the checklist goes away once all
// six are done, and "Hide this for now" puts it away until a step that was done opens again.
import { Button, Progress } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { type MouseEventHandler, useEffect, useRef, useState } from 'react';
import { request } from '../../shared/http';
import { setPref, usePrefs } from '../lib/prefs';
import { getPeople, PEOPLE_KEY } from '../people/api';
import { viewName } from '../review/views';
import { href, linkClick } from '../router';
import { CONFIG_KEY, type ConfigView, VOICE_REF_KEY } from '../settings/model';
import { DEVICES_KEY, type Devices, getDevices, getStats, STATS_KEY, type Stats } from '../sources/api';
import { WHOS_THIS } from '../whosthis/labels';
import classes from './gettingStarted.module.css';

const POLL_MS = 30_000; // while the card shows: a step done elsewhere (Settings, a terminal) ticks within this

/** One step: done or not, what it says, and where to go to do it. */
export interface Step {
  id: 'packages' | 'models' | 'sources' | 'name' | 'voice' | 'face';
  title: string;
  note: string;
  done: boolean;
  action: string;
  go: { href: string } | { open: () => void };
}

/** What the steps are made from, as each read answers (undefined while it loads). */
export interface Facts {
  stats?: Stats;
  devices?: Devices;
  owner?: string; // owner.name, '' when unset
  voiceRef?: boolean;
  ownerFaces?: number; // face samples of the person who is the owner
}

const list = (names: string[]): string => names.join(', ');

/** The six steps, in the design's order. Pure, so each step's rule is tested on its own. whosThis: where Tag your
 *  face goes, Live's own Who's this? when it's on the page, else People's. */
export function steps(f: Facts, whosThis: 'here' | 'people' = 'people'): Step[] {
  const sys = f.stats?.system;
  const needed = (sys?.missing ?? []).filter(m => !m.optional).map(m => m.name);
  const ollama = f.stats?.ollama;
  const missingModels = ollama?.missing ?? [];
  const cam = f.devices?.cameras.find(c => c.selected && c.present);
  const mic = f.devices?.mics.find(m => m.selected && m.present);
  const machine = { href: `${href('/settings')}#machine` }; // Settings' This machine (#386, #787)
  return [
    {
      id: 'packages',
      title: 'System packages installed',
      done: !!sys && !sys.error && needed.length === 0,
      note: !sys
        ? "Couldn't check yet"
        : sys.error
          ? `Couldn't check: ${sys.error}`
          : needed.length
            ? `Missing: ${list(needed)}`
            : 'sox, espeak-ng, and the rest are present',
      action: 'Open This machine',
      go: machine,
    },
    {
      id: 'models',
      title: 'Models downloaded',
      done: !!ollama?.up && missingModels.length === 0,
      note: !ollama?.up
        ? ollama?.message || "Ollama isn't answering yet"
        : missingModels.length
          ? `Missing: ${list(missingModels)}`
          : 'The image and embedding models are ready',
      action: 'Check models',
      go: machine,
    },
    {
      id: 'sources',
      title: 'Pick a camera and a microphone',
      done: !!cam && !!mic,
      note:
        cam && mic
          ? `${viewName(`cam${cam.index}`)} and ${mic.name} are on`
          : cam
            ? 'A camera is on; pick a microphone'
            : mic
              ? `${mic.name} is on; pick a camera`
              : 'Choose which camera and microphone Liberational uses',
      action: 'Choose them',
      go: { href: `${href('/settings')}#devices` }, // Settings' Devices section (#384)
    },
    {
      id: 'name',
      title: 'Name yourself',
      done: !!f.owner?.trim(),
      note: f.owner?.trim() ? `You're ${f.owner.trim()}` : 'So the assistant knows who the owner is',
      action: 'Set your name',
      go: { href: `${href('/settings')}#set-owner.name` },
    },
    {
      id: 'voice',
      title: 'Record your voice reference',
      done: !!f.voiceRef,
      note: f.voiceRef ? 'Your voice reference is saved' : 'About 15 seconds of you talking',
      action: 'Record now',
      go: { href: `${href('/settings')}#set-owner.voice_ref` },
    },
    {
      id: 'face',
      title: 'Tag your face',
      done: (f.ownerFaces ?? 0) > 0,
      note: (f.ownerFaces ?? 0) > 0 ? 'Liberational knows your face' : 'Pick yourself in a camera view',
      action: 'Open Who’s this?',
      go: { href: whosThis === 'here' ? `${href('/')}#${WHOS_THIS}` : href('/people') },
    },
  ];
}

/** The open steps, as Hide remembers them: "models,voice". */
const openKey = (s: readonly Step[]): string =>
  s
    .filter(x => !x.done)
    .map(x => x.id)
    .sort()
    .join(',');

/** Whether Hide still applies: it does while no step is open that wasn't when it was pressed. */
export function stillHidden(hidden: string | undefined, now: readonly Step[]): boolean {
  if (hidden == null) return false;
  const then = new Set(hidden.split(',').filter(Boolean));
  return now.every(s => s.done || then.has(s.id));
}

function useFacts(poll: boolean): { facts: Facts; ready: boolean; readAt: number } {
  const every = poll ? POLL_MS : false;
  const stats = useQuery({
    queryKey: STATS_KEY,
    queryFn: ({ signal }) => getStats({ signal }),
    refetchInterval: every,
  });
  // Never polled or refetched on focus: probing devices is costly, and the Sources drop-down keeps unsaved picks only
  // while its copy is unchanged. A save there updates this cache itself.
  const devices = useQuery({
    queryKey: DEVICES_KEY,
    queryFn: ({ signal }) => getDevices({ signal }),
    refetchOnWindowFocus: false,
  });
  const config = useQuery({
    queryKey: CONFIG_KEY,
    queryFn: ({ signal }) => request<ConfigView>('/config', undefined, signal),
  });
  const voice = useQuery({
    queryKey: VOICE_REF_KEY,
    queryFn: ({ signal }) => request<{ exists: boolean }>('/config/voice_ref', undefined, signal),
  });
  const people = useQuery({ queryKey: PEOPLE_KEY, queryFn: getPeople, refetchInterval: every });
  const owner = config.data?.sections.flatMap(s => s.settings).find(r => r.key === 'owner.name')?.value;
  const facts: Facts = {
    stats: stats.data,
    devices: devices.data,
    owner: typeof owner === 'string' ? owner : '',
    voiceRef: voice.data?.exists,
    ownerFaces: people.data?.people.find(p => p.owner)?.faces ?? 0,
  };
  // Shown once every read has answered (or failed), so the list doesn't flash steps as not done while they load.
  const reads = [stats, devices, config, voice, people];
  const ready = reads.every(q => !q.isPending);
  return { facts, ready, readAt: Math.max(...reads.map(q => q.dataUpdatedAt)) };
}

export function GettingStarted({
  whosThis = 'people',
  onNameFace,
}: {
  whosThis?: 'here' | 'people';
  onNameFace?: MouseEventHandler<HTMLAnchorElement>;
} = {}) {
  const hidden = usePrefs(s => s.gettingStartedHidden);
  const finished = usePrefs(s => s.gettingStartedDone === true);
  const [poll, setPoll] = useState(false);
  const { facts, ready, readAt } = useFacts(poll && !finished);
  const all = steps(facts, whosThis);
  const done = all.filter(s => s.done).length;
  const key = openKey(all);
  // After Hide, a newly open step brings the card back only once two reads agree, so one slow answer (Ollama timing
  // out, a camera blinking) doesn't.
  const seen = useRef<{ key: string; at: number } | null>(null);
  const comeBack = hidden != null && !stillHidden(hidden, all);
  const confirmed = comeBack && seen.current?.key === key && seen.current.at !== readAt;
  const showing = ready && !finished && done < all.length && (hidden == null || confirmed);
  useEffect(() => {
    if (comeBack && seen.current?.key !== key) seen.current = { key, at: readAt };
    if (!comeBack) seen.current = null;
  }, [comeBack, key, readAt]);
  useEffect(() => {
    if (ready && done === all.length) setPref('gettingStartedDone', true); // all six once: setup is over for good
  }, [ready, done, all.length]);
  // Polls only while the card shows, or while a hidden one waits to confirm a step that opened.
  useEffect(() => setPoll(showing || comeBack), [showing, comeBack]);
  if (!showing) return null;
  const hide = () => {
    setPref('gettingStartedHidden', key);
    // The button goes with the card: leave focus on the page's own heading rather than nowhere.
    const h1 = document.querySelector<HTMLElement>('h1');
    if (h1) {
      h1.tabIndex = -1;
      h1.focus();
    }
  };
  const next = all.findIndex(s => !s.done);
  return (
    <section className={classes.card} aria-labelledby="getting-started-h">
      <div className={classes.head}>
        <h2 id="getting-started-h" className={classes.title}>
          Getting started
        </h2>
        <p className={classes.sub}>{`${done} of ${all.length} done. Live fills in as each one is ready.`}</p>
      </div>
      {/* Steps, not percent: the bar says "5 of 6", as the line above it does. */}
      <Progress.Root
        size={8}
        bg="var(--lib-bg2)"
        role="progressbar"
        aria-label="Setup progress"
        aria-valuemin={0}
        aria-valuemax={all.length}
        aria-valuenow={done}
      >
        <Progress.Section value={Math.round((done / all.length) * 100)} color="var(--lib-ok)" withAria={false} />
      </Progress.Root>
      {/* Only the next step to do is a Primary; the rest wait their turn as Default buttons (#2594). */}
      <ol className={classes.steps}>
        {all.map((s, i) => (
          <li
            key={s.id}
            className={s.done ? classes.done : undefined}
            aria-label={`${s.title}: ${s.done ? 'done' : 'to do'}`}
          >
            <span className={classes.dot} aria-hidden="true">
              {s.done && (
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  aria-hidden="true"
                >
                  <path d="M5 12l5 5 9-10" />
                </svg>
              )}
            </span>
            <span className={classes.text}>
              <span className={classes.stepTitle}>
                {s.title}
                <span className={classes.srOnly}>{s.done ? ' (done)' : ' (to do)'}</span>
              </span>
              <span className={classes.note}>{s.note}</span>
            </span>
            {!s.done &&
              ('href' in s.go ? (
                <Button
                  component="a"
                  variant={i === next ? 'filled' : 'default'}
                  href={s.go.href}
                  onClick={s.id === 'face' && onNameFace ? onNameFace : linkClick}
                >
                  {s.action}
                </Button>
              ) : (
                <Button variant={i === next ? 'filled' : 'default'} onClick={s.go.open}>
                  {s.action}
                </Button>
              ))}
          </li>
        ))}
      </ol>
      <Button variant="default" className={classes.hide} onClick={hide}>
        Hide this for now
      </Button>
    </section>
  );
}
