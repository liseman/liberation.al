// Starting a service from the page, and what Live shows before anything is running.
import { Button, Code, Group, Text, Title } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { request } from '../../shared/http';
import { SERVICES_KEY, type ServicesView } from '../settings/services/model';
import { startService } from './api';
import { GettingStarted } from './GettingStarted';

/** What starting the cameras takes on Continuum: capture and its broker, the tier worker that keeps the video, the
 *  gateway Live's views come from, the shared decoder, and the vision workers that read it. */
export const CAMERA_SERVICES = [
  'continuum-broker',
  'continuum-capture',
  'continuum-tier-worker',
  'continuum-gateway',
  'see-decode',
  'see-captions',
  'see-actions',
];

/** A button that turns on `names` (one service, or the several a feature needs), those of them not running yet. */
export function StartButton({
  name,
  label,
  size,
}: {
  name: string | string[];
  label: string;
  size?: string; // a Mantine button size, for a button inside a line of text
}) {
  const client = useQueryClient();
  const view = useQuery({ queryKey: SERVICES_KEY, queryFn: () => request<ServicesView>('/services') }).data;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const names = typeof name === 'string' ? [name] : name;
  const all = names.map(n => view?.services?.find(s => s.name === n)).filter(s => s !== undefined);
  const what = typeof name === 'string' ? name : label.replace(/^Start /, '').toLowerCase();
  if (!view || !all.length) return null;
  if (!view.control)
    return (
      <Text size="sm" c="dimmed">
        Start {what} from the machine Liberational runs on, or in its Services panel.
      </Text>
    );
  const chosen = all.filter(s => s.selected);
  if (!chosen.length)
    return (
      <Text size="sm" c="dimmed">
        {what} isn't enabled for this machine (see <Code>[services]</Code> in <Code>pack.toml</Code>).
      </Text>
    );
  const off = chosen.filter(s => !s.installed && !s.pids.length);
  if (!off.length) return null;
  const go = async () => {
    setBusy(true);
    let failed: string | null = null;
    for (const s of off) failed = failed ?? (await startService(client, s.name));
    setError(failed);
    setBusy(false);
  };
  return (
    <>
      <Button onClick={go} loading={busy} size={size}>
        {label}
      </Button>
      {error && (
        <Text size="sm" c="var(--lib-bad)" role="alert">
          {error}
        </Text>
      )}
    </>
  );
}

export function Setup() {
  return (
    <section className="setup">
      <Title order={1}>Liberational isn't listening yet</Title>
      <GettingStarted />
      <Text>
        Start the microphone service and what people say will show up here, line by line, with who said it. Cameras are
        optional: they add who's in view and a video record for Review.
      </Text>
      <Group className="setup-actions" gap="sm">
        <StartButton name="listen" label="Start listening" />
        <StartButton name={CAMERA_SERVICES} label="Start cameras" />
      </Group>
      <Text size="sm" c="dimmed">
        Services can also be switched under Services, top right.
      </Text>
    </section>
  );
}
