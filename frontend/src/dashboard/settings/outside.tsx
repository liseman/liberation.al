// What reaches outside this machine, for Settings (#2728, Joe's Settings board ideas 1 and 2): Phone's line saying
// whether calls work, and People and privacy's short list of what leaves. Both read what the page already has: the
// services (is the phone bridge or the GitHub backup set to run), the config (an agent's address), and the imported
// media sources (Google Photos signed in). Nothing here changes anything; a link goes to where it can be.
import { Anchor, Text } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { getIngest, INGEST_KEY } from '../ingest/api';
import { useDraft } from './draft';
import { SettingsRow, SettingsSection } from './layout';
import layout from './layout.module.css';
import { servicesStatusQuery } from './services/api';
import type { Service } from './services/model';
import classes from './settings.module.css';

const RUNNING = new Set(['active', 'running', 'activating', 'reloading']);
const running = (s: Service | undefined): boolean => !!s && RUNNING.has(s.state);

function useServices() {
  return useQuery(servicesStatusQuery()).data?.services;
}

/** Phone: whether calls work, from the phone bridge's state, with the way to turn it on when they don't. */
export function PhoneStatus() {
  const services = useServices();
  const bridge = services?.find(s => s.name === 'phone-bridge');
  let line: ReactNode;
  if (!services) line = 'Checking the phone bridge…';
  else if (running(bridge)) line = 'Calls work: the phone bridge is running.';
  else if (bridge?.state === 'failed')
    line = (
      <>
        Calls are off: the phone bridge stopped.{' '}
        <Anchor href="#services" underline="always">
          See Services
        </Anchor>
      </>
    );
  else
    line = (
      <>
        Calls are off: the phone bridge isn't set to run on this machine.{' '}
        <Anchor href="#services" underline="always">
          Turn it on in Services
        </Anchor>
      </>
    );
  return (
    <SettingsSection>
      <SettingsRow>
        <Text role="status" size="sm">
          {line}
        </Text>
      </SettingsRow>
    </SettingsSection>
  );
}

/** An agent's address in words: one on this machine, or the host it's at. */
function agentWhere(url: string): { local: boolean; host: string } | null {
  try {
    const u = new URL(url);
    return { local: ['127.0.0.1', 'localhost', '::1', '[::1]'].includes(u.hostname), host: u.host };
  } catch {
    return null;
  }
}

function Item({ on, label, children }: { on: boolean; label: string; children: ReactNode }) {
  return (
    <SettingsRow className={classes.leaves}>
      <div className={layout.words}>
        <span className={layout.label}>{label}</span>
        <Text size="sm" c="dimmed">
          {children}
        </Text>
      </div>
      <Text size="sm" fw={on ? 600 : undefined} c={on ? undefined : 'dimmed'}>
        {on ? 'On' : 'Off'}
      </Text>
    </SettingsRow>
  );
}

/** People and privacy: what leaves this machine, each on or off, honest and short; everything else stays here. */
export function WhatLeaves() {
  const d = useDraft();
  const services = useServices();
  const ingest = useQuery({
    queryKey: INGEST_KEY,
    queryFn: ({ signal }) => getIngest(signal),
    refetchInterval: 30_000,
  });
  const of = (name: string) => services?.find(s => s.name === name);
  const phone = of('phone-bridge');
  const agent = agentWhere(String(d.value('agent.url') ?? '').trim());
  const google = ingest.data?.sources.find(s => s.name === 'google_photos');
  return (
    <SettingsSection title="What leaves this machine">
      <Item label="Phone calls" on={!!phone?.selected}>
        Calls go through Twilio: the other person's voice and the assistant's replies.
      </Item>
      <Item label="Agent" on={!!agent && !agent.local}>
        {agent && !agent.local
          ? `Requests the local model can't handle go to the agent at ${agent.host}.`
          : agent
            ? "Requests the local model can't handle go to an agent on this machine, so they stay here."
            : "No agent is set: requests the local model can't handle aren't sent anywhere."}
      </Item>
      <Item label="Google Photos" on={!!google?.configured}>
        A browser on this machine stays signed in to your Google account to bring your photos in.
      </Item>
      <SettingsRow>
        <Text size="sm">
          Everything else stays on this machine: recordings, faces, voices, transcripts, and the models that read them.
        </Text>
      </SettingsRow>
    </SettingsSection>
  );
}
