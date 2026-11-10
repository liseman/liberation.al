// Known › Devices: the phones, watches, and tags the Bluetooth sensors heard and the router saw,
// each linked to a person or named as a thing. It's the card Settings › Devices had (settings/devices/DeviceLinks.tsx),
// moved here as it was, since linking a device is naming what the house knows, like People and Pets. Above it, who's
// home by their devices, where each linked thing was last seen, and what's on the Wi-Fi today
// (DeviceSections.tsx). Until a person has a linked phone, the Wi-Fi card leads as the starter ("Whose phones are
// these?") in place of the two that would be empty. Whether changes are allowed (only from the machine Liberational runs on) comes from GET /config, as Settings reads it.
import { Text } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { request } from '../../shared/http';
import { PageHeader, PageLoading } from '../lib/page';
import { DeviceLinks } from '../settings/devices/DeviceLinks';
import { CONFIG_KEY, type ConfigView } from '../settings/model';
import { asNamed, splitDay, Whereabouts, WhoIsHome, WifiToday } from './DeviceSections';
import { comingsQuery, presenceQuery, whereaboutsQuery } from './devicesApi';

export function DevicesTab() {
  const config = useQuery({
    queryKey: CONFIG_KEY,
    queryFn: ({ signal }) => request<ConfigView>('/config', undefined, signal),
  });
  const control = config.data?.control ?? false;
  // The three sections show together, once each has answered (or failed): filling in one at a time, each pushed the
  // ones under it down (a load shift). These are the sections' own reads, shared by key, not extra ones. The device
  // list isn't waited for: it's the last card, so it fills in without moving anything, and on a fresh start its first
  // read is the slowest by far (a day of Bluetooth rows).
  const presence = useQuery(presenceQuery());
  const whereabouts = useQuery(whereaboutsQuery());
  const comings = useQuery(comingsQuery());
  const ready = [presence, whereabouts, comings].every(r => r.data !== undefined || r.isError);
  // No person linked yet: Who's home has nothing to say, so the Wi-Fi card asks whose phones these are instead.
  const starter = presence.data?.people.length === 0;
  const things = whereabouts.data?.links.some(l => l.target === 'thing') ?? true;
  const upper = (ds: readonly { mac: string }[]) => new Set(ds.map(d => d.mac.toUpperCase()));
  const day = comings.data ? splitDay(comings.data) : { moving: [], steady: [] };
  const linkCard = (
    <DeviceLinks
      control={control}
      moving={upper(day.moving)}
      steady={upper(day.steady)}
      wifi={(comings.data?.devices ?? []).map(asNamed)}
    />
  );
  return (
    <section aria-labelledby="devices-h">
      <PageHeader
        title="Devices"
        titleId="devices-h"
        lede="Who's home, where your things are, what's on the Wi-Fi today, and the phones, watches, and tags nearby to link. A device seen is a clue, not proof of who carried it."
      />
      {ready ? (
        starter ? (
          <>
            <WifiToday starter control={control} />
            {things && <Whereabouts control={control} />}
            {linkCard}
          </>
        ) : (
          <>
            <WhoIsHome />
            <Whereabouts control={control} />
            <WifiToday control={control} />
            {linkCard}
          </>
        )
      ) : (
        <PageLoading label="devices" shape="rows" lines={6} />
      )}
      {config.data && !control && (
        <Text size="sm" c="dimmed" mt="sm">
          Read only. Devices can only be linked from the machine Liberational runs on.
        </Text>
      )}
    </section>
  );
}
