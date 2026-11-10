// Known › Devices' two answers: where each linked thing and person was last seen ("where my
// things are"), and what's on the home Wi-Fi today and when each device came and went. Both read what the router and
// the Bluetooth sensors already recorded; nothing here contacts a device.
import { Anchor, Badge, Group, Loader, Text } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { useQuery } from '@tanstack/react-query';
import { type ReactNode, useState } from 'react';
import { messageOf } from '../../shared/http';
import { ago, hm, plural } from '../lib/format';
import { PageError, PHONE_WIDTH } from '../lib/page';
import { PlaceName } from '../settings/devices/AccessPoint';
import { cardKeys, focusCard, LinkChips, LinkPicker } from '../settings/devices/DeviceActions';
import cards from '../settings/devices/deviceCards.module.css';
import {
  aboutDevice,
  DEVICE_LINKS_KEY,
  getDeviceLinks,
  labelsOver,
  type PlaceOption,
  type SeenDevice,
} from '../settings/devices/links';
import { SettingsRow, SettingsSection } from '../settings/layout';
import classes from './devices.module.css';
import {
  type Comings,
  comingsQuery,
  type HomeStatus,
  presenceQuery,
  type Stay,
  type WifiDevice,
  whereaboutsQuery,
} from './devicesApi';

/** How many Wi-Fi devices show before "Show all". */
export const WIFI_SHOWN = 12;

/** A device's stays today in words: "08:12 to 09:40, 11:05 to now". The last stay is "to now" while the device was
 *  seen within the gap. */
export function staysText(stays: readonly Stay[], at: number, gap: number): string {
  return stays
    .map((s, i) => {
      const here = i === stays.length - 1 && at - s.end <= gap;
      return here
        ? `${hm(s.start)} to now`
        : s.end - s.start < 60
          ? `${hm(s.start)}`
          : `${hm(s.start)} to ${hm(s.end)}`;
    })
    .join(', ');
}

/** Whether a device has been on the Wi-Fi all day: one stay, from the day's first reading to now. These don't come
 *  and go, so they're listed by name only, after the ones that did. */
export function allDay(d: WifiDevice, today: Comings): boolean {
  const [only] = d.stays;
  return (
    d.stays.length === 1 &&
    !!only &&
    only.start - today.start <= today.gap_seconds &&
    today.at - only.end <= today.gap_seconds
  );
}

/** The devices that came or went today, newest arrival first, and those on all day. */
export function splitDay(today: Comings): { moving: WifiDevice[]; steady: WifiDevice[] } {
  const moving = today.devices
    .filter(d => !allDay(d, today))
    .sort((a, b) => (b.stays.at(-1)?.start ?? 0) - (a.stays.at(-1)?.start ?? 0));
  return { moving, steady: today.devices.filter(d => allDay(d, today)) };
}

/** A Wi-Fi device as the link labels take it: its address and the hostname the router reported. */
export const asNamed = (d: WifiDevice) => ({ identifier: d.mac, name: d.name, vendor: d.vendor });

/** How many all-day names show before "and N more". */
export const STEADY_SHOWN = 12;

function Loading({ what }: { what: string }) {
  return (
    <SettingsRow>
      <Group gap="xs" role="status" aria-live="polite">
        <Loader size="xs" />
        <Text size="sm" c="dimmed">
          {what}
        </Text>
      </Group>
    </SettingsRow>
  );
}

/** Where each linked thing and person was last seen. */
export function Whereabouts({ control = false }: { control?: boolean }) {
  const q = useQuery(whereaboutsQuery());
  const data = q.data;
  // People are in Who's home, with whether they're here; this is the things.
  const things = data?.links.filter(w => w.target === 'thing') ?? [];
  return (
    <SettingsSection title="Your things" description="Where each linked thing was last heard or seen, and when.">
      {!data ? (
        q.error ? (
          <SettingsRow>
            <PageError
              title="Couldn't find your things"
              detail={messageOf(q.error)}
              onRetry={() => void q.refetch()}
              busy={q.isFetching}
            />
          </SettingsRow>
        ) : (
          <Loading what="Looking for your things…" />
        )
      ) : !things.length ? (
        <SettingsRow>
          <Text size="sm" c="dimmed">
            No things linked yet. Link a tag, earbuds, or a laptop below as a thing, and this says where it was last
            seen.
          </Text>
        </SettingsRow>
      ) : (
        <SettingsRow>
          <ul className={cards.grid} aria-label="Your things">
            {things.map(w => (
              <li key={`${w.target}:${w.name}`} className={cards.card}>
                <Text fw={600} size="sm" className={cards.title} title={w.name}>
                  {w.name}
                </Text>
                <Text size="xs" c={w.last_seen ? undefined : 'dimmed'}>
                  {w.last_seen ? (
                    <>
                      Seen {ago(w.last_seen, data.at)} by{' '}
                      <PlaceName place={w.place ?? 'a sensor'} placeId={w.place_id} control={control} />,{' '}
                      {hm(w.last_seen)}
                    </>
                  ) : (
                    `Not seen in ${plural(data.days, 'day', 'days')}`
                  )}
                </Text>
              </li>
            ))}
          </ul>
        </SettingsRow>
      )}
    </SettingsSection>
  );
}

/** A person's home status in words, from their linked devices. */
export const HOME_WORDS: Record<HomeStatus, { label: string; color: string }> = {
  recent: { label: 'Here', color: 'green' },
  not_seen: { label: 'Not seen recently', color: 'gray' },
  unknown: { label: "Can't tell", color: 'yellow' },
};

/** Whether each person with a linked phone or watch is home, by their devices (GET /device-presence). */
export function WhoIsHome() {
  const q = useQuery(presenceQuery());
  const data = q.data;
  return (
    <SettingsSection title="Who's home" description="By the phones and watches linked to each person.">
      {!data ? (
        q.error ? (
          <SettingsRow>
            <PageError
              title="Couldn't tell who's home"
              detail={messageOf(q.error)}
              onRetry={() => void q.refetch()}
              busy={q.isFetching}
            />
          </SettingsRow>
        ) : (
          <Loading what="Checking who's home…" />
        )
      ) : !data.people.length ? (
        <SettingsRow>
          <Text size="sm" c="dimmed">
            No phones or watches are linked to people yet. Find someone's phone below and link it to them as a person;
            then this says whether they're home.
          </Text>
        </SettingsRow>
      ) : (
        <SettingsRow>
          <ul className={cards.grid} aria-label="Who's home">
            {data.people.map(p => {
              const words = HOME_WORDS[p.status];
              return (
                <li key={p.name} className={cards.card}>
                  <div className={cards.head}>
                    <Text fw={600} size="sm" className={cards.title} title={p.name}>
                      {p.name}
                    </Text>
                    <Badge size="sm" variant="light" color={words.color}>
                      {words.label}
                    </Badge>
                  </div>
                  <Text size="xs" c={p.last_seen ? undefined : 'dimmed'}>
                    {p.last_seen
                      ? `Device seen ${ago(p.last_seen, data.at)} by ${p.place ?? 'a sensor'}, ${hm(p.last_seen)}`
                      : p.status === 'unknown'
                        ? "The sensors that would see their devices weren't running."
                        : "Their devices weren't seen recently."}
                  </Text>
                </li>
              );
            })}
          </ul>
        </SettingsRow>
      )}
    </SettingsSection>
  );
}

/** The hours a day bar marks. */
const TICKS = [6, 12, 18];

/** One device's day as a bar: the whole day, midnight to midnight, with its visits filled in, faint marks at 06, 12,
 *  and 18, and an edge at now (the rest of the day is still to come). */
function DayBar({ stays, start, at, label }: { stays: readonly Stay[]; start: number; at: number; label: string }) {
  const day = 86400;
  const pct = (t: number) => `${(100 * Math.min(day, Math.max(0, t - start))) / day}%`;
  return (
    <div className={classes.bar} role="img" aria-label={label}>
      <span className={classes.later} style={{ left: pct(at) }} />
      {TICKS.map(h => (
        <span key={h} className={classes.tick} style={{ left: pct(start + h * 3600) }} />
      ))}
      {stays.map(s => (
        <span
          key={s.start}
          className={classes.stay}
          style={{ left: pct(s.start), width: `max(3px, ${(100 * Math.max(0, s.end - s.start)) / day}%)` }}
        />
      ))}
      <span className={classes.now} style={{ left: pct(at) }} />
    </div>
  );
}

/** A Wi-Fi device as the link form takes it: recognized by its network address. */
const asSeen = (d: WifiDevice): SeenDevice => ({
  source: 'network.device_observation',
  identifier: d.mac,
  name: d.name,
  vendor: d.vendor,
  place: d.place ?? '',
  rssi: null,
  last_seen: d.stays.at(-1)?.end ?? 0,
  sightings: d.stays.length,
  selectors: [{ kind: 'net_mac', value: d.mac }],
  linked: d.linked,
  location: d.location,
});

/** A Wi-Fi card's frame: focusable, so L opens its picker; its name with the Link button beside it (asking "Whose is
 *  it?" on the ones that came and went, while unlinked); and its link and where it lives as chips at its foot. */
function WifiCard({
  d,
  name,
  control,
  places,
  ask,
  children,
}: {
  d: WifiDevice;
  name: string;
  control: boolean;
  places: readonly PlaceOption[];
  ask: boolean;
  children: ReactNode;
}) {
  const [picking, setPicking] = useState(false);
  const canLink = control && !(d.linked && d.location);
  const about = aboutDevice(asNamed(d));
  return (
    <li
      className={cards.card}
      tabIndex={-1}
      aria-label={name}
      onKeyDown={cardKeys(canLink ? () => setPicking(true) : null)}
      onClick={focusCard}
    >
      <div className={cards.head}>
        <Text fw={600} size="sm" className={cards.title} title={name}>
          {name}
        </Text>
        {canLink && (
          <LinkPicker
            device={asSeen(d)}
            label={name}
            name={ask && !d.linked ? `Whose is ${name}?` : `Link ${name}`}
            places={places}
            opened={picking}
            setOpened={setPicking}
          />
        )}
      </div>
      {about && (
        <Text size="xs" c="dimmed" className={cards.name}>
          {about}
        </Text>
      )}
      {children}
      <LinkChips label={name} linked={d.linked} location={d.location} control={control} />
    </li>
  );
}

function WifiRow({
  d,
  today,
  name,
  control,
  places,
}: {
  d: WifiDevice;
  today: Comings;
  name: string;
  control: boolean;
  places: readonly PlaceOption[];
}) {
  const when = staysText(d.stays, today.at, today.gap_seconds);
  return (
    <WifiCard d={d} name={name} control={control} places={places} ask>
      <Text size="xs">{when}</Text>
      <DayBar stays={d.stays} start={today.start} at={today.at} label={`${name} on the Wi-Fi: ${when}`} />
      {d.place && (
        <Text size="xs" c="dimmed" className={cards.name}>
          via <PlaceName place={d.place} placeId={d.place_id} control={control} />
        </Text>
      )}
    </WifiCard>
  );
}

/** A device on all day, once the list is opened: its name, where, and its link. No day bar: it would be full. */
function SteadyRow({
  d,
  name,
  control,
  places,
}: {
  d: WifiDevice;
  name: string;
  control: boolean;
  places: readonly PlaceOption[];
}) {
  return (
    <WifiCard d={d} name={name} control={control} places={places} ask={false}>
      <Text size="xs" c="dimmed" className={cards.name}>
        {d.place ? (
          <>
            via <PlaceName place={d.place} placeId={d.place_id} control={control} />
          </>
        ) : (
          'on all day'
        )}
      </Text>
    </WifiCard>
  );
}

/** What's on the home Wi-Fi today, and when each device came and went. As the `starter`, while no person has a
 *  linked phone, it leads the tab and asks whose these are: the ones that came and went are probably phones. */
export function WifiToday({ control = false, starter = false }: { control?: boolean; starter?: boolean }) {
  const q = useQuery(comingsQuery());
  const links = useQuery({ queryKey: DEVICE_LINKS_KEY, queryFn: ({ signal }) => getDeviceLinks(signal) });
  const phone = useMediaQuery(PHONE_WIDTH, undefined, { getInitialValueInEffect: false }) ?? false;
  const [all, setAll] = useState(false);
  const [allSteady, setAllSteady] = useState(false);
  const data = q.data;
  const { moving, steady } = data ? splitDay(data) : { moving: [], steady: [] };
  // One label per device, the same as the link card's (it's given the same two lists).
  const labelOf = labelsOver(links.data?.devices ?? [], (data?.devices ?? []).map(asNamed));
  const nameOf = (d: WifiDevice) => labelOf(asNamed(d));
  const places = links.data?.places ?? [];
  const shown = all ? moving : moving.slice(0, WIFI_SHOWN);
  const steadyNames = steady.map(nameOf);
  return (
    <SettingsSection
      title={starter ? 'Whose phones are these?' : 'On the Wi-Fi today'}
      description={
        starter
          ? "These came and went on the Wi-Fi today, so they're probably phones. Say whose each one is, and this tab shows who's home."
          : 'What came and went since midnight, newest arrival first, by what the router said was connected. Ten minutes away starts a new visit.'
      }
    >
      {!data ? (
        q.error ? (
          <SettingsRow>
            <PageError
              title="Couldn't read the Wi-Fi"
              detail={messageOf(q.error)}
              onRetry={() => void q.refetch()}
              busy={q.isFetching}
            />
          </SettingsRow>
        ) : (
          <Loading what="Reading today's Wi-Fi…" />
        )
      ) : !data.devices.length ? (
        <SettingsRow>
          <Text size="sm" c="dimmed">
            {starter
              ? "Nothing on the Wi-Fi yet today to link. Find someone's phone in the list below and link it to them as a person."
              : "Nothing on the Wi-Fi yet today, or the router isn't being read. Its client list comes in under Settings › Devices once it's set up."}
          </Text>
        </SettingsRow>
      ) : (
        <>
          {moving.length > 0 && (
            <SettingsRow>
              <ul className={cards.grid} aria-label="Came and went today">
                {shown.map(d => (
                  <WifiRow key={d.mac} d={d} today={data} name={nameOf(d)} control={control} places={places} />
                ))}
              </ul>
            </SettingsRow>
          )}
          {moving.length > WIFI_SHOWN && (
            <SettingsRow>
              <Anchor component="button" type="button" size="sm" onClick={() => setAll(v => !v)}>
                {all ? 'Show fewer' : `Show all ${moving.length}`}
              </Anchor>
            </SettingsRow>
          )}
          {steady.length > 0 &&
            (allSteady ? (
              <>
                <SettingsRow>
                  <Text size="sm">
                    <b>{plural(steady.length, 'device', 'devices')} on all day</b>.{' '}
                    <Anchor component="button" type="button" size="sm" onClick={() => setAllSteady(false)}>
                      Show fewer
                    </Anchor>
                  </Text>
                </SettingsRow>
                <SettingsRow>
                  <ul className={cards.grid} aria-label="On all day">
                    {steady.map(d => (
                      <SteadyRow key={d.mac} d={d} name={labelOf(asNamed(d))} control={control} places={places} />
                    ))}
                  </ul>
                </SettingsRow>
              </>
            ) : (
              <SettingsRow>
                <Text size="sm">
                  <b>{plural(steady.length, 'device', 'devices')} on all day</b>
                  {/* On a phone the names wait behind a tap; elsewhere the first few show. Opened, each is a row
                      with its link. */}
                  {phone ? '. ' : `: ${steadyNames.slice(0, STEADY_SHOWN).join(', ')} `}
                  <Anchor component="button" type="button" size="sm" onClick={() => setAllSteady(true)}>
                    {phone
                      ? 'Show them'
                      : steady.length > STEADY_SHOWN
                        ? `and ${steady.length - STEADY_SHOWN} more`
                        : control
                          ? 'Link one'
                          : 'Show them'}
                  </Anchor>
                </Text>
              </SettingsRow>
            ))}
        </>
      )}
    </SettingsSection>
  );
}
