// Settings' Phones, watches, and tags card, now Known › Devices' device list: every device the Bluetooth
// sensors heard or the router saw in the last day, known ones first, each a compact card with where and when it was last
// seen. Its Link button opens a picker to link it to a person, name it as a thing (the same links `pack-devices
// link` makes, proximity/devices.py), or say where it lives (a place or a room of its plan); chips show each, with a
// quick remove (DeviceActions.tsx). Links and locations whose device wasn't seen in the window are listed too, so one
// can always be removed. Only this machine changes them.
import { ActionIcon, Button, Group, Loader, NativeSelect, Popover, Text, TextInput } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { IconInfoCircle } from '@tabler/icons-react';
import { useQuery } from '@tanstack/react-query';
import { Fragment, type JSX, useState } from 'react';
import { messageOf } from '../../../shared/http';
import { ago, plural } from '../../lib/format';
import { PageError, PHONE_WIDTH } from '../../lib/page';
import { SettingsRow, SettingsSection } from '../layout';
import { PlaceName } from './AccessPoint';
import { DEVICE_CARD } from './cards';
import { cardKeys, focusCard, LinkChips, LinkPicker } from './DeviceActions';
import cards from './deviceCards.module.css';
import {
  aboutDevice,
  DEVICE_LINKS_KEY,
  type DeviceLink,
  type DeviceLocation,
  getDeviceLinks,
  labelsOver,
  matchesDevice,
  matchesSource,
  type PlaceOption,
  rankCandidates,
  rawId,
  type SeenDevice,
  type Selector,
  type SourceFilter,
  selectorLabel,
  sourceLabel,
} from './links';

/** Unknown devices shown before "Show all": a phone's rotating Bluetooth addresses can add up to hundreds a day. */
export const UNKNOWN_SHOWN = 20;

/** A device's details, behind the card's (i): its address, the sensor that heard it, and how often. Kept off the card
 *  face: a raw id ("AABBCCDDEE01") says nothing at a glance. */
function Details({ device, label }: { device: SeenDevice; label: string }) {
  return (
    <Popover position="bottom-end" withArrow shadow="md">
      <Popover.Target>
        <ActionIcon variant="subtle" color="gray" size="sm" aria-label={`Details for ${label}`}>
          <IconInfoCircle size={16} />
        </ActionIcon>
      </Popover.Target>
      <Popover.Dropdown>
        <dl className={cards.details}>
          <dt>Address</dt>
          <dd>{device.identifier}</dd>
          {device.place && (
            <>
              <dt>Heard by</dt>
              <dd>{device.place}</dd>
            </>
          )}
          <dt>Sightings</dt>
          <dd>{device.sightings}</dd>
        </dl>
      </Popover.Dropdown>
    </Popover>
  );
}

/** One device as a compact card: its name, who made it and what it is, how and when it was last heard, its link and
 *  where it lives as chips, and the Link button with its picker (L on the focused card opens it too). */
function DeviceCard({
  device,
  control,
  label,
  places,
}: {
  device: SeenDevice;
  control: boolean;
  label: string;
  places: readonly PlaceOption[];
}) {
  const [picking, setPicking] = useState(false);
  const about = aboutDevice(device);
  const canLink = control && device.selectors.length > 0 && !(device.linked && device.location);
  // A Bluetooth sensor named only by its chip id says nothing on the card; a network one is the access point, which
  // the owner can name in place.
  const place =
    device.place && !(device.source === 'bluetooth.sighting' && rawId(device.place)) ? (
      <PlaceName key="place" place={device.place} placeId={device.place_id} control={control} />
    ) : null;
  const where = [
    sourceLabel(device.source),
    place,
    device.rssi === null ? null : `${device.rssi} dBm`,
    ago(device.last_seen),
  ].filter(Boolean);
  return (
    <li
      className={cards.card}
      tabIndex={-1}
      aria-label={label}
      onKeyDown={cardKeys(canLink ? () => setPicking(true) : null)}
      onClick={focusCard}
    >
      <div className={cards.head}>
        <Text fw={600} size="sm" className={cards.title} title={label}>
          {label}
        </Text>
        <Group gap={4} wrap="nowrap" align="flex-start">
          {canLink && (
            <LinkPicker device={device} label={label} places={places} opened={picking} setOpened={setPicking} />
          )}
          <Details device={device} label={label} />
        </Group>
      </div>
      {about && (
        <Text size="xs" c="dimmed" className={cards.name}>
          {about}
        </Text>
      )}
      <Text size="xs" c="dimmed">
        {where.map((part, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: the parts are fixed in place
          <Fragment key={i}>
            {i > 0 && ' · '}
            {part}
          </Fragment>
        ))}
      </Text>
      <LinkChips label={label} linked={device.linked} location={device.location} control={control} />
    </li>
  );
}

/** A link whose device wasn't heard in the window, as a card, so it can still be removed. */
function UnseenLink({ link, control, span }: { link: DeviceLink; control: boolean; span: string }) {
  return (
    <li className={cards.card}>
      <Text fw={600} size="sm" className={cards.title} title={link.value}>
        {link.name}
      </Text>
      <Text size="xs" c="dimmed">
        Not seen in {span} · {selectorLabel(link.kind)}
      </Text>
      <LinkChips label={link.name} linked={link} control={control} />
    </li>
  );
}

/** A location whose device wasn't heard in the window, as a card, so it can still be removed. */
function UnseenLocation({
  where,
  control,
  span,
}: {
  where: DeviceLocation & Selector;
  control: boolean;
  span: string;
}) {
  return (
    <li className={cards.card}>
      <Text fw={600} size="sm" className={cards.title} title={where.value}>
        {where.value}
      </Text>
      <Text size="xs" c="dimmed">
        Not seen in {span} · {selectorLabel(where.kind)}
      </Text>
      <LinkChips label={where.value} linked={null} location={where} control={control} />
    </li>
  );
}

export function DeviceLinks({
  control,
  moving = new Set(),
  steady = new Set(),
  wifi = [],
}: {
  control: boolean;
  /** Addresses (upper case) of the devices that came and went today, and of those on all day: the first are the
   *  likeliest to link, the second are kept behind "Show all" (Known › Devices passes them from its Wi-Fi read). */
  moving?: ReadonlySet<string>;
  steady?: ReadonlySet<string>;
  /** Today's Wi-Fi devices, so a device is labeled the same here as in the Wi-Fi card. */
  wifi?: readonly { identifier: string; name?: string | null }[];
}): JSX.Element {
  const view = useQuery({ queryKey: DEVICE_LINKS_KEY, queryFn: ({ signal }) => getDeviceLinks(signal) });
  const [all, setAll] = useState(false);
  const [search, setSearch] = useState('');
  const [source, setSource] = useState<SourceFilter>('all');
  const query = search.trim().toLocaleLowerCase();
  const filtering = Boolean(query) || source !== 'all';
  const data = view.data;
  const span = data && data.hours === 24 ? 'the last day' : `the last ${data?.hours ?? 24} hours`;
  const matching = data?.devices.filter(d => matchesSource(d.source, source) && matchesDevice(d, query)) ?? [];
  const known = matching.filter(d => d.linked || d.location);
  const unknown = matching.filter(d => !d.linked && !d.location);
  // A linked device filtered out above still counts as seen: never also show its link as unseen.
  const seen = new Set(data?.devices.filter(d => d.linked).map(d => d.linked?.id));
  const unseen =
    data?.links.filter(
      l =>
        !seen.has(l.id) &&
        matchesSource(l.kind === 'net_mac' ? 'network.device_observation' : 'bluetooth.sighting', source) &&
        [l.name, l.value].some(value => value.toLocaleLowerCase().includes(query)),
    ) ?? [];
  const seenWhere = new Set(data?.devices.map(d => d.location?.id).filter(Boolean));
  const unseenWhere =
    data?.locations?.filter(
      l =>
        !seenWhere.has(l.id) &&
        matchesSource(l.kind === 'net_mac' ? 'network.device_observation' : 'bluetooth.sighting', source) &&
        [l.value, l.place, l.room ?? ''].some(value => value.toLocaleLowerCase().includes(query)),
    ) ?? [];
  const places = data?.places ?? [];
  const label = labelsOver(data?.devices ?? [], wifi);
  const ranked = rankCandidates(unknown, moving, steady);
  // On a phone the unknown devices start folded (the page is long there); elsewhere the likeliest show first, and the
  // ones on all day wait behind "Show all".
  const phone = useMediaQuery(PHONE_WIDTH, undefined, { getInitialValueInEffect: false }) ?? false;
  const first = phone ? [] : ranked.likely.slice(0, UNKNOWN_SHOWN);
  const shown = all || filtering ? [...ranked.likely, ...ranked.steady] : first;
  return (
    <SettingsSection
      title="Phones, watches, and tags"
      id={DEVICE_CARD.links}
      tabIndex={-1}
      description="Devices heard over Bluetooth or seen on the network. Link one to a person, so Ask can tell whether they're home, or name it as a thing, so Ask can say where it was last seen."
    >
      {!data ? (
        <SettingsRow>
          {view.error ? (
            <PageError
              title="Couldn't list devices"
              detail={messageOf(view.error)}
              onRetry={() => void view.refetch()}
              busy={view.isFetching}
            />
          ) : (
            <Group gap="xs" role="status" aria-live="polite">
              <Loader size="xs" />
              <Text size="sm" c="dimmed">
                Looking for devices…
              </Text>
            </Group>
          )}
        </SettingsRow>
      ) : (
        <>
          {view.error && (
            <SettingsRow>
              <PageError title="Couldn't refresh devices" detail={messageOf(view.error)}>
                Showing the last list.
              </PageError>
            </SettingsRow>
          )}
          <SettingsRow>
            <Text size="sm">
              {data.total === 0
                ? `No devices seen in ${span}.`
                : `${plural(data.total, 'device', 'devices')} seen in ${span}: ${data.linked} linked, ${data.unknown} unknown.`}
            </Text>
          </SettingsRow>
          <SettingsRow>
            <Group align="flex-end">
              <TextInput
                label="Search devices"
                placeholder="Name, maker, person, or place"
                value={search}
                onChange={e => {
                  setSearch(e.currentTarget.value);
                  setAll(false);
                }}
              />
              <NativeSelect
                label="Device source"
                value={source}
                onChange={e => {
                  setSource(e.currentTarget.value as SourceFilter);
                  setAll(false);
                }}
                data={[
                  { value: 'all', label: 'All' },
                  { value: 'network', label: 'Network' },
                  { value: 'bluetooth', label: 'Bluetooth' },
                ]}
              />
            </Group>
            {filtering && (
              <Text size="sm" c="dimmed" role="status">
                {matching.length + unseen.length === 0
                  ? 'No devices or saved links match these filters.'
                  : `${plural(matching.length, 'matching device', 'matching devices')} seen in ${span}${unseen.length ? `, plus ${plural(unseen.length, 'saved link', 'saved links')} not seen` : ''}.`}
              </Text>
            )}
          </SettingsRow>
          {known.length + unseen.length + unseenWhere.length + shown.length > 0 && (
            <SettingsRow>
              <ul className={cards.grid} aria-label="Devices">
                {known.map(d => (
                  <DeviceCard
                    key={`${d.source} ${d.identifier}`}
                    device={d}
                    control={control}
                    label={label(d)}
                    places={places}
                  />
                ))}
                {unseen.map(l => (
                  <UnseenLink key={l.id} link={l} control={control} span={span} />
                ))}
                {unseenWhere.map(l => (
                  <UnseenLocation key={l.id} where={l} control={control} span={span} />
                ))}
                {shown.map(d => (
                  <DeviceCard
                    key={`${d.source} ${d.identifier}`}
                    device={d}
                    control={control}
                    label={label(d)}
                    places={places}
                  />
                ))}
              </ul>
            </SettingsRow>
          )}
          {unknown.length > shown.length && (
            <SettingsRow>
              <Button size="xs" variant="subtle" onClick={() => setAll(true)}>
                {shown.length ? `Show all ${unknown.length} unknown devices` : `Show ${unknown.length} unknown devices`}
              </Button>
            </SettingsRow>
          )}
        </>
      )}
    </SettingsSection>
  );
}
