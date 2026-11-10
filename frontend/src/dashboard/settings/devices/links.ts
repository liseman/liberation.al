// Settings' phones, watches, and tags (#1363): the devices the Bluetooth sensors and the router saw, and the owner's
// links from a device to a person or a thing (GET and POST /device-links, dashboard/device_links.py).
import { request } from '../../../shared/http';

export const DEVICE_LINKS_KEY = ['device-links'] as const;

export type SelectorKind = 'ble_identifier' | 'ble_fingerprint' | 'ble_name' | 'net_mac';
export type Target = 'person' | 'thing';

export interface Selector {
  kind: SelectorKind;
  value: string;
}

export interface SeenDevice {
  source: string;
  identifier: string;
  name?: string | null;
  vendor?: string | null; // who made it, from the IEEE and Bluetooth SIG registries (proximity/vendors.py)
  kind?: string | null; // what it is, when its advertisement says (AirPods or Beats, Heart Rate)
  place: string;
  place_id?: string | null; // a Wi-Fi access point's MAC, to rename it by; null for a Bluetooth sensor
  rssi: number | null;
  last_seen: number;
  sightings: number;
  selectors: Selector[];
  linked: { id: string; target: Target; name: string } | null;
  location?: DeviceLocation | null; // where the owner said it lives (#3274)
}

/** Where a device lives: a place, or a room of its plan (proximity/devices.py's `device.location`). */
export interface DeviceLocation {
  id: string;
  place_id: string;
  place: string; // the place's name
  room: string | null; // null for the place as a whole
}

/** A place a device can live in, with its plan's rooms. */
export interface PlaceOption {
  id: string;
  name: string;
  rooms: string[];
}

export interface DeviceLink extends Selector {
  id: string;
  target: Target;
  name: string;
}

export interface DeviceLinksView {
  devices: SeenDevice[];
  links: DeviceLink[];
  total: number;
  linked: number;
  unknown: number;
  hours: number;
  locations?: (DeviceLocation & Selector)[];
  places?: PlaceOption[];
}

export const getDeviceLinks = (signal?: AbortSignal) => request<DeviceLinksView>('/device-links', undefined, signal);
export const saveDeviceLink = (selector: Selector, target: Target, name: string) =>
  request<{ ok: true; link: DeviceLink }>('/device-links', { ...selector, target, name: name.trim() });
export const removeDeviceLink = (id: string) => request<{ ok: true }>('/device-links/remove', { id });
export const saveDeviceLocation = (selector: Selector, place: string, room: string) =>
  request<{ ok: true; location: DeviceLocation }>('/device-locations', { ...selector, place, room });
export const removeDeviceLocation = (id: string) => request<{ ok: true }>('/device-locations/remove', { id });

/** Where a device lives in words: "Living room, Home", or "Home" for the whole place. */
export const locationWords = (where: DeviceLocation): string =>
  where.room ? `${where.room}, ${where.place}` : where.place;

/** How each kind of sensor is named on the page. */
export const SOURCE_LABEL: Record<string, string> = {
  'bluetooth.sighting': 'Bluetooth',
  'network.device_observation': 'Network',
};

/** What linking by each selector means, for the "Link by" choice. */
export const SELECTOR_LABEL: Record<SelectorKind, string> = {
  ble_identifier: 'Its Bluetooth identifier',
  ble_fingerprint: 'What it broadcasts (for an address that changes)',
  ble_name: 'Its Bluetooth name',
  net_mac: 'Its network address',
};

type Named = Pick<SeenDevice, 'name' | 'identifier' | 'vendor' | 'kind'>;

/** A device's name for the page: what it calls itself, else what kind it is, else who made it, else "Unknown
 *  device". Never its address (#3267): that's in the card's details. */
export const deviceName = (d: Named): string =>
  d.name?.trim() || d.kind || (d.vendor ? `${d.vendor} device` : 'Unknown device');

/** What a card says under the name: the maker and the kind, leaving out what the name already says. */
export function aboutDevice(d: Named): string {
  const name = deviceName(d);
  return [d.vendor && name !== `${d.vendor} device` ? d.vendor : null, d.kind && name !== d.kind ? d.kind : null]
    .filter(Boolean)
    .join(' · ');
}

/** A sensor name that's only a chip id or an address ("AABBCCDDEE01"): the card leaves it to the details. */
export const rawId = (text: string): boolean => /^(?:[0-9a-f]{12}|(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2})$/i.test(text);

/** One label per device, the same wherever it shows (Known › Devices): its name, and when other devices share that
 *  name ("iPhone"), the end of its address too ("iPhone (C7:EB)"). `among` is every device the page knows of. */
export function deviceLabels(among: readonly Named[]): (d: Named) => string {
  const count = new Map<string, number>();
  for (const d of among) count.set(deviceName(d), (count.get(deviceName(d)) ?? 0) + 1);
  return d => ((count.get(deviceName(d)) ?? 0) > 1 ? `${deviceName(d)} (${d.identifier.slice(-5)})` : deviceName(d));
}

/** `deviceLabels` over several lists that may name the same device (the link listing, today's Wi-Fi), each device
 *  counted once by its address, the first list's name winning. Both Known › Devices cards call it with the same lists,
 *  so a device reads the same in each. */
export function labelsOver(...lists: readonly (readonly Named[])[]): (d: Named) => string {
  const once = new Map<string, Named>();
  for (const list of lists)
    for (const d of list) {
      const key = d.identifier.toUpperCase();
      if (!once.has(key)) once.set(key, d);
    }
  const label = deviceLabels([...once.values()]);
  return d => {
    const kept = once.get(d.identifier.toUpperCase());
    return label({ ...d, name: kept?.name ?? d.name, vendor: d.vendor ?? kept?.vendor, kind: d.kind ?? kept?.kind });
  };
}

/** A name that sounds like something carried (a phone, a watch, a tablet): a likelier device to link to a person. */
export const CARRIED = /\b(iphone|ipad|galaxy|pixel|watch|phone|android|oneplus|moto|xiaomi|redmi|huawei)\b/i;

/** Unlinked devices in the order they're likeliest to be someone's: one that came and went today, then one heard over
 *  Bluetooth, then one whose name sounds carried, then the rest; ties keep their order. The ones on all day
 *  (`steady`) are a separate, last group, since a plug or a TV is rarely what's being linked. */
export function rankCandidates<T extends SeenDevice>(
  devices: readonly T[],
  moving: ReadonlySet<string>,
  steady: ReadonlySet<string>,
): { likely: T[]; steady: T[] } {
  const id = (d: T) => d.identifier.toUpperCase();
  const score = (d: T) =>
    moving.has(id(d)) ? 0 : d.source.startsWith('bluetooth') ? 1 : CARRIED.test(deviceName(d)) ? 2 : 3;
  const likely = devices
    .filter(d => !steady.has(id(d)))
    .map((d, i) => [d, score(d), i] as const)
    .sort((a, b) => a[1] - b[1] || a[2] - b[2])
    .map(([d]) => d);
  return { likely, steady: devices.filter(d => steady.has(id(d))) };
}

/** API records supply these keys: inherited object properties are never labels. */
export const sourceLabel = (source: string): string =>
  Object.hasOwn(SOURCE_LABEL, source) ? (SOURCE_LABEL[source] ?? source) : source;
export const selectorLabel = (kind: string): string =>
  Object.hasOwn(SELECTOR_LABEL, kind) ? SELECTOR_LABEL[kind as SelectorKind] : kind;

export type SourceFilter = 'all' | 'network' | 'bluetooth';

export const matchesSource = (source: string, filter: SourceFilter): boolean =>
  filter === 'all' ||
  (filter === 'network' && source === 'network.device_observation') ||
  (filter === 'bluetooth' && source === 'bluetooth.sighting');

/** Search complete results before the display limit, including the person explicitly linked by the owner. */
export const matchesDevice = (device: SeenDevice, query: string): boolean =>
  [
    device.name,
    device.identifier,
    device.vendor,
    device.kind,
    device.place,
    device.linked?.name,
    ...device.selectors.map(s => s.value),
  ].some(value => value?.toLocaleLowerCase().includes(query));
