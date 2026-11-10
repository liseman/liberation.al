import { expect, it } from 'vitest';
import { labelsOver, rankCandidates, type SeenDevice } from './links';

const seen = (identifier: string, name: string | null, source = 'network.device_observation'): SeenDevice => ({
  source,
  identifier,
  name,
  place: 'Hall',
  rssi: null,
  last_seen: 0,
  sightings: 1,
  selectors: [],
  linked: null,
});

it('ranks what came and went first, then Bluetooth, then a carried name, and keeps the all-day ones apart', () => {
  const plug = seen('aa:00:00:00:00:01', 'plug');
  const tv = seen('aa:00:00:00:00:02', 'tv');
  const pixel = seen('aa:00:00:00:00:03', 'Pixel 9');
  const tag = seen('c0:ff:ee:00:00:01', null, 'bluetooth.sighting');
  const moved = seen('aa:00:00:00:00:04', 'laptop');
  const { likely, steady } = rankCandidates(
    [plug, tv, pixel, tag, moved],
    new Set(['AA:00:00:00:00:04']),
    new Set(['AA:00:00:00:00:02']),
  );
  expect(likely).toEqual([moved, tag, pixel, plug]);
  expect(steady).toEqual([tv]);
});

it('labels a device once across lists, telling a shared name apart by the end of each address', () => {
  const label = labelsOver(
    [seen('aa:00:00:00:00:01', 'iPhone'), seen('aa:00:00:00:00:03', 'plug')],
    // The same phone again in the second list (upper case) is one device, not a second iPhone.
    [
      { identifier: 'AA:00:00:00:00:01', name: 'iPhone' },
      { identifier: 'bb:00:00:00:00:02', name: 'iPhone' },
    ],
  );
  expect(label({ identifier: 'aa:00:00:00:00:01', name: 'iPhone' })).toBe('iPhone (00:01)');
  expect(label({ identifier: 'BB:00:00:00:00:02', name: 'iPhone' })).toBe('iPhone (00:02)');
  expect(label({ identifier: 'aa:00:00:00:00:03', name: 'plug' })).toBe('plug');
  expect(
    labelsOver([seen('aa:00:00:00:00:01', 'iPhone')], [])({ identifier: 'aa:00:00:00:00:01', name: 'iPhone' }),
  ).toBe('iPhone');
});
