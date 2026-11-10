import { describe, expect, it } from 'vitest';
import {
  ago,
  clipLength,
  dateTime,
  daysWords,
  dayWords,
  gb,
  hm,
  hms,
  isoDay,
  mb,
  minutesText,
  momentWords,
  plural,
  secondsText,
  seenStatus,
} from './format';

const NOW = 1_000_000;

describe('ago', () => {
  it.each([
    [NOW - 12, '12 s ago'],
    [NOW - 59.4, '59 s ago'],
    [NOW - 60, '1 min ago'],
    [NOW - 3599, '60 min ago'],
    [NOW - 9000, '2.5 h ago'],
    [NOW - 86400 * 3, '3 d ago'],
    [NOW + 30, '0 s ago'],
    [0, '-'],
    [null, '-'],
  ])('%s is %s', (ts, text) => {
    expect(ago(ts, NOW)).toBe(text);
  });
});

describe('the rest', () => {
  it('pluralizes', () => {
    expect([plural(1, 'voice', 'voices'), plural(0, 'voice', 'voices'), plural(3, 'face', 'faces')]).toEqual([
      '1 voice',
      '0 voices',
      '3 faces',
    ]);
  });

  it('formats sizes', () => {
    expect([mb(5 * 1048576), gb(1.5 * 1073741824)]).toEqual(['5 MB', '1.5 GB']);
  });

  it('gives the local day, zero-padded', () => {
    expect(isoDay(new Date(2026, 0, 5, 23, 59))).toBe('2026-01-05');
  });
});

describe('seenStatus', () => {
  const fmt = (ts: number) => `at ${ts}`;
  it.each([
    [{ last_seen: 5 }, 'seen at 5'],
    [{ last_seen: null }, ''],
    [{ last_seen: 5, last_seen_pending: true }, 'Face history updating'],
    [{ last_seen: 5, last_seen_error: 'db locked', last_seen_pending: true }, 'Face history unavailable'],
  ])('%j says %s', (person, text) => {
    expect(seenStatus(person, fmt)).toBe(text);
  });
});

describe('hm and hms (#440: one formatter, made once)', () => {
  it('read exactly as toLocaleTimeString does, at every hour of a day and across a year', () => {
    const start = new Date(2026, 0, 1).getTime() / 1000;
    for (let i = 0; i < 400; i++) {
      const ts = start + i * 86_400 * 0.91 + i * 37.3; // spread over the year and through the day, DST included
      const d = new Date(ts * 1000);
      expect(hms(ts)).toBe(
        d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }),
      );
      expect(hm(ts)).toBe(d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }));
    }
  });
});

describe('clipLength', () => {
  it.each([
    [0, '0:00'],
    [54.5, '0:55'],
    [59.4, '0:59'],
    [605, '10:05'],
    [3723, '1:02:03'],
    [-3, '0:00'],
  ])('%d s is %j', (s, text) => {
    expect(clipLength(s)).toBe(text);
  });
});

describe('minutesText', () => {
  it.each([
    [0.2, '12 s'],
    [0.06, '4 s'],
    [18.3, '18 min'],
    [64.61, '1 h 5 min'],
    [120, '2 h'],
    [0, '0 min'],
    [Number.NaN, '0 min'],
  ])('%s minutes reads %s', (minutes, text) => {
    expect(minutesText(minutes)).toBe(text);
  });
});

describe('dayWords and daysWords', () => {
  const now = new Date(2026, 10, 5, 15);
  it('says today and yesterday, and a date for the rest', () => {
    expect(dayWords('2026-11-05', now)).toBe('today');
    expect(dayWords('2026-11-04', now)).toBe('yesterday');
    expect(dayWords('2026-11-02', now)).not.toMatch(/2026|today|yesterday/);
  });
  it('lists days with the Oxford comma', () => {
    expect(daysWords(['2026-11-05'], now)).toBe('today');
    expect(daysWords(['2026-11-04', '2026-11-05'], now)).toBe('yesterday and today');
    expect(daysWords(['2026-11-02', '2026-11-04', '2026-11-05'], now)).toMatch(/^.+, yesterday, and today$/);
  });
});

describe('secondsText', () => {
  it.each([
    [45, '45 s'],
    [59.6, '1 min'],
    [7170, '2 h'], // 1 h 59.5 min: chipStyle used to print "1 h 60 min"
    [3725, '1 h 2 min'],
    [90000, '1 d 1 h'],
    [-3, '0 s'],
  ])('%s s reads %s', (seconds, text) => {
    expect(secondsText(seconds)).toBe(text);
  });
});

describe('dateTime', () => {
  it('says the fallback when there is no time', () => {
    expect(dateTime(null)).toBe('never');
    expect(dateTime(0, 'unknown time')).toBe('unknown time');
  });
});

describe('momentWords (#2674)', () => {
  const now = new Date('2026-11-06T15:00:00');
  const at = (iso: string) => new Date(iso).getTime() / 1000;
  it('says today, yesterday, or the day, then the 24-hour time', () => {
    expect(momentWords(at('2026-11-06T13:31:00'), now)).toBe('Today, 13:31');
    expect(momentWords(at('2026-11-05T09:05:00'), now)).toBe('Yesterday, 09:05');
    expect(momentWords(at('2026-11-04T13:31:00'), now)).toMatch(/^Wed,? 4 Nov, 13:31$|^Wed, Nov 4, 13:31$/);
  });
});
