import { describe, expect, it, vi } from 'vitest';
import { HttpError } from '../../../shared/http';
import type { Person } from '../api';
import { dayTime, failure, personMeta, refusal, removedText, roleLine, shortWhen, whereabouts } from './role';

const person = (over: Partial<Person> = {}): Person => ({ name: 'Dana', voices: [], ...over });

describe('roleLine', () => {
  it.each([
    [{ role: 'CEO', organization: 'Acme' }, 'CEO at Acme'],
    [{ role: 'CEO' }, 'CEO'],
    [{ role: 'CEO', organization: '' }, 'CEO'],
    [{ organization: 'Acme' }, 'at Acme'],
    [{ role: null, organization: 'Acme' }, 'at Acme'],
    [{}, ''],
    [{ role: '', organization: null }, ''],
  ])('%o is %j', (p, text) => {
    expect(roleLine(p)).toBe(text);
  });
});

describe('personMeta', () => {
  it('leads with the role, then the lines heard, then the face, the face history, and the phone', () => {
    const p = person({
      role: 'CTO',
      organization: 'Acme',
      lines: 3,
      voices: [1],
      faces: 2,
      last_seen_pending: true,
      number: '+14155550123',
    });
    expect(personMeta(p)).toBe('CTO at Acme, 3 lines heard, face known, Face history updating, phone +14155550123');
  });

  it('says one line in the singular', () => {
    expect(personMeta(person({ lines: 1 }))).toBe('1 line heard');
  });

  it.each([
    [{ enrolled: true, voices: [4] }, 'Voice learned from a recording'],
    [{ voices: [4] }, 'Voice known'],
    [{ faces: 3 }, 'Face known'],
    [{}, 'Nothing heard yet'],
  ])('with no lines, %o says %j', (over, text) => {
    expect(personMeta(person(over))).toBe(text);
  });

  it('adds "face known" only when a voice or lines come first', () => {
    expect(personMeta(person({ voices: [4], faces: 1 }))).toBe('Voice known, face known');
    expect(personMeta(person({ faces: 1 }))).toBe('Face known');
  });

  it('says when face history cannot be read', () => {
    expect(personMeta(person({ last_seen_error: 'boom', last_seen: 100 }))).toBe(
      'Nothing heard yet, Face history unavailable',
    );
  });
});

describe('removedText', () => {
  it('lists only what was removed', () => {
    expect(removedText('Dana', { voices: 2, faces: 3, photos: 4, sightings: 5 })).toBe(
      'Forgot Dana: 2 voices, 3 face samples, 4 photos, 5 recorded sightings.',
    );
    expect(removedText('Dana', { voices: 1, faces: 0, sightings: 1 })).toBe(
      'Forgot Dana: 1 voice, 1 recorded sighting.',
    );
    expect(removedText('Dana', {})).toBe('Forgot Dana.');
  });
});

describe('failure and refusal', () => {
  it("shows the server's message, or the fallback when it gave none", () => {
    expect(failure(new HttpError('no such person', 404), 'That failed')).toBe('no such person');
    expect(failure(new HttpError(null, 500), 'That failed')).toBe('That failed');
    expect(failure(new TypeError('Failed to fetch'), 'That failed')).toBe('Failed to fetch');
  });

  it('reads the error a refusing reply carries', () => {
    expect(refusal({ ok: false, error: 'taken' }, 'That failed')).toBe('taken');
    expect(refusal({ ok: false }, 'That failed')).toBe('That failed');
    expect(refusal(null, 'That failed')).toBe('That failed');
  });
});

/** Unix seconds for a local date and time. */
const at = (y: number, mo: number, d: number, h: number, mi: number) => new Date(y, mo - 1, d, h, mi).getTime() / 1000;
const NOW = new Date(2026, 9, 28, 21, 5); // Oct 28, 2026, 21:05 local

describe('dayTime', () => {
  it.each([
    [at(2026, 10, 28, 18, 42), 'today at 18:42'],
    [at(2026, 10, 28, 0, 0), 'today at 00:00'],
    [at(2026, 10, 27, 23, 59), 'yesterday at 23:59'],
    [at(2026, 10, 27, 20, 12), 'yesterday at 20:12'],
  ])('%d is %j', (ts, text) => {
    expect(dayTime(ts, NOW)).toBe(text);
  });

  it('names the date two or more days back, and the year only when it is another year', () => {
    expect(dayTime(at(2026, 10, 26, 9, 10), NOW)).toMatch(/^Oct 26 at 09:10$/);
    expect(dayTime(at(2025, 12, 31, 9, 10), NOW)).toMatch(/2025 at 09:10$/);
  });
});

describe('whereabouts', () => {
  it('says here now over last seen, with the camera when known', () => {
    expect(whereabouts({ here_now: { cam: 'cam0' }, last_seen: at(2026, 10, 1, 9, 0) })).toBe('Here now, seen by cam0');
    expect(whereabouts({ here_now: { cam: null } })).toBe('Here now');
  });

  it('names the camera that saw them, by its name from Settings when it has one, else its id', () => {
    expect(whereabouts({ here_now: { cam: 'cam0', place: 'front camera' } })).toBe('Here now, seen by Front camera');
    expect(whereabouts({ here_now: { cam: 'cam0', place: 'cam0' } })).toBe('Here now, seen by cam0');
    vi.useFakeTimers({ now: NOW, toFake: ['Date'] });
    try {
      expect(
        whereabouts({ last_seen: at(2026, 10, 27, 20, 12), last_seen_cam: 'cam1', last_seen_place: 'rear camera' }),
      ).toBe('Last seen yesterday at 20:12 by Rear camera');
    } finally {
      vi.useRealTimers();
    }
  });

  it('says when and where they were last seen, or nothing', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date'] });
    try {
      expect(whereabouts({ last_seen: at(2026, 10, 27, 20, 12), last_seen_cam: 'cam1' })).toBe(
        'Last seen yesterday at 20:12 by cam1',
      );
      expect(whereabouts({ last_seen: at(2026, 10, 28, 18, 42) })).toBe('Last seen today at 18:42');
      expect(whereabouts({ last_seen: null, here_now: null })).toBe('');
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('shortWhen', () => {
  it.each([
    [at(2026, 10, 28, 19, 30), 'today 19:30'],
    [at(2026, 10, 27, 8, 5), 'yesterday 08:05'],
    [at(2026, 10, 26, 19, 30), 'Mon 19:30'],
    [at(2026, 10, 22, 7, 0), 'Thu 07:00'],
  ])('%d is %j', (ts, text) => {
    expect(shortWhen(ts, NOW)).toBe(text);
  });

  it('names the date from a week back, and the year only when it is another year', () => {
    expect(shortWhen(at(2026, 10, 21, 9, 10), NOW)).toBe('Oct 21 09:10');
    expect(shortWhen(at(2025, 12, 31, 9, 10), NOW)).toMatch(/2025.* 09:10$/);
  });
});
