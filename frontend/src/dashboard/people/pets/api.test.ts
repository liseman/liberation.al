import { describe, expect, it } from 'vitest';
import { byWindow, PET_WINDOW_S, petVideos, type Seen, span, waitingGroups } from './api';

const seen = (id: string, extra: Partial<Seen> = {}): Seen => ({
  track_id: id,
  cam: 'cam0',
  start: 100,
  end: 101,
  how: 'labeled',
  ...extra,
});

it('uses only server-prepared thumbnails among the newest four sightings, independent of photo readiness', () => {
  const list = [
    seen('pending', { video_status: 'pending', video_thumb: '/do-not-fetch.jpg' }),
    seen('ready', { video_status: 'ready', video_thumb: '/saved/scene.jpg', photo_ready: false }),
    seen('missing', { video_status: 'ready', video_thumb: null }),
    seen('unavailable', { video_status: 'unavailable', video_thumb: '/also-do-not-fetch.jpg' }),
    seen('older', { video_status: 'ready', video_thumb: '/old.jpg' }),
  ];
  expect(petVideos(list)).toEqual([
    { cam: 'cam0', start: 100, end: 105, duration_s: 5, thumb: '/saved/scene.jpg', what: null },
  ]);
  expect(petVideos([seen('unprepared')])).toEqual([]);
});

it('passes over sightings the server offers no video of, such as one without a camera', () => {
  const ready = (id: string, start: number) =>
    seen(id, { start, end: start + 1, video_status: 'ready', video_thumb: `/saved/${id}.jpg` });
  const list = [
    seen('photo-only-a', { cam: '' }),
    seen('photo-only-b', { cam: '' }),
    ready('a', 400),
    ready('b', 300),
    ready('c', 200),
    ready('d', 100),
    ready('fifth', 50),
  ];
  expect(petVideos(list).map(v => v.thumb)).toEqual(['/saved/a.jpg', '/saved/b.jpg', '/saved/c.jpg', '/saved/d.jpg']);
});

it('says when sightings were in 24-hour time, the day once', () => {
  const t = (h: number, m: number) => new Date(2026, 10, 6, h, m).getTime() / 1000;
  expect(span(t(13, 9), t(13, 9))).toMatch(/^\S+ \S+, 13:09$/);
  expect(span(t(11, 9), t(13, 10))).toMatch(/, 11:09 to 13:10$/);
});

describe('one card per pet per short window', () => {
  const T = 1_800_000_000;
  const sighting = (id: string, at: number) => ({
    track_id: id,
    species: 'dog',
    cam: 'cam0',
    start: T + at,
    end: T + at + 5,
    samples: 1,
    label: null,
  });
  const group = (id: string, at: number, guess: { id: string; score: number; band: 'probable' | 'maybe' } | null) => ({
    id,
    species: 'dog',
    sightings: [sighting(`${id}-t`, at)],
    guess: guess && { ...guess, name: guess.id, close: null },
  });
  const BESS = { id: 'bess', score: 90, band: 'probable' as const };

  it('joins groups guessed as the same pet within the window, as unsure as its least sure part', () => {
    const [card, ...rest] = byWindow([
      group('g3', 600, BESS),
      group('g2', 300, { id: 'bess', score: 70, band: 'maybe' }),
      group('g1', 0, BESS),
    ]);
    expect(rest).toEqual([]);
    expect(card?.id).toBe('g3');
    expect(card?.sightings.map(s => s.track_id)).toEqual(['g3-t', 'g2-t', 'g1-t']);
    expect(card?.guess).toMatchObject({ id: 'bess', score: 70, band: 'maybe' });
  });

  it('starts a new card once a card would span more than the window, however close each step is', () => {
    const steps = [0, 1, 2, 3].map(i => group(`g${i}`, (3 - i) * 25 * 60, BESS)); // 25 minutes apart, newest first
    expect(byWindow(steps).map(c => c.sightings.length)).toEqual([3, 1]);
  });

  it('keeps apart groups further apart than the window, other pets, and ones with no guess', () => {
    const cards = byWindow([
      group('late', PET_WINDOW_S + 120, BESS),
      group('other', 60, { id: 'mia', score: 90, band: 'probable' }),
      group('none-a', 30, null),
      group('none-b', 20, null),
      group('early', 0, BESS),
    ]);
    expect(cards.map(c => c.id)).toEqual(['late', 'other', 'none-a', 'none-b', 'early']);
  });

  it('counts the cards, unless the server listed only some of the groups', () => {
    const groups = [group('a', 60, BESS), group('b', 0, BESS)];
    const view = { groups, recognition: [], names: {} };
    expect(waitingGroups(view)).toBe(1);
    expect(waitingGroups({ ...view, count: 40 })).toBe(40);
  });
});
