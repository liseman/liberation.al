import { describe, expect, it } from 'vitest';
import {
  around,
  CAP,
  colorOf,
  EMPTY,
  groupServices,
  hello,
  type Level,
  type LogState,
  levelCounts,
  levelOf,
  pause,
  receive,
  resume,
  visible,
  type WireLine,
} from './logs';

const HELLO = { ts: 1, services: ['ambient', 'vision'], missing: [] };
const line = (s: string, l: string, t: number | null = 100): WireLine => ({ s, l, t });
const texts = (state: LogState) => state.lines.map(x => `${x.s}: ${x.l}`);
const levels = (state: LogState) => state.lines.map(x => x.lv);
const ALL = { hidden: new Set<string>(), query: '', mode: 'filter', min: 'debug' } as const;

describe('the level a line states', () => {
  it.each<[string, Level]>([
    ['14:02:07 [backfill] DEBUG frame 3 has no faces', 'debug'],
    ['14:02:07 [backfill] INFO segment cam0/2026-10-28/14-00.mp4: 312 faces', 'info'],
    ['14:02:09 [vision] WARN camera cam1 lost, reconnecting in 2 s', 'warn'],
    ['14:02:11 [dashboard] ERROR GET /review/presence failed: ValueError: bad day', 'error'],
    ['14:02:12 [voice] WARNING disk nearly full', 'warn'],
    ['14:02:13 [voice] CRITICAL out of memory', 'error'],
    ['14:02:14 ERROR no tag on this one', 'error'],
    ['[backfill] WARN no time on this one', 'warn'],
    ['INFO neither time nor tag', 'info'],
    ['14:02:15 [dashboard] ERROR', 'error'],
    ['14:02:16 [backfill] INFO retried after an ERROR earlier', 'info'],
    ['2026-11-03T02:20:15-08:00 [backfill] INFO dated stamp (#1696)', 'info'],
    ['2026-11-03T02:20:16+05:30 [continuum-broker] ERROR bind failed', 'error'],
  ])('reads %j as %s', (text, level) => {
    expect(levelOf(text)).toBe(level);
  });

  it.each([
    '14:01:00 [backfill] segment done',
    'Traceback (most recent call last):',
    '  File "x.py", line 3, in f',
    'ValueError: boom',
    '14:01:00 something',
    '14:01:00 [backfill] segment had ERROR frames',
    '14:01:00 [backfill] error: lowercase is not a level',
    '14:01:00 [backfill] ERRORS 3',
    'ERROR:root:no space after it',
    '14:01:00 [backfill] NOTICE not a level we know',
    '2026-11-03T02:37:28-08:00 capture worker kitchen: reconnecting',
    '',
  ])('finds no level in %j', text => {
    expect(levelOf(text)).toBeNull();
  });
});

describe('levels the line does not state', () => {
  it('a traceback after an error is an error, and the next stated level takes over', () => {
    const s = receive(hello(EMPTY, HELLO), [
      line('vision', '14:00:00 [vision] ERROR capture failed'),
      line('vision', 'Traceback (most recent call last):'),
      line('vision', '  File "x.py", line 3, in f'),
      line('vision', 'ValueError: boom'),
      line('vision', '14:00:01 [vision] INFO capture back'),
      line('vision', '14:00:02 [vision] frame ok'),
    ]);
    expect(levels(s)).toEqual(['error', 'error', 'error', 'error', 'info', 'info']);
    expect(s.lines.map(x => x.inherited)).toEqual([false, true, true, true, false, false]);
  });

  it('a line of its own with a time but no level is info, even right after a warning', () => {
    const s = receive(hello(EMPTY, HELLO), [
      line('vision', '14:00:00 [vision] WARN vlm error: timed out'),
      line('vision', '14:00:01   cam0 yaw 90 | 12 ms | No new caption.'),
      line('vision', '  retrying in 5 s'),
    ]);
    expect(levels(s)).toEqual(['warn', 'info', 'info']);
  });

  it('goes by the same service only, and a first line with no level is info', () => {
    const s = receive(hello(EMPTY, HELLO), [
      line('vision', '14:00:00 [vision] ERROR capture failed'),
      line('ambient', '14:00:00 [ambient] heard something'),
      line('ambient', '14:00:01 [ambient] WARN mic clipped'),
      line('vision', 'ValueError: boom'),
      line('ambient', 'still clipping'),
    ]);
    expect(levels(s)).toEqual(['error', 'info', 'warn', 'error', 'warn']);
  });

  it('carries across batches, while paused, and after a reconnect', () => {
    let s = receive(hello(EMPTY, HELLO), [line('vision', '14:00:00 [vision] ERROR capture failed', null)]);
    s = receive(pause(s), [line('vision', 'Traceback (most recent call last):')]);
    s = resume(s);
    s = hello(s, HELLO);
    s = receive(s, [
      line('vision', '14:00:00 [vision] ERROR capture failed', null),
      line('vision', 'Traceback (most recent call last):', null),
      line('vision', 'ValueError: boom', null),
    ]);
    expect(texts(s)).toEqual([
      'vision: 14:00:00 [vision] ERROR capture failed',
      'vision: Traceback (most recent call last):',
      'vision: ValueError: boom',
    ]);
    expect(levels(s)).toEqual(['error', 'error', 'error']);
  });
});

describe('the buffer', () => {
  it('keeps lines in order with ids that stay unique', () => {
    const s = receive(receive(hello(EMPTY, HELLO), [line('ambient', 'a')]), [line('vision', 'v')]);
    expect(texts(s)).toEqual(['ambient: a', 'vision: v']);
    expect(new Set(s.lines.map(x => x.id)).size).toBe(2);
  });

  it('lets the oldest lines go past the cap, and counts them', () => {
    const many = Array.from({ length: CAP + 3 }, (_, i) => line('ambient', `n${i}`));
    const s = receive(hello(EMPTY, HELLO), many);
    expect(s.lines).toHaveLength(CAP);
    expect(s.lines[0]?.l).toBe('n3');
    expect(s.dropped).toBe(3);
  });

  it('holds lines while paused and shows them on resume', () => {
    let s = receive(hello(EMPTY, HELLO), [line('ambient', 'before')]);
    s = receive(pause(s), [line('ambient', 'during 1'), line('vision', 'during 2')]);
    expect(texts(s)).toEqual(['ambient: before']);
    expect(s.held).toHaveLength(2);
    s = resume(s);
    expect(texts(s)).toEqual(['ambient: before', 'ambient: during 1', 'vision: during 2']);
    expect(s.held).toEqual([]);
  });

  it('after a reconnect, adds only the backlog lines it does not have yet', () => {
    let s = receive(hello(EMPTY, HELLO), [line('ambient', 'a1', null), line('ambient', 'a2', null)]);
    s = receive(s, [line('vision', 'v1')]);
    s = hello(s, HELLO); // the stream dropped and came back: its backlog repeats what we have
    s = receive(s, [line('ambient', 'a1', null), line('ambient', 'a2', null), line('ambient', 'a3', null)]);
    s = receive(s, [line('vision', 'v1', null), line('vision', 'v2', null)]);
    expect(texts(s)).toEqual(['ambient: a1', 'ambient: a2', 'vision: v1', 'ambient: a3', 'vision: v2']);
  });

  it('after a reconnect, keeps new lines that repeat the last one shown', () => {
    let s = receive(hello(EMPTY, HELLO), [line('ambient', 'try 1', null), line('ambient', 'unavailable', null)]);
    s = hello(s, HELLO);
    s = receive(s, [
      line('ambient', 'try 1', null),
      line('ambient', 'unavailable', null),
      line('ambient', 'try 2', null),
      line('ambient', 'unavailable', null),
    ]);
    expect(texts(s)).toEqual(['ambient: try 1', 'ambient: unavailable', 'ambient: try 2', 'ambient: unavailable']);
  });

  it('a log that starts after connecting is no longer counted missing', () => {
    let s = hello(EMPTY, { ts: 1, services: ['ambient', 'whisper'], missing: ['whisper'] });
    s = receive(s, [line('whisper', 'listening')]);
    expect(s.missing).toEqual([]);
  });
});

describe('what is shown', () => {
  const s = receive(hello(EMPTY, HELLO), [
    line('ambient', 'Heard Dana'),
    line('vision', 'cam0 dark'),
    line('ambient', 'idle'),
  ]);

  it('hides services and, in filter mode, lines that do not match the search', () => {
    const hidden = new Set(['vision']);
    expect(visible(s.lines, { ...ALL, hidden }).map(x => x.l)).toEqual(['Heard Dana', 'idle']);
    expect(visible(s.lines, { ...ALL, query: 'DANA' }).map(x => x.l)).toEqual(['Heard Dana']);
  });

  it('keeps every line in highlight mode', () => {
    expect(visible(s.lines, { ...ALL, query: 'dana', mode: 'highlight' })).toHaveLength(3);
  });

  it('gives the lines around one, from every service', () => {
    const middle = s.lines[1]?.id ?? -1;
    expect(around(s.lines, middle, 1).map(x => x.l)).toEqual(['Heard Dana', 'cam0 dark', 'idle']);
    expect(around(s.lines, 999)).toEqual([]);
  });

  it('colors services by their place in the list, so two services never share one', () => {
    expect(colorOf(['ambient', 'vision'], 'ambient')).not.toBe(colorOf(['ambient', 'vision'], 'vision'));
    expect(colorOf(['ambient'], 'ambient')).toBe(colorOf(['ambient', 'x'], 'ambient'));
  });
});

describe('the level filter', () => {
  const s = receive(hello(EMPTY, HELLO), [
    line('ambient', '10:00:00 [ambient] DEBUG frame'),
    line('ambient', '10:00:01 [ambient] INFO Heard Dana'),
    line('vision', '10:00:02 [vision] WARN cam1 lost'),
    line('vision', '10:00:03 [vision] ERROR cam1 gone'),
    line('vision', 'Traceback (most recent call last):'),
    line('ambient', '10:00:04 [ambient] WARN Dana too quiet'),
  ]);
  const at = (min: Level) => visible(s.lines, { ...ALL, min }).map(x => x.l);

  it('shows the chosen level and the ones above it', () => {
    expect(at('debug')).toHaveLength(6);
    expect(at('info')).toEqual([
      '10:00:01 [ambient] INFO Heard Dana',
      '10:00:02 [vision] WARN cam1 lost',
      '10:00:03 [vision] ERROR cam1 gone',
      'Traceback (most recent call last):',
      '10:00:04 [ambient] WARN Dana too quiet',
    ]);
    expect(at('warn')).toEqual([
      '10:00:02 [vision] WARN cam1 lost',
      '10:00:03 [vision] ERROR cam1 gone',
      'Traceback (most recent call last):',
      '10:00:04 [ambient] WARN Dana too quiet',
    ]);
    expect(at('error')).toEqual(['10:00:03 [vision] ERROR cam1 gone', 'Traceback (most recent call last):']);
  });

  it('combines with a hidden service and a search', () => {
    expect(visible(s.lines, { ...ALL, min: 'warn', hidden: new Set(['vision']) }).map(x => x.l)).toEqual([
      '10:00:04 [ambient] WARN Dana too quiet',
    ]);
    expect(visible(s.lines, { ...ALL, min: 'info', query: 'dana' }).map(x => x.l)).toEqual([
      '10:00:01 [ambient] INFO Heard Dana',
      '10:00:04 [ambient] WARN Dana too quiet',
    ]);
  });

  it('counts each level among the lines the other filters let through', () => {
    expect(levelCounts(s.lines, { ...ALL, min: 'error' })).toEqual({ debug: 1, info: 1, warn: 2, error: 2 });
    expect(levelCounts(s.lines, { ...ALL, hidden: new Set(['ambient']) })).toEqual({
      debug: 0,
      info: 0,
      warn: 1,
      error: 2,
    });
    expect(levelCounts(s.lines, { ...ALL, query: 'dana', mode: 'highlight' })).toEqual({
      debug: 1,
      info: 1,
      warn: 2,
      error: 2,
    });
  });
});

describe('groupServices', () => {
  it('puts each service in its section in the listed order, and an unknown one in Other', () => {
    expect(groupServices(['dashboard', 'see-actions', 'new-thing', 'see-decode', 'listen'])).toEqual([
      { title: 'Vision', services: ['see-decode', 'see-actions'] },
      { title: 'Audio and speech', services: ['listen'] },
      { title: 'Dashboard', services: ['dashboard'] },
      { title: 'Other', services: ['new-thing'] },
    ]);
  });

  it('places every service exactly once', () => {
    const all = ['dashboard', 'see-actions', 'x', 'listen', 'continuum-broker', 'y'];
    expect(
      groupServices(all)
        .flatMap(g => g.services)
        .sort(),
    ).toEqual([...all].sort());
  });
});
