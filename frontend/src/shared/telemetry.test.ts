import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  DEDUPE_MS,
  flush,
  MAX_PER_MINUTE,
  pathOf,
  queued,
  report,
  requestEnded,
  requestStarted,
  resetTelemetry,
  SLOW_NAV_MS,
  SLOW_REQUEST_MS,
  startNavigation,
} from './telemetry';

beforeEach(() => resetTelemetry());
afterEach(() => {
  resetTelemetry();
  vi.useRealTimers();
});

// #2007: the page's own log, kept to paths, capped, and traced per navigation.
describe('what a log line keeps', () => {
  it.each([
    ['/people#card=Dana', '/people'],
    ['/people/profile?name=Dana', '/people/profile'],
    ['http://devbox:8765/pets?x=1#y', '/pets'],
  ])('keeps %s as the path %s', (url, path) => {
    expect(pathOf(url)).toBe(path);
  });
});

describe('the rate cap', () => {
  it('keeps the same event once in a while, and at most so many a minute', () => {
    expect(report({ type: 'error', message: 'boom' }, 1000)).toBe(true);
    expect(report({ type: 'error', message: 'boom' }, 1000 + DEDUPE_MS - 1)).toBe(false);
    expect(report({ type: 'error', message: 'boom' }, 1000 + DEDUPE_MS)).toBe(true);
    resetTelemetry();
    for (let n = 0; n < MAX_PER_MINUTE; n++) expect(report({ type: 'error', message: `e${n}` }, 5000)).toBe(true);
    expect(report({ type: 'error', message: 'one too many' }, 5000)).toBe(false);
    expect(report({ type: 'error', message: 'a minute on' }, 65_000)).toBe(true);
  });
});

describe('requests', () => {
  it('reports a failed and a slow request, by path, and never an aborted one', () => {
    requestStarted();
    requestEnded('/people/profile?name=Dana', 'GET', 500, 120);
    requestStarted();
    requestEnded('/review/summary?day=x', 'GET', 200, SLOW_REQUEST_MS + 1);
    requestStarted();
    requestEnded('/pets', 'GET', 200, 40);
    requestStarted();
    requestEnded('/vision', 'GET', null, 25_000, true);
    expect(queued().map(e => [e.type, e.url, e.status ?? null])).toEqual([
      ['request', '/people/profile', 500],
      ['request', '/review/summary', 200],
    ]);
  });
});

describe('navigations', () => {
  it('reports a slow one with the requests it waited on, once they settle', () => {
    vi.useFakeTimers();
    const clock = vi.spyOn(performance, 'now');
    clock.mockReturnValue(0);
    startNavigation('/pets#species=dog');
    requestStarted();
    clock.mockReturnValue(SLOW_NAV_MS + 100);
    requestEnded('/pets', 'GET', 200, SLOW_NAV_MS + 100);
    vi.advanceTimersByTime(600); // it settles once nothing is loading
    const nav = queued().find(e => e.type === 'navigation');
    expect(nav).toMatchObject({ route: '/pets', requests: [{ url: '/pets', status: 200 }] });
    clock.mockRestore();
  });

  it('keeps a quick one with no failure out of the log', () => {
    vi.useFakeTimers();
    const clock = vi.spyOn(performance, 'now');
    clock.mockReturnValue(0);
    startNavigation('/review');
    requestStarted();
    clock.mockReturnValue(80);
    requestEnded('/review/summary', 'GET', 200, 80);
    vi.advanceTimersByTime(600);
    expect(queued()).toEqual([]);
    clock.mockRestore();
  });
});

describe('sending', () => {
  it('sends what is queued in batches, as {events: [...]}, and empties the queue', () => {
    for (let n = 0; n < 3; n++) report({ type: 'error', message: `e${n}` }, 1000);
    const sent: unknown[] = [];
    flush(body => sent.push(JSON.parse(body)));
    expect(sent).toHaveLength(1);
    expect((sent[0] as { events: { message: string }[] }).events.map(e => e.message)).toEqual(['e0', 'e1', 'e2']);
    expect(queued()).toEqual([]);
  });
});
