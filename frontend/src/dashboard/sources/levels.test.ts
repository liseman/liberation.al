import { afterEach, describe, expect, it, vi } from 'vitest';
import { FakeEventSource } from '../camera/testFakes';
import { holdPeak, levelWidth, subscribeLevels } from './levels';

describe('levelWidth', () => {
  it.each([
    [-90, '0.0%'],
    [-45, '50.0%'],
    [-30, '66.7%'],
    [0, '100.0%'],
    [-120, '0.0%'],
    [6, '100.0%'],
  ])('%d dBFS is %s', (db, width) => {
    expect(levelWidth(db)).toBe(width);
  });
});

describe('holdPeak', () => {
  it('keeps a louder new peak', () => {
    expect(holdPeak(-10, -30, 0.1)).toBe(-10);
  });
  it('lets the held peak fall 20 dB a second', () => {
    expect(holdPeak(-40, -10, 0.5)).toBe(-20);
  });
  it('stops falling at the new peak', () => {
    expect(holdPeak(-15, -10, 0.5)).toBe(-15);
  });
  it('starts from the first peak', () => {
    expect(holdPeak(-50, undefined, 0.04)).toBe(-50);
  });
});

describe('subscribeLevels (#353)', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('shares one stream between listeners, and closes it with the last', () => {
    FakeEventSource.all = [];
    vi.stubGlobal('EventSource', FakeEventSource);
    const a: unknown[] = [];
    const b: unknown[] = [];
    const offA = subscribeLevels(l => a.push(l));
    const offB = subscribeLevels(l => b.push(l));
    expect(FakeEventSource.all).toHaveLength(1);
    FakeEventSource.all[0]?.send({ levels: { desk: { rms: -40, peak: -30 } } });
    expect(a).toEqual([{ desk: { rms: -40, peak: -30 } }]);
    expect(b).toEqual(a);
    offA();
    expect(FakeEventSource.all[0]?.closed).toBe(false);
    offB();
    expect(FakeEventSource.all[0]?.closed).toBe(true);
  });

  it('opens a new stream for the next listener after the server closed the last one', () => {
    FakeEventSource.all = [];
    vi.stubGlobal('EventSource', FakeEventSource);
    const seen: unknown[] = [];
    const off = subscribeLevels(l => seen.push(l));
    FakeEventSource.all[0]?.fail();
    expect(seen).toEqual([{}]); // the meters empty out
    const offLater = subscribeLevels(() => {});
    expect(FakeEventSource.all).toHaveLength(2);
    expect(FakeEventSource.all[1]?.closed).toBe(false);
    off();
    offLater();
  });
});
