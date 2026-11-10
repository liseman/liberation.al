import { describe, expect, it, vi } from 'vitest';
import { PersonPlayback } from './personPlayback';

const first = { start: 10, end: 12, cam: null };
const second = { start: 50, end: 53, cam: null };
const make = () => {
  const host = { play: vi.fn(), stop: vi.fn() };
  return { queue: new PersonPlayback(host), host };
};

describe('person-only playback', () => {
  it('plays the selected spans back to back and stops at the final endpoint', () => {
    const { queue, host } = make();
    queue.start('Alice', 'audio', [first, second]);
    expect(host.play.mock.calls).toEqual([[first, 'audio']]);
    expect(queue.atEnd(11.99)).toBe(false);
    expect(queue.atEnd(12)).toBe(true);
    expect(host.play.mock.calls).toEqual([
      [first, 'audio'],
      [second, 'audio'],
    ]);
    expect(queue.run?.index).toBe(1);
    expect(queue.atEnd(53)).toBe(true);
    expect(queue.run).toBeNull();
    expect(host.stop).toHaveBeenCalledOnce();
  });

  it('consumes line endings, but leaves room-audio endings under video to the engine', () => {
    const { queue, host } = make();
    queue.start('Alice', 'audio', [first, second]);
    expect(queue.audioEnded()).toBe(true);
    expect(queue.current).toEqual(second);
    queue.start('Alice', 'video', [{ start: 100, end: 110, cam: 'front' }]);
    expect(queue.audioEnded()).toBe(false);
    expect(queue.current?.cam).toBe('front');
    expect(host.stop).not.toHaveBeenCalled();
  });

  it('a manual takeover cancels future transitions without pausing the new playback', () => {
    const { queue, host } = make();
    queue.start('Alice', 'audio', [first, second]);
    queue.cancel();
    expect(queue.atEnd(100)).toBe(false);
    expect(queue.audioEnded()).toBe(false);
    expect(host.play).toHaveBeenCalledOnce();
    expect(host.stop).not.toHaveBeenCalled();
  });

  it('keeps a snapshot if the caller mutates its moments, and a replacement starts fresh', () => {
    const { queue, host } = make();
    const selection = [{ ...first }, { ...second }];
    queue.start('Alice', 'audio', selection);
    const selectedSecond = selection[1];
    if (!selectedSecond) throw new Error('missing test selection');
    selectedSecond.end = 500;
    selection.length = 0;
    queue.next();
    expect(queue.current?.end).toBe(53);
    queue.start('Dana', 'video', [{ start: 200, end: 202, cam: 'rear' }]);
    expect(queue.run?.person).toBe('Dana');
    expect(queue.run?.index).toBe(0);
    expect(host.play).toHaveBeenLastCalledWith({ start: 200, end: 202, cam: 'rear' }, 'video');
  });

  it('skips unplayable input and stops an empty replacement instead of retaining the old queue', () => {
    const { queue, host } = make();
    queue.start('Alice', 'video', [first, { start: 20, end: 20, cam: 'front' }, { start: 30, end: 31, cam: 'rear' }]);
    expect(queue.current).toEqual({ start: 30, end: 31, cam: 'rear' });
    queue.start('Alice', 'audio', [{ start: 1, end: Number.NaN, cam: null }]);
    expect(queue.run).toBeNull();
    expect(host.stop).toHaveBeenCalledOnce();
  });
});
