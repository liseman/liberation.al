import { describe, expect, it } from 'vitest';
import { CopyRoom } from './small';

/** Run frames through a CopyRoom: every gapMs a frame that takes baseMs, plus copyMs when the copy is made. Returns
 *  how many of them were made with the copy. */
function run(room: CopyRoom, frames: number, gapMs: number, baseMs: number, copyMs: number, from = 0): number {
  let made = 0;
  for (let i = 0; i < frames; i++) {
    const now = from + i * gapMs;
    const copy = room.wanted(now);
    if (copy) made += 1;
    room.took(now * 1000, baseMs, copy ? copyMs : undefined);
  }
  return made;
}

describe('CopyRoom', () => {
  it('makes the copy with every frame where the frame and the copy fit well within the time between frames', () => {
    // 30 a second, 8 ms a frame, 4 ms a copy
    expect(run(new CopyRoom(), 300, 1000 / 30, 8, 4)).toBe(300);
  });

  it('leaves the copy to the page where it would not fit, trying again only now and then', () => {
    // 60 a second, 11 ms a frame, 4 ms a copy: together most of the 16.7 ms there is
    const room = new CopyRoom();
    const made = run(room, 3600, 1000 / 60, 11, 4);
    // A minute of frames: a few to learn the cost, three early tries, then one every ten seconds
    expect(made).toBeGreaterThan(3);
    expect(made).toBeLessThan(20);
  });

  it('takes the copy up again once there is room', () => {
    const room = new CopyRoom();
    run(room, 600, 1000 / 60, 11, 4);
    // The stream drops to 30 a second: the same work now fits
    const made = run(room, 900, 1000 / 30, 11, 4, 10_000);
    expect(made).toBeGreaterThan(800);
  });

  it('makes the copy while it does not yet know the frame rate or what the copy costs', () => {
    const room = new CopyRoom();
    expect(room.wanted(0)).toBe(true);
    room.took(0, 50, 50);
    expect(room.wanted(1)).toBe(true);
  });
});
