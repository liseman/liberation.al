import { describe, expect, it } from 'vitest';
import { type ShuttleInput, WheelShuttle } from './wheelShuttle';

const input = (more: Partial<ShuttleInput> = {}): ShuttleInput => ({
  deltaX: 0,
  deltaY: 100,
  deltaMode: 0,
  width: 1000,
  span: 3600,
  now: 0,
  ...more,
});
describe('wheel shuttle', () => {
  it('starts small even for an oversized notch', () => {
    expect(new WheelShuttle().step(input())).toBeCloseTo(0.9);
    expect(new WheelShuttle().step(input({ deltaY: 10000 }))).toBeCloseTo(0.9);
  });
  it('ramps a fast burst up to the cap', () => {
    const shuttle = new WheelShuttle();
    const steps = Array.from({ length: 40 }, (_, i) => shuttle.step(input({ now: i * 20 })));
    expect(steps[1]).toBeGreaterThan(steps[0] ?? 0);
    expect(steps[5]).toBeGreaterThan(steps[1] ?? 0);
    expect(steps.at(-1)).toBeCloseTo(14.4);
    expect(Math.max(...steps)).toBeLessThanOrEqual(14.4);
  });
  it('resets after a pause and on reversal', () => {
    const shuttle = new WheelShuttle();
    shuttle.step(input());
    expect(shuttle.step(input({ now: 20 }))).toBeGreaterThan(0.9);
    expect(shuttle.step(input({ now: 270 }))).toBeCloseTo(0.9);
    expect(shuttle.step(input({ now: 280, deltaY: -100 }))).toBeCloseTo(-0.9);
  });
  it('keeps slow notches small and does not accelerate merely on trackpad event rate', () => {
    const mouse = new WheelShuttle();
    expect(mouse.step(input())).toBeCloseTo(0.9);
    expect(mouse.step(input({ now: 300 }))).toBeCloseTo(0.9);
    const trackpad = new WheelShuttle();
    const steps = Array.from({ length: 100 }, (_, i) => trackpad.step(input({ deltaY: 1, now: i * 2 })));
    expect(steps.reduce((a, b) => a + b, 0)).toBeCloseTo(0.9); // one notch's distance spread over 200 ms
  });
  it('normalizes pixels, lines, pages, and the dominant axis', () => {
    expect(new WheelShuttle().step(input({ deltaY: 48 }))).toBeCloseTo(0.432);
    expect(new WheelShuttle().step(input({ deltaY: 3, deltaMode: 1 }))).toBeCloseTo(0.432);
    expect(new WheelShuttle().step(input({ deltaY: 1, deltaMode: 2 }))).toBeCloseTo(0.9);
    expect(new WheelShuttle().step(input({ deltaX: -50, deltaY: 5 }))).toBeCloseTo(-0.45);
  });
  it('scales with zoom, bounding an initial whole-day step to two seconds', () => {
    expect(new WheelShuttle().step(input({ span: 1800 }))).toBeCloseTo(0.45);
    expect(new WheelShuttle().step(input({ span: 86400 }))).toBe(2);
  });
  it.each([{ deltaY: 0 }, { deltaY: NaN }, { width: 0 }, { span: -1 }, { now: Infinity }])(
    'ignores invalid input without priming acceleration: %j',
    more => {
      const shuttle = new WheelShuttle();
      expect(shuttle.step(input(more))).toBe(0);
      expect(shuttle.step(input())).toBeCloseTo(0.9);
    },
  );
});
