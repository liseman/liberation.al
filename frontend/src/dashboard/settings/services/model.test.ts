import { expect, describe as group, it } from 'vitest';
import {
  type Described,
  describe,
  errorTime,
  groupCount,
  grouped,
  isUp,
  locked,
  overall,
  rowStatus,
  type Service,
  type ServicesView,
  startable,
  summary,
  swept,
  toggleAction,
  turnedOff,
} from './model';

const svc = (over: Partial<Service> = {}): Service => ({
  name: 'voice',
  description: 'Listens',
  state: 'not running',
  installed: false,
  pids: [],
  selected: true,
  ...over,
});

const view = (services: Service[], over: Partial<ServicesView> = {}): ServicesView => ({
  ok: true,
  control: true,
  services,
  ...over,
});

group('describe', () => {
  const cases: [string, Partial<Service>, Described][] = [
    ['active', { installed: true, state: 'active' }, { label: 'running', tone: 'ok', on: true }],
    ['running', { installed: true, state: 'running' }, { label: 'running', tone: 'ok', on: true }],
    ['scheduled', { installed: true, state: 'active', scheduled: true }, { label: 'scheduled', tone: 'ok', on: true }],
    ['activating', { installed: true, state: 'activating' }, { label: 'starting', tone: 'warn', on: true }],
    ['reloading', { installed: true, state: 'reloading' }, { label: 'starting', tone: 'warn', on: true }],
    ['failed', { installed: true, state: 'failed' }, { label: 'failed', tone: 'bad', on: true }],
    ['inactive', { installed: true, state: 'inactive' }, { label: 'stopped', tone: 'warn', on: true }],
    ['not running', { installed: true, state: 'not running' }, { label: 'stopped', tone: 'warn', on: true }],
    ['another state', { installed: true, state: 'deactivating' }, { label: 'deactivating', tone: 'warn', on: true }],
    ['by hand', { pids: [42] }, { label: 'running by hand', tone: 'ok', on: false }],
    ['not installed', { state: 'active' }, { label: 'off', tone: 'none', on: false }],
    [
      'a crash loop',
      { installed: true, state: 'active', restarts: 4, up_s: 30 },
      { label: 'keeps restarting', tone: 'bad', on: true },
    ],
    [
      'a crash loop, coming up again',
      { installed: true, state: 'activating', restarts: 82 },
      { label: 'keeps restarting', tone: 'bad', on: true },
    ],
  ];
  it.each(cases)('%s', (_, over, expected) => {
    expect(describe(svc(over))).toEqual(expected);
  });
});

group('overall', () => {
  it('is grey before the first read', () => {
    expect(overall(undefined)).toBe('none');
  });

  it('is red when status is unreadable', () => {
    expect(overall(view([svc({ installed: true, state: 'active' })], { ok: false }))).toBe('bad');
  });

  it('is green when every selected service is up, by hand or not, ignoring the unselected', () => {
    const list = [
      svc({ name: 'a', installed: true, state: 'active' }),
      svc({ name: 'b', pids: [7] }),
      svc({ name: 'c', selected: false }),
    ];
    expect(overall(view(list))).toBe('ok');
  });

  it('is amber when a selected service is stopped or starting', () => {
    const list = [
      svc({ name: 'a', installed: true, state: 'active' }),
      svc({ name: 'b', installed: true, state: 'inactive' }),
    ];
    expect(overall(view(list))).toBe('warn');
    expect(overall(view([svc({ installed: true, state: 'activating' })]))).toBe('warn');
  });

  it('is red when a selected service failed or keeps restarting', () => {
    expect(
      overall(view([svc({ installed: true, state: 'active' }), svc({ name: 'b', installed: true, state: 'failed' })])),
    ).toBe('bad');
    expect(overall(view([svc({ installed: true, state: 'active', restarts: 5, up_s: 20 })]))).toBe('bad');
  });
});

group('summary', () => {
  it('counts the running, the ones needing attention, the other stopped ones, the enabled, and the turned off', () => {
    const list = [
      svc({ name: 'up', installed: true, state: 'active' }),
      svc({ name: 'byhand', selected: false, pids: [4] }), // running, though not enabled here
      svc({ name: 'failed', installed: true, state: 'failed' }),
      svc({ name: 'loop', installed: true, state: 'active', restarts: 3, up_s: 10 }),
      svc({ name: 'stopped', installed: true, state: 'inactive' }),
      svc({ name: 'off', selected: false }),
      svc({ name: 'off too', selected: false, installed: true, state: 'inactive' }),
    ];
    const n = summary(view(list));
    expect(n.running).toBe(2);
    expect(n.attention.map(s => s.name)).toEqual(['failed', 'loop']);
    expect(n.stopped).toBe(1);
    expect(n.enabled).toBe(4);
    expect(n.off).toBe(2);
  });
});

group('groupCount', () => {
  it('says how many of a group run, red while one is down', () => {
    const up = svc({ installed: true, state: 'active' });
    expect(groupCount([up, up])).toEqual({ text: '2 of 2 running', tone: 'ok' });
    expect(groupCount([up, svc({ installed: true, state: 'failed' })])).toEqual({
      text: '1 of 2 running',
      tone: 'bad',
    });
    expect(groupCount([up])).toEqual({ text: 'Running', tone: 'ok' });
    expect(groupCount([svc()])).toEqual({ text: 'Not running', tone: 'bad' });
  });
});

group('startable', () => {
  it('picks the enabled services that are not up, never the dashboard or one running by hand', () => {
    const list = [
      svc({ name: 'off' }),
      svc({ name: 'stopped', installed: true, state: 'inactive' }),
      svc({ name: 'failed', installed: true, state: 'failed' }),
      svc({ name: 'up', installed: true, state: 'active' }),
      svc({ name: 'byhand', pids: [9] }),
      svc({ name: 'stopped, but a copy runs by hand', installed: true, state: 'inactive', pids: [8] }),
      svc({ name: 'dashboard' }),
      svc({ name: 'unselected', selected: false }),
      svc({ name: 'starting', installed: true, state: 'activating' }),
      svc({ name: 'looping', installed: true, state: 'activating', restarts: 9 }), // the manager starts it again
    ];
    expect(startable(list).map(s => s.name)).toEqual(['off', 'stopped', 'failed']);
  });
});

group('rowStatus', () => {
  const up = svc({ installed: true, state: 'active' });

  it('shows the way a switch in flight is going', () => {
    expect(rowStatus(up, { name: 'voice', on: false })).toEqual({ label: 'stopping', tone: 'warn' });
    expect(rowStatus(svc(), { name: 'voice', on: true })).toEqual({ label: 'starting', tone: 'warn' });
  });

  it('describes the service when nothing is in flight', () => {
    expect(rowStatus(up, null)).toEqual({ label: 'running', tone: 'ok', on: true });
  });
});

group('swept', () => {
  it('touches nothing when no bulk action runs, and never the dashboard', () => {
    expect(swept(svc({ installed: true }), null)).toBeNull();
    expect(swept(svc({ name: 'dashboard', installed: true }), 'repair')).toBeNull();
    expect(swept(svc({ name: 'dashboard', installed: true }), 'off')).toBeNull();
  });

  it('starts every selected service on Repair except one running by hand', () => {
    expect(swept(svc({ installed: true, pids: [3] }), 'repair')).toEqual({ name: 'voice', on: true });
    expect(swept(svc(), 'repair')).toEqual({ name: 'voice', on: true });
    expect(swept(svc({ pids: [3] }), 'repair')).toBeNull();
    expect(swept(svc({ selected: false, installed: true }), 'repair')).toBeNull();
  });

  it('stops every installed service on All off, selected or not', () => {
    expect(swept(svc({ installed: true, selected: false }), 'off')).toEqual({ name: 'voice', on: false });
    expect(swept(svc({ pids: [3] }), 'off')).toBeNull();
  });
});

group('locked', () => {
  it('names the first reason a switch is off limits', () => {
    expect(locked(svc({ selected: false }), false)).toBe('Services can only be switched from this machine');
    expect(locked(svc({ selected: false }), true)).toBe('Not enabled for this machine: enable it first');
    expect(locked(svc({ name: 'dashboard' }), true)).toBe(
      'This page runs on the dashboard; use pack-services in a terminal',
    );
    expect(locked(svc(), true)).toBeNull();
  });
});

group("a service's switch", () => {
  it('is on while the service runs or is starting, by hand or not', () => {
    expect(isUp(svc({ installed: true, state: 'active' }))).toBe(true);
    expect(isUp(svc({ installed: true, state: 'activating' }))).toBe(true);
    expect(isUp(svc({ pids: [4] }))).toBe(true);
    expect(isUp(svc({ installed: true, state: 'failed' }))).toBe(false);
    expect(isUp(svc({ state: 'active' }))).toBe(false); // not installed: the state is no service's
  });

  it('stops it, starts it again, or turns on one not installed', () => {
    expect(toggleAction(svc({ installed: true, state: 'active' }), false)).toBe('stop');
    expect(toggleAction(svc({ installed: true, state: 'failed' }), true)).toBe('start');
    expect(toggleAction(svc(), true)).toBe('on');
  });
});

group('the sections', () => {
  it('lists the turned-off services apart: stopped and not enabled here', () => {
    expect(turnedOff(svc({ selected: false, installed: true, state: 'inactive' }))).toBe(true);
    expect(turnedOff(svc({ selected: false }))).toBe(true);
    expect(turnedOff(svc({ selected: false, pids: [3] }))).toBe(false); // still running: in the main list
    expect(turnedOff(svc())).toBe(false); // enabled here, even though off
  });

  it('groups the rest in a fixed order, with unknown groups under Other last, and skips empty ones', () => {
    const list = [
      svc({ name: 'listen', group: 'workers' }),
      svc({ name: 'mystery', group: 'nope' }),
      svc({ name: 'broker', group: 'core' }),
      svc({ name: 'old' }),
      svc({ name: 'gone', group: 'core', selected: false }),
      svc({ name: 'see', group: 'workers' }),
    ];
    expect(grouped(list).map(g => [g.label, g.services.map(s => s.name)])).toEqual([
      ['Continuum core', ['broker']],
      ['Workers', ['listen', 'see']],
      ['Other', ['mystery', 'old']],
    ]);
  });
});

group('errorTime', () => {
  const now = new Date(2026, 9, 30, 12, 0);
  it("gives the clock for today's errors and the date as well for older ones", () => {
    expect(errorTime({ at: '09:02:11', ts: new Date(2026, 9, 30, 9, 2, 11).getTime() / 1000 }, now)).toBe('09:02');
    expect(errorTime({ at: '23:58:00', ts: new Date(2026, 9, 29, 23, 58).getTime() / 1000 }, now)).toBe('Oct 29 23:58');
  });

  it("falls back to the log's own clock, or nothing", () => {
    expect(errorTime({ at: '09:02:11' }, now)).toBe('09:02');
    expect(errorTime({ at: '', ts: null }, now)).toBe('');
  });
});
