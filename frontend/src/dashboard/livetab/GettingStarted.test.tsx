import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { setPref, usePrefs } from '../lib/prefs';
import type { Devices, Stats } from '../sources/api';
import { type Facts, GettingStarted, steps, stillHidden } from './GettingStarted';

const cam = (index: number, selected = true, present = true) => ({
  index,
  name: `Cam ${index}`,
  present,
  selected,
  in_use: selected,
});
const mic = (name: string, selected = true, present = true) => ({
  id: name,
  name,
  present,
  selected,
  in_use: selected,
});
const devices = (over: Partial<Devices> = {}): Devices => ({
  control: true,
  mics: [],
  cameras: [],
  notes: {},
  ...over,
});
const READY: Stats = { system: { missing: [] }, ollama: { up: true, missing: [] } };
const ALL_DONE: Facts = {
  stats: READY,
  devices: devices({ cameras: [cam(0)], mics: [mic('USB Microphone')] }),
  owner: 'Joe',
  voiceRef: true,
  ownerFaces: 3,
};
const step = (f: Facts, id: string) => steps(f).find(s => s.id === id);

beforeEach(() => vi.stubGlobal('NAMES', { views: { cam0: 'front camera' } }));
afterEach(() => {
  vi.unstubAllGlobals();
  setPref('gettingStartedHidden', undefined);
  setPref('gettingStartedDone', undefined);
});

describe('the steps', () => {
  it('are all to do on a fresh install, in the design’s order', () => {
    expect(steps({}).map(s => [s.id, s.done])).toEqual([
      ['packages', false],
      ['models', false],
      ['sources', false],
      ['name', false],
      ['voice', false],
      ['face', false],
    ]);
  });

  it('are all done when everything is set up', () => {
    expect(steps(ALL_DONE).every(s => s.done)).toBe(true);
  });

  it('name the missing system packages, and never claim they are present before the check answers', () => {
    const missing = { name: 'sox', why: '', optional: false, services: [] };
    expect(step({ stats: { system: { missing: [missing] } } }, 'packages')?.note).toBe('Missing: sox');
    expect(step({}, 'packages')).toMatchObject({ done: false, note: "Couldn't check yet" });
    expect(step({ stats: { system: { error: 'no apt' } } }, 'packages')).toMatchObject({
      done: false,
      note: "Couldn't check: no apt",
    });
  });

  it('count models done only with Ollama up and nothing missing', () => {
    expect(step({ stats: { ollama: { up: true, missing: ['qwen3-embedding:4b'] } } }, 'models')).toMatchObject({
      done: false,
      note: 'Missing: qwen3-embedding:4b',
    });
    expect(step({ stats: { ollama: { up: false, message: 'Start Ollama' } } }, 'models')?.note).toBe('Start Ollama');
  });

  it('need a camera and a microphone that are both picked and present, named by the camera’s name', () => {
    const f = (cameras: Devices['cameras'], mics: Devices['mics']) =>
      step({ devices: devices({ cameras, mics }) }, 'sources');
    expect(f([cam(0)], [mic('USB Microphone')])).toMatchObject({
      done: true,
      note: 'Front camera and USB Microphone are on',
    });
    expect(f([cam(0)], [])).toMatchObject({ done: false, note: 'A camera is on; pick a microphone' });
    expect(f([], [mic('USB Microphone')])?.note).toBe('USB Microphone is on; pick a camera');
    expect(f([cam(1, true, false)], [mic('USB Microphone')])?.done).toBe(false); // picked, but unplugged
  });

  it("send Tag your face to Live's own Who's this? when it's on the page, else to People's", () => {
    expect(steps({}, 'here').find(s => s.id === 'face')?.go).toEqual({ href: '/#whos-this' });
    expect(steps({}).find(s => s.id === 'face')?.go).toEqual({ href: '/people' });
  });

  it('need the owner named, their voice reference, and their face', () => {
    expect(step({ owner: '  ' }, 'name')?.done).toBe(false);
    expect(step({ owner: 'Joe' }, 'name')?.note).toBe("You're Joe");
    expect(step({ voiceRef: true }, 'voice')?.done).toBe(true);
    expect(step({ ownerFaces: 0 }, 'face')?.done).toBe(false);
    expect(step({ ownerFaces: 2 }, 'face')?.done).toBe(true);
  });
});

describe('Hide this for now', () => {
  const open = (...ids: string[]) => steps({}).map(s => ({ ...s, done: !ids.includes(s.id) }));

  it('holds while the open steps are the same or fewer', () => {
    expect(stillHidden('face,voice', open('face', 'voice'))).toBe(true);
    expect(stillHidden('face,voice', open('face'))).toBe(true);
  });

  it('lets go once a step opens that was done when it was pressed', () => {
    expect(stillHidden('face,voice', open('face', 'models'))).toBe(false);
  });

  it('isn’t in effect until pressed', () => {
    expect(stillHidden(undefined, open('face'))).toBe(false);
  });
});

function serve({
  stats = {} as Stats,
  dev = devices(),
  owner = '',
  voice = false,
  people = [] as { name: string; owner?: boolean; faces?: number; voices: number[] }[],
} = {}) {
  server.use(
    http.get('/stats', () => HttpResponse.json(stats)),
    http.get('/devices', () => HttpResponse.json(dev)),
    http.get('/config', () =>
      HttpResponse.json({
        ok: true,
        control: true,
        needs_setup: false,
        path: 'pack.toml',
        sections: [{ name: 'Owner', description: '', settings: [{ key: 'owner.name', value: owner }] }],
      }),
    ),
    http.get('/config/voice_ref', () => HttpResponse.json({ exists: voice })),
    http.get('/people', () => HttpResponse.json({ people, unknown: { voices: [], faces: [] } })),
  );
}
const card = () => screen.findByRole('region', { name: 'Getting started' });

describe('Getting started on Live', () => {
  it('shows each open step with a button to the control that does it', async () => {
    serve();
    renderPage(<GettingStarted />);
    const c = await card();
    expect(within(c).getByText('0 of 6 done. Live fills in as each one is ready.')).toBeInTheDocument();
    expect(within(c).getByRole('progressbar', { name: 'Setup progress' })).toHaveAttribute('aria-valuenow', '0');
    expect(within(c).getByRole('link', { name: 'Set your name' })).toHaveAttribute('href', '/settings#set-owner.name');
    expect(within(c).getByRole('link', { name: 'Record now' })).toHaveAttribute(
      'href',
      '/settings#set-owner.voice_ref',
    );
    expect(within(c).getByRole('link', { name: 'Open Who’s this?' })).toHaveAttribute('href', '/people');
    expect(within(c).getByRole('link', { name: 'Check models' })).toHaveAttribute('href', '/settings#machine');
    expect(within(c).getByRole('link', { name: 'Open This machine' })).toHaveAttribute('href', '/settings#machine');
    expect(within(c).getByRole('link', { name: 'Choose them' })).toHaveAttribute('href', '/settings#devices');
  });

  it('ticks what is done, with no button, and counts it', async () => {
    serve({ stats: READY, owner: 'Joe', voice: true });
    renderPage(<GettingStarted />);
    const c = await card();
    expect(within(c).getByText('4 of 6 done. Live fills in as each one is ready.')).toBeInTheDocument();
    const item = (title: string) => within(c).getByText(title).closest('li') as HTMLElement;
    expect(item('Name yourself')).toHaveTextContent("Name yourself (done)You're Joe");
    expect(within(c).queryByRole('link', { name: 'Set your name' })).not.toBeInTheDocument();
    expect(item('Tag your face')).toHaveTextContent('Tag your face (to do)Pick yourself in a camera view');
  });

  it('is gone once all six are done', async () => {
    serve({
      stats: READY,
      dev: devices({ cameras: [cam(0)], mics: [mic('USB Microphone')] }),
      owner: 'Joe',
      voice: true,
      people: [{ name: 'Joe', owner: true, faces: 4, voices: [1] }],
    });
    const { client } = renderPage(<GettingStarted />);
    // Every read answered, and still no card.
    await waitFor(() =>
      expect(
        client
          .getQueryCache()
          .getAll()
          .filter(q => q.state.status === 'success'),
      ).toHaveLength(5),
    );
    await waitFor(() => expect(client.isFetching()).toBe(0));
    expect(screen.queryByRole('region', { name: 'Getting started' })).not.toBeInTheDocument();
  });

  it('hides for now, and comes back once two reads agree that something done stopped being done', async () => {
    serve({ stats: READY });
    const { client } = renderPage(<GettingStarted />);
    await userEvent.click(within(await card()).getByRole('button', { name: 'Hide this for now' }));
    await waitFor(() => expect(screen.queryByRole('region', { name: 'Getting started' })).not.toBeInTheDocument());
    expect(usePrefs.getState().gettingStartedHidden).toBe('face,name,sources,voice');
    serve({ stats: { ...READY, ollama: { up: false, message: 'Start Ollama' } } }); // the models went away
    await client.refetchQueries({ queryKey: ['stats'] });
    await new Promise(r => setTimeout(r, 20));
    expect(screen.queryByRole('region', { name: 'Getting started' })).not.toBeInTheDocument(); // one read: not yet
    await client.refetchQueries({ queryKey: ['stats'] });
    expect(await card()).toBeInTheDocument();
  });

  it('stays away for good once all six were done, even if one opens again later', async () => {
    serve({
      stats: READY,
      dev: devices({ cameras: [cam(0)], mics: [mic('USB Microphone')] }),
      owner: 'Joe',
      voice: true,
      people: [{ name: 'Joe', owner: true, faces: 4, voices: [1] }],
    });
    const { client } = renderPage(<GettingStarted />);
    await waitFor(() => expect(usePrefs.getState().gettingStartedDone).toBe(true));
    serve({ stats: { ...READY, ollama: { up: false } } });
    await client.refetchQueries({ queryKey: ['stats'] });
    await client.refetchQueries({ queryKey: ['stats'] });
    expect(screen.queryByRole('region', { name: 'Getting started' })).not.toBeInTheDocument();
  });

  it('gives focus to the page heading when hidden, not to nothing', async () => {
    serve();
    renderPage(
      <>
        <h1>Live</h1>
        <GettingStarted />
      </>,
    );
    await userEvent.click(within(await card()).getByRole('button', { name: 'Hide this for now' }));
    expect(screen.getByRole('heading', { level: 1, name: 'Live' })).toHaveFocus();
  });
});
