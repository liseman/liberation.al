import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { server } from '../../test/server';
import { hms } from './format';
import { lineKey, play, playbackKeyBlocked, playLine, seek, setPlayingTs, stop, toggle, usePlayer } from './player';

let playResult: () => Promise<void>;
const audio = () => document.querySelector('audio') as HTMLAudioElement;
const state = () => usePlayer.getState();
const fire = (type: string) => audio().dispatchEvent(new Event(type));

beforeEach(() => {
  playResult = () => Promise.resolve();
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) {
    Object.defineProperty(this, 'paused', { value: false, configurable: true });
    return playResult();
  });
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) {
    Object.defineProperty(this, 'paused', { value: true, configurable: true });
  });
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
});
afterEach(() => stop());

describe('play and stop', () => {
  it('plays a line, then stops it when the same line is played again', () => {
    playLine({ ts_start: 1700000000.5 });
    expect(state()).toEqual({
      key: 'ts:1700000000.5',
      ts: 1700000000.5,
      paused: false,
      status: 'Loading',
      position: 0,
      duration: null,
    });
    expect(audio().getAttribute('src')).toBe('/clip?ts_start=1700000000.5');
    playLine({ ts_start: 1700000000.5 });
    expect(state()).toEqual({ key: null, ts: null, paused: false, status: '', position: 0, duration: null });
    expect(audio().hasAttribute('src')).toBe(false);
  });

  it('switches to another clip without stopping first', () => {
    play('/a.wav', 'a');
    play('/b.wav', 'b');
    expect(state().key).toBe('b');
    expect(audio().getAttribute('src')).toBe('/b.wav');
  });

  it('says how long the clip is once it knows', () => {
    play('/a.wav', 'a', 1700000000);
    Object.defineProperty(audio(), 'duration', { value: 3.24, configurable: true });
    fire('loadedmetadata');
    expect(state().status).toBe(`${hms(1700000000)}, 3.2 s`);
  });

  it('stops by itself at the end', () => {
    play('/a.wav', 'a');
    fire('ended');
    expect(state().key).toBeNull();
  });

  it('moves the highlight only for what is playing', () => {
    play('/a.wav', 'a', 1);
    setPlayingTs('other', 5);
    setPlayingTs('a', 9);
    expect(state().ts).toBe(9);
  });
});

describe("when a clip can't play", () => {
  it('says autoplay was blocked', async () => {
    playResult = () => Promise.reject(new DOMException('no', 'NotAllowedError'));
    play('/a.wav', 'a');
    await vi.waitFor(() => expect(state().status).toBe("Can't play this: the browser blocked autoplay; click again"));
    expect(state().key).toBeNull();
  });

  it("shows the server's reason when the clip fails to load", async () => {
    server.use(http.get('/clip', () => HttpResponse.json({ error: 'That audio was deleted.' }, { status: 404 })));
    playLine({ ts_start: 42 });
    fire('error');
    await vi.waitFor(() => expect(state().status).toBe("Can't play this: That audio was deleted."));
    expect(state().key).toBeNull();
  });
});

describe('Space', () => {
  const space = (target: EventTarget = document.body) => {
    const e = new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true });
    target.dispatchEvent(e);
    return e;
  };

  it('pauses and resumes what is playing', () => {
    play('/a.wav', 'a');
    expect(space().defaultPrevented).toBe(true);
    expect(audio().paused).toBe(true);
    fire('pause');
    expect(state().paused).toBe(true);
    space();
    expect(audio().paused).toBe(false);
  });

  it('types a space in a text field instead', () => {
    play('/a.wav', 'a');
    const input = document.createElement('input');
    document.body.append(input);
    expect(space(input).defaultPrevented).toBe(false);
    expect(audio().paused).toBe(false);
    input.remove();
  });

  it('does not toggle a second time when another handler consumed Space', () => {
    play('/a.wav', 'a');
    const e = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
    e.preventDefault();
    document.body.dispatchEvent(e);
    expect(audio().paused).toBe(false);
  });

  it('leaves an open modal alone, including a focusable dialog container', () => {
    play('/a.wav', 'a');
    const dialog = document.body.appendChild(document.createElement('dialog'));
    dialog.open = true;
    expect(space(dialog).defaultPrevented).toBe(false);
    expect(audio().paused).toBe(false);
    dialog.remove();
  });

  it('leaves nested editable text and modified Space alone', () => {
    play('/a.wav', 'a');
    const editable = document.body.appendChild(document.createElement('div'));
    editable.setAttribute('contenteditable', 'true');
    const child = editable.appendChild(document.createElement('span'));
    expect(space(child).defaultPrevented).toBe(false);
    const e = new KeyboardEvent('keydown', { key: ' ', ctrlKey: true, bubbles: true, cancelable: true });
    document.body.dispatchEvent(e);
    expect(e.defaultPrevented).toBe(false);
    expect(audio().paused).toBe(false);
    editable.remove();
  });

  it('does nothing when nothing is playing', () => {
    expect(toggle()).toBe(false);
  });
});

describe('the playback guard', () => {
  it('asks the server to ignore the room mic while a clip plays, and ends the window when it pauses', async () => {
    const calls: unknown[] = [];
    server.use(
      http.post('/playback/start', async ({ request }) => {
        calls.push(['start', await request.json()]);
        return HttpResponse.json({ ok: true, token: 't1' });
      }),
      http.post('/playback/stop', async ({ request }) => {
        calls.push(['stop', await request.json()]);
        return HttpResponse.json({ ok: true });
      }),
    );
    play('/a.wav', 'a');
    Object.defineProperty(audio(), 'duration', { value: 25, configurable: true });
    Object.defineProperty(audio(), 'currentTime', { value: 5, configurable: true });
    fire('playing');
    await vi.waitFor(() => expect(calls).toHaveLength(1));
    await new Promise(r => setTimeout(r, 10)); // the token arrives
    fire('pause');
    await vi.waitFor(() => expect(calls).toHaveLength(2));
    expect(calls).toEqual([
      ['start', { seconds: 20 }],
      ['stop', { token: 't1' }],
    ]);
  });
});

it("plays a line by its key when it has one, so not the other microphone's (#247)", () => {
  playLine({ ts_start: 1700000000.5, line: 'mic-a@1700000000.500' });
  expect(state().key).toBe('line:mic-a@1700000000.500');
  expect(lineKey({ ts_start: 1700000000.5 })).toBe('ts:1700000000.5');
  stop();
});

it('lets Review search buttons own Space before the player capture handler runs', () => {
  const search = document.body.appendChild(document.createElement('section'));
  search.setAttribute('data-review-search', '');
  const button = search.appendChild(document.createElement('button'));
  const event = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
  const capture = vi.fn((key: KeyboardEvent) => playbackKeyBlocked(key));
  document.addEventListener('keydown', capture, true);
  try {
    button.dispatchEvent(event);
    expect(capture).toHaveBeenCalledOnce();
    expect(capture).toHaveReturnedWith(true);
    expect(event.defaultPrevented).toBe(false);
  } finally {
    document.removeEventListener('keydown', capture, true);
    search.remove();
  }
});

it.each([
  [']', false],
  ['ArrowRight', false],
  [' ', true],
  ['Enter', true],
])(
  'lets %s through from a checkbox in a [data-review-keys] area only when it is not the toggle key (#2287)',
  (key, blocked) => {
    const area = document.body.appendChild(document.createElement('div'));
    area.setAttribute('data-review-keys', '');
    const pill = area.appendChild(document.createElement('input'));
    pill.type = 'checkbox';
    const plain = document.body.appendChild(document.createElement('input'));
    plain.type = 'checkbox';
    const seen: boolean[] = [];
    const capture = (e: KeyboardEvent) => seen.push(playbackKeyBlocked(e));
    document.addEventListener('keydown', capture);
    try {
      pill.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
      plain.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
      expect(seen).toEqual([blocked, true]); // a checkbox outside such an area always keeps its keys
    } finally {
      document.removeEventListener('keydown', capture);
      area.remove();
      plain.remove();
    }
  },
);

it.each([
  ['checkbox', ' ', false],
  ['radio', ' ', false],
  ['range', ' ', false],
  ['text', ' ', true],
  ['search', ' ', true],
  ['radio', 'ArrowRight', true], // a radio group keeps its arrows
])('over controls, a focused %s input with %j blocks playback: %s (#2352)', (type, key, blocked) => {
  const input = document.body.appendChild(document.createElement('input'));
  input.type = type;
  const seen: boolean[] = [];
  const capture = (e: KeyboardEvent) => seen.push(playbackKeyBlocked(e, { spaceOverControls: true }));
  document.addEventListener('keydown', capture);
  try {
    input.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
    expect(seen).toEqual([blocked]);
  } finally {
    document.removeEventListener('keydown', capture);
    input.remove();
  }
});

it('leaves a textarea its Space over controls too', () => {
  const area = document.body.appendChild(document.createElement('textarea'));
  const seen: boolean[] = [];
  const capture = (e: KeyboardEvent) => seen.push(playbackKeyBlocked(e, { spaceOverControls: true }));
  document.addEventListener('keydown', capture);
  try {
    area.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
    expect(seen).toEqual([true]);
  } finally {
    document.removeEventListener('keydown', capture);
    area.remove();
  }
});

describe('where it is (Calls, #2621 board)', () => {
  it('follows the audio as it plays, seeks within what is playing, and leaves another key alone', () => {
    play('/calls/audio?x=1', 'call:a');
    const a = audio();
    Object.defineProperty(a, 'duration', { value: 182, configurable: true });
    a.dispatchEvent(new Event('loadedmetadata'));
    Object.defineProperty(a, 'currentTime', { value: 42, writable: true, configurable: true }); // others pin it
    a.dispatchEvent(new Event('timeupdate'));
    expect(state()).toMatchObject({ position: 42, duration: 182 });
    seek('call:a', 90);
    expect(state().position).toBe(90);
    seek('call:b', 10); // not what's playing
    expect(state().position).toBe(90);
    stop();
    expect(state()).toMatchObject({ position: 0, duration: null });
  });
});
