import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http, type JsonBodyType } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { stop } from '../../lib/player';
import type { Card, Person } from '../api';
import { Profile } from './Profile';

const t0 = new Date(2026, 9, 28, 18, 41).getTime() / 1000;
const shot = (id: string | null, i: number) => ({
  id,
  url: id == null ? '/people/Dana/photos/1.jpg' : `/faces/shot/${id}.jpg`,
  ts: t0 - i * 86400,
  cam: id == null ? null : 'cam0',
  score: id == null ? null : 0.9,
});
const CARD: Card = {
  name: 'Dana',
  photo: shot('d30', 0),
  shots: [shot('d30', 0), shot('d31', 1)],
  videos: [
    {
      id: 'cam0@1793212345.50',
      cam: 'cam0',
      start: t0,
      end: t0 + 54,
      duration_s: 54,
      thumb: '/saved/coffee.jpg',
      what: 'Coffee',
    },
  ],
  voices: [{ line: 'usb-mic@1793212345.500', ts_start: t0, text: 'Morning all', mic: null }],
  last_seen: null,
  here_now: null,
};
const AFTER: Card = { ...CARD, photo: shot('d31', 1), shots: [shot('d31', 1)] };
const PEOPLE: Person[] = [
  { name: 'Dana', voices: [3] },
  { name: 'Eli', voices: [4] },
  { name: 'Luca', voices: [5] },
];

function serve(card: Card = CARD) {
  server.use(
    http.get('/people/:name', () => HttpResponse.json({ name: 'Dana', voices: [3] })),
    http.get('/people/:name/card', () => HttpResponse.json(card)),
    http.get('/clones', () => HttpResponse.json({ clones: [] })),
  );
}

function posts(path: string, reply: JsonBodyType, status = 200) {
  const bodies: unknown[] = [];
  server.use(
    http.post(path, async ({ request }) => {
      bodies.push(await request.json());
      return HttpResponse.json(reply, { status });
    }),
  );
  return bodies;
}

function show() {
  const props = {
    name: 'Dana',
    person: PEOPLE[0],
    people: PEOPLE,
    onChanged: vi.fn(),
    onDeleted: vi.fn(),
    onBack: vi.fn(),
  };
  renderPage(<Profile {...props} />);
  return props;
}

const photos = () => screen.findByRole('region', { name: 'Photos' });
/** Open the ⋯ menu of the item whose menu is labelled label, and pick item. */
async function itemMenu(label: string | RegExp, item: string) {
  await userEvent.click(await screen.findByRole('button', { name: label }));
  await userEvent.click(await screen.findByRole('menuitem', { name: item }));
}

beforeEach(() => {
  vi.useFakeTimers({ now: new Date(2026, 9, 28, 21, 5), toFake: ['Date'] }); // t0 is today, at 18:41
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
});
afterEach(() => {
  stop();
  vi.useRealTimers();
});

describe('Not Dana…', () => {
  it('moves a photo to someone else, fills in the next best shot, and undoes it', async () => {
    serve();
    const sent = posts('/people/correct', { ok: true, undo: 'tok-1', to: 'Eli', card: AFTER });
    const undone = posts('/people/undo', { ok: true, card: CARD });
    const props = show();
    await photos();
    await itemMenu(/^More for the photo from today/, 'Not Dana…');
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    expect(within(who).queryByRole('button', { name: 'Dana' })).not.toBeInTheDocument();
    await userEvent.click(within(who).getByRole('button', { name: 'Eli' }));

    expect(await screen.findByText('Moved to Eli')).toBeInTheDocument();
    expect(sent).toEqual([{ name: 'Dana', kind: 'shot', id: 'd30', action: 'move', to: 'Eli' }]);
    await waitFor(() =>
      expect(within(screen.getByRole('region', { name: 'Photos' })).getAllByRole('img')).toHaveLength(1),
    );
    expect(screen.getByRole('img', { name: 'Dana' })).toHaveAttribute('src', '/faces/shot/d31.jpg');
    expect(props.onChanged).toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(await screen.findByText('Undone')).toBeInTheDocument();
    expect(undone).toEqual([{ undo: 'tok-1' }]);
    await waitFor(() =>
      expect(within(screen.getByRole('region', { name: 'Photos' })).getAllByRole('img')).toHaveLength(2),
    );
  });

  it('finds people by search', async () => {
    serve();
    show();
    await photos();
    await itemMenu(/^More for the photo from today/, 'Not Dana…');
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    await userEvent.type(within(who).getByRole('textbox', { name: 'Search people' }), 'lu');
    expect(within(who).getByRole('button', { name: 'Luca' })).toBeInTheDocument();
    expect(within(who).queryByRole('button', { name: 'Eli' })).not.toBeInTheDocument();
    await userEvent.clear(within(who).getByRole('textbox', { name: 'Search people' }));
    await userEvent.type(within(who).getByRole('textbox', { name: 'Search people' }), 'zed');
    expect(within(who).getByText('Nobody matches “zed”.')).toBeInTheDocument();
  });

  it('moves a line to someone new, and says the name as the server saved it', async () => {
    serve();
    const sent = posts('/people/correct', { ok: true, undo: 'tok-2', to: 'Kara Ng', card: CARD });
    show();
    await itemMenu('More for the line “Morning all”', 'Not Dana…');
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    await userEvent.click(within(who).getByRole('button', { name: 'Someone new…' }));
    const field = within(who).getByRole('textbox', { name: "New person's name" });
    expect(within(who).getByRole('button', { name: 'Move' })).toBeDisabled();
    await userEvent.type(field, 'kara ng');
    await userEvent.click(within(who).getByRole('button', { name: 'Move to kara ng' }));
    expect(await screen.findByText('Moved to Kara Ng')).toBeInTheDocument();
    expect(sent).toEqual([
      { name: 'Dana', kind: 'voice', id: 'usb-mic@1793212345.500', action: 'move', to: 'kara ng' },
    ]);
  });

  it('takes it off Dana without saying who it is', async () => {
    serve();
    const sent = posts('/people/correct', { ok: true, undo: 'tok-3', card: CARD });
    show();
    await itemMenu(/^More for the video from today/, 'Not Dana…');
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    await userEvent.click(within(who).getByRole('button', { name: 'Not sure, just remove it from Dana' }));
    expect(await screen.findByText('Removed from Dana')).toBeInTheDocument();
    expect(sent).toEqual([{ name: 'Dana', kind: 'video', id: 'cam0@1793212345.50', action: 'remove' }]);
  });
});

describe('one correction at a time', () => {
  it('ignores a second correction while the first is on its way, so its Undo is never lost', async () => {
    serve();
    let release = () => {};
    const held = new Promise<void>(r => {
      release = r;
    });
    const sent: unknown[] = [];
    server.use(
      http.post('/people/correct', async ({ request }) => {
        sent.push(await request.json());
        await held;
        return HttpResponse.json({ ok: true, undo: 'tok-1', card: CARD });
      }),
    );
    show();
    await photos();
    await itemMenu(/^More for the photo from today/, 'Hide from the card');
    await itemMenu(/^More for the photo from yesterday/, 'Hide from the card');
    release();
    expect(await screen.findByText('Hidden from the card')).toBeInTheDocument();
    expect(sent).toEqual([{ name: 'Dana', kind: 'shot', id: 'd30', action: 'hide' }]);
  });
});

describe('Who is this? and focus', () => {
  it('gives focus back to the menu button when it closes', async () => {
    serve();
    show();
    await photos();
    const more = await screen.findByRole('button', { name: /^More for the photo from today/ });
    await itemMenu(/^More for the photo from today/, 'Not Dana…');
    await screen.findByRole('dialog', { name: 'Who is this?' });
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Who is this?' })).not.toBeInTheDocument());
    await waitFor(() => expect(more).toHaveFocus());
  });

  it('opens fresh each time, without the last search', async () => {
    serve();
    show();
    await photos();
    await itemMenu(/^More for the photo from today/, 'Not Dana…');
    let who = await screen.findByRole('dialog', { name: 'Who is this?' });
    await userEvent.type(within(who).getByRole('textbox', { name: 'Search people' }), 'lu');
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Who is this?' })).not.toBeInTheDocument());
    await itemMenu(/^More for the photo from yesterday/, 'Not Dana…');
    who = await screen.findByRole('dialog', { name: 'Who is this?' });
    expect(within(who).getByRole('textbox', { name: 'Search people' })).toHaveValue('');
    expect(within(who).getByText("It's not Dana in the photo from yesterday at 18:41. Who is it?")).toBeInTheDocument();
  });
});

describe('the other corrections', () => {
  it('uses a photo as the main photo', async () => {
    serve();
    const sent = posts('/people/correct', { ok: true, undo: 'tok-4', card: { ...CARD, photo: shot('d31', 1) } });
    show();
    await photos();
    await itemMenu(/^More for the photo from yesterday/, 'Use as main photo');
    expect(await screen.findByText('Now the main photo')).toBeInTheDocument();
    expect(sent).toEqual([{ name: 'Dana', kind: 'shot', id: 'd31', action: 'main' }]);
    await waitFor(() =>
      expect(screen.getByRole('img', { name: 'Dana' })).toHaveAttribute('src', '/faces/shot/d31.jpg'),
    );
  });

  it('offers Use as main photo on photos only', async () => {
    serve();
    show();
    await userEvent.click(await screen.findByRole('button', { name: /^More for the video from today/ }));
    const items = (await screen.findAllByRole('menuitem')).map(i => i.textContent);
    expect(items).toEqual(['Not Dana…', 'Hide from the card']);
  });

  it('hides a line from the card', async () => {
    serve();
    const sent = posts('/people/correct', { ok: true, undo: 'tok-5', card: { ...CARD, voices: [] } });
    show();
    await itemMenu('More for the line “Morning all”', 'Hide from the card');
    expect(await screen.findByText('Hidden from the card')).toBeInTheDocument();
    expect(sent).toEqual([{ name: 'Dana', kind: 'voice', id: 'usb-mic@1793212345.500', action: 'hide' }]);
    expect(await screen.findByText('Nothing heard from Dana yet.')).toBeInTheDocument();
  });

  it("shows the server's refusal, and leaves the card as it was", async () => {
    serve();
    posts('/people/correct', { ok: false, error: 'That shot is already gone.' });
    show();
    await photos();
    await itemMenu(/^More for the photo from today/, 'Hide from the card');
    expect(await screen.findByText('That shot is already gone.')).toBeInTheDocument();
    expect(within(await photos()).getAllByRole('img')).toHaveLength(2);
    expect(screen.queryByRole('button', { name: 'Undo' })).not.toBeInTheDocument();
  });

  it('says what could not be put back when an undo only partly works, and shows what did', async () => {
    serve();
    posts('/people/correct', { ok: true, undo: 'tok-7', card: AFTER });
    posts('/people/undo', { ok: false, errors: ["The face match couldn't be refit."], people: [], card: CARD });
    show();
    await photos();
    await itemMenu(/^More for the photo from today/, 'Hide from the card');
    await userEvent.click(await screen.findByRole('button', { name: 'Undo' }));
    expect(await screen.findByText("Partly undone: The face match couldn't be refit.")).toBeInTheDocument();
    await waitFor(() =>
      expect(within(screen.getByRole('region', { name: 'Photos' })).getAllByRole('img')).toHaveLength(2),
    );
  });

  it('says when an undo fails', async () => {
    serve();
    posts('/people/correct', { ok: true, undo: 'tok-6', card: CARD });
    posts('/people/undo', { ok: false, error: 'Already undone.' }, 409);
    show();
    await photos();
    await itemMenu(/^More for the photo from today/, 'Hide from the card');
    await userEvent.click(await screen.findByRole('button', { name: 'Undo' }));
    expect(await screen.findByText('Already undone.')).toBeInTheDocument();
  });

  it('has no menu on a saved crop standing in for recorded shots', async () => {
    serve({ ...CARD, photo: shot(null, 0), shots: [shot(null, 0)] });
    show();
    expect(within(await photos()).getAllByRole('img')).toHaveLength(1);
    expect(screen.queryByRole('button', { name: /^More for the photo/ })).not.toBeInTheDocument();
  });
});

describe('why this is Dana', () => {
  const why = (q: string) => ({
    name: 'Dana',
    subject: q.startsWith('shot') ? 'shot' : 'line',
    via: q.startsWith('shot') ? 'face' : 'voice',
    basis: q.startsWith('shot') ? null : 'owner_tag',
    asserted_at: t0 - 86400,
    source: 'dashboard',
    confidence: 1,
    similarity: q.startsWith('shot') ? 0.62 : 0.71,
    threshold: q.startsWith('shot') ? 0.45 : 0.55,
    voice: null,
    mark: {},
  });
  let asked: string[];
  beforeEach(() => {
    asked = [];
    server.use(
      http.get('/identity/why', ({ request }) => {
        const q = new URL(request.url).searchParams.toString();
        asked.push(q);
        return HttpResponse.json(why(q));
      }),
    );
  });

  it("says why a photo is Dana's, by its sighting, with Not Dana… beside it", async () => {
    serve();
    show();
    await itemMenu(/^More for the photo from today/, 'Why this is Dana');
    await vi.waitFor(() => expect(asked).toEqual(['shot=d30'])); // asked by its sighting
    const pop = await vi.waitFor(() => screen.getByRole('dialog', { name: 'Why Dana' }));
    expect(await within(pop).findByText("The face was recognized as Dana's when it was seen.")).toBeInTheDocument();
    expect(within(pop).getByText('Face match 0.62 (a match needs 0.45).')).toBeInTheDocument();
    await userEvent.click(within(pop).getByRole('button', { name: 'Not Dana…' }));
    expect(await screen.findByRole('dialog', { name: 'Who is this?' })).toBeInTheDocument();
  });

  it('says why a line is Dana, by its key', async () => {
    serve();
    show();
    await itemMenu('More for the line “Morning all”', 'Why this is Dana');
    await screen.findByText(/^Its voice was named Dana on /);
    expect(asked).toEqual([`line=${encodeURIComponent('usb-mic@1793212345.500')}`]);
  });

  it('has no Why for a video, which holds more than one face', async () => {
    serve();
    show();
    await userEvent.click(await screen.findByRole('button', { name: /^More for the video from today/ }));
    expect(await screen.findByRole('menuitem', { name: 'Not Dana…' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Why this is Dana' })).not.toBeInTheDocument();
  });
});
