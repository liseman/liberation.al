import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { newQueryClient, Providers } from '../../shared/Providers';
import { server } from '../../test/server';
import { stop, usePlayer } from '../lib/player';
import { liveKey } from '../live/feed';
import { reduce } from '../live/reduce';
import type { TranscriptEntry } from '../live/types';
import type { Turn } from '../transcript/model';
import { sameTurn } from '../transcript/model';
import { InlineName } from '../transcript/parts';
import { HEAD_PX, LINE_PX } from '../transcript/Transcript';
import { refreshNames } from './api';
import { Transcript } from './Transcript';

const T0 = 1_793_000_000;
const LINES: TranscriptEntry[] = [
  { ts_start: T0, ts_end: T0 + 2, text: 'hello there', speaker_id: 1, speaker: 'Dana', line: `mic@${T0}.000` },
  {
    ts_start: T0 + 10,
    ts_end: T0 + 12,
    text: 'sunny weather',
    speaker_id: 2,
    speaker: 'Speaker 2',
    line: `mic@${T0 + 10}.000`,
  },
  {
    ts_start: T0 + 20,
    ts_end: T0 + 22,
    text: 'who is this',
    speaker_id: null,
    speaker: 'unknown',
    line: `mic@${T0 + 20}.000`,
  },
];
const SPEAKERS = {
  speakers: [
    { id: 1, label: 'Dana' },
    { id: 2, label: 'Speaker 2' },
  ],
  merged: {},
};

let posts: [string, unknown][];
let asked: string[]; // the lines Why this name was asked about

// Why the first line is Dana: its voice was named in People, and it matched that voice at 0.71.
const WHY = {
  name: 'Dana',
  subject: 'line',
  via: 'voice',
  basis: 'owner_tag',
  asserted_at: T0 - 86400,
  source: 'dashboard',
  confidence: 1,
  similarity: 0.71,
  threshold: 0.55,
  voice: { id: 1, label: 'Dana' },
  mark: { kind: 'line', ts_start: T0, ts_end: T0 + 2 },
};

function renderTranscript(lines = LINES) {
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false } });
  client.setQueryData(liveKey('speakers'), reduce('speakers', undefined, SPEAKERS));
  client.setQueryData(liveKey('transcript'), reduce('transcript', undefined, { entries: lines, reset: true }));
  const view = render(
    <Providers client={client} test>
      <Transcript listening />
    </Providers>,
  );
  return { client, ...view };
}

beforeEach(() => {
  posts = [];
  asked = [];
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Luca', cams: ['cam0'], hotkeys: false });
  server.use(
    http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }] })),
    http.post('/people/add', async ({ request }) => HttpResponse.json(await request.json())),
    http.get('/speakers', () => HttpResponse.json(SPEAKERS)),
    http.get('/transcript', () => HttpResponse.json({ entries: LINES, size: 1, reset: true })),
    http.get('/identity/why', ({ request }) => {
      const q = new URL(request.url).searchParams;
      asked.push(q.get('line') ?? q.get('ts_start') ?? '');
      return HttpResponse.json(WHY);
    }),
    ...['speakers/label', 'speakers/tag', 'people/claim', 'transcript/unlabel', 'measurement/mark'].map(path =>
      http.post(`/${path}`, async ({ request }) => {
        posts.push([path, await request.json()]);
        return HttpResponse.json({ ok: true, speakers: SPEAKERS });
      }),
    ),
    http.post('/people/correct', async ({ request }) => {
      posts.push(['people/correct', await request.json()]);
      return HttpResponse.json({ ok: true, undo: `u${posts.length}` });
    }),
  );
  return () => {
    stop();
    vi.unstubAllGlobals();
  };
});

/** In Who is this?, which a Not <name> opens: just take it off the voice it was put down to. */
async function justRemove(from: string) {
  const who = await screen.findByRole('dialog', { name: 'Who is this?' });
  await userEvent.click(within(who).getByRole('button', { name: `Not sure, just remove it from ${from}` }));
}
const removed = (name: string, ...ids: string[]) =>
  ids.map(id => ['people/correct', { name, kind: 'voice', id, action: 'remove' }]);

const badges = () => screen.getAllByRole('button', { name: /^(Rename|Why|.*: say who this is)/ });
/** Rename turn i: a named voice's name opens Why this name, and Rename this voice… is in it. */
async function rename(i: number, name: string) {
  const badge = badges()[i] as HTMLElement;
  await userEvent.click(badge);
  if (badge.getAttribute('aria-haspopup'))
    await userEvent.click(await screen.findByRole('button', { name: /^(Rename this voice|Say who this is)…$/ }));
  const box = await screen.findByRole('combobox', { name: 'Speaker name' });
  expect(box).toHaveFocus();
  await userEvent.clear(box);
  await userEvent.type(box, `${name}{Enter}`);
  // Naming a voice nobody has named with a new name asks to add them first.
  const add = screen.queryByRole('button', { name: 'Add' });
  if (add) await userEvent.click(add);
}

describe('name suggestions', () => {
  const editor = async () => {
    const onDone = vi.fn();
    server.use(http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dan' }, { name: 'Dana' }] })));
    render(
      <Providers client={newQueryClient()} test>
        <InlineName value="" onDone={onDone} />
      </Providers>,
    );
    return { input: await screen.findByRole('combobox', { name: 'Speaker name' }), onDone };
  };

  it('commits a clicked suggestion once, using its full name rather than the draft', async () => {
    const { input, onDone } = await editor();
    await userEvent.type(input, 'Da');
    await userEvent.click(await screen.findByRole('option', { name: 'Dana' }));
    fireEvent.blur(input);
    expect(onDone).toHaveBeenCalledExactlyOnceWith('Dana');
  });

  it('commits a highlighted suggestion with Enter rather than the draft', async () => {
    const { input, onDone } = await editor();
    await userEvent.type(input, 'Da');
    await screen.findByRole('option', { name: 'Dana' });
    await userEvent.keyboard('{ArrowDown}{ArrowDown}{Enter}');
    fireEvent.blur(input);
    expect(onDone).toHaveBeenCalledExactlyOnceWith('Dana');
  });

  it('keeps typed exact matches editable and saves free text with Enter once', async () => {
    const { input, onDone } = await editor();
    await userEvent.type(input, 'Dan');
    expect(onDone).not.toHaveBeenCalled();
    await userEvent.type(input, 'a Ruiz{Enter}');
    fireEvent.blur(input);
    expect(onDone).toHaveBeenCalledExactlyOnceWith('Dana Ruiz');
  });

  it('saves trimmed free text on blur', async () => {
    const { input, onDone } = await editor();
    await userEvent.type(input, '  Dane  ');
    await userEvent.tab();
    expect(onDone).toHaveBeenCalledExactlyOnceWith('Dane');
  });

  it('does not save composing input, and Escape cancels without a blur save', async () => {
    const { input, onDone } = await editor();
    fireEvent.change(input, { target: { value: 'Dana' } });
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter', isComposing: true });
    expect(onDone).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: 'Escape' });
    fireEvent.blur(input);
    expect(onDone).toHaveBeenCalledExactlyOnceWith(null);
  });
});

describe('naming a voice from its badge', () => {
  it('names voices and queues only the visible turn lines for confirmation', async () => {
    renderTranscript();
    await screen.findByText('hello there');
    await vi.waitFor(() => expect(posts).toEqual([])); // the people list has loaded
    await rename(0, 'Zed');
    await vi.waitFor(() =>
      expect(posts).toEqual([
        ['speakers/label', { id: 1, name: 'Zed' }],
        ['speakers/tag', { lines: [`mic@${T0}.000`], name: 'Zed' }],
      ]),
    );
    posts.length = 0;
    await rename(1, 'Eli');
    await vi.waitFor(() =>
      expect(posts).toEqual([
        ['people/claim', { kind: 'voice', id: 2, name: 'Eli' }],
        ['speakers/tag', { lines: [`mic@${T0 + 10}.000`], name: 'Eli' }],
      ]),
    );
    posts.length = 0;
    await rename(2, 'Dana');
    await vi.waitFor(() => expect(posts).toEqual([['speakers/tag', { lines: [`mic@${T0 + 20}.000`], name: 'Dana' }]]));
    expect(
      within(screen.getByText('who is this').closest('article') as HTMLElement).getByRole('status'),
    ).toHaveTextContent('Tagging as Dana');
  });

  it('shows a name given to an unmatched turn at once, before its lines come back with it', async () => {
    // The microphone service relabels tagged lines a while later, and the voices are read again slower still.
    server.use(
      http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }, { name: 'Carly' }] })),
      http.get('/speakers', () => new Promise<never>(() => {})),
      http.post('/people/claim', () => HttpResponse.json({ ok: true, name: 'Carly', undo: 'u1', people: [] })),
    );
    renderTranscript();
    await screen.findByText('hello there');
    await new Promise(r => setTimeout(r, 20)); // the people list has loaded
    const turnOf = (text: string) => screen.getByText(text).closest('article') as HTMLElement;
    // A voice: claimed, and the claim's reply has no voices in it, as the server's doesn't.
    await rename(1, 'Carly');
    await vi.waitFor(() =>
      expect(within(turnOf('sunny weather')).getByRole('button', { name: 'Why Carly' })).toBeInTheDocument(),
    );
    expect(turnOf('sunny weather')).not.toHaveAttribute('data-unmatched');
    // Lines with no voice: tagged, and shown with the name while the tag is pending.
    await userEvent.click(within(turnOf('who is this')).getByRole('button', { name: 'Unmatched: say who this is' }));
    await userEvent.type(screen.getByRole('combobox', { name: 'Speaker name' }), 'Carly{Enter}');
    const pending = turnOf('who is this');
    await vi.waitFor(() => expect(within(pending).getByRole('status')).toHaveTextContent('Tagging as Carly'));
    expect(within(pending).getByRole('button', { name: 'Carly: say who this is' })).toHaveTextContent(/^Carly$/);
    expect(pending.querySelector('[aria-hidden]')).toHaveTextContent(/^C$/);
    expect(pending).not.toHaveAttribute('data-unmatched');
  });

  it('tags just the lines when a named voice is given someone else who exists (heard wrong, no merge)', async () => {
    server.use(http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }, { name: 'Eli' }] })));
    renderTranscript([LINES[0] as TranscriptEntry]);
    await screen.findByText('hello there');
    await new Promise(r => setTimeout(r, 20)); // the people list has loaded
    await rename(0, 'Eli');
    await vi.waitFor(() => expect(posts).toEqual([['speakers/tag', { lines: [`mic@${T0}.000`], name: 'Eli' }]]));
  });

  it('saves nothing when Escape cancels', async () => {
    renderTranscript();
    await userEvent.click(
      (await screen.findAllByRole('button', { name: 'Unmatched: say who this is' }))[0] as HTMLElement,
    );
    const box = await screen.findByRole('combobox', { name: 'Speaker name' });
    await userEvent.type(box, 'Nope{Escape}');
    fireEvent.blur(box);
    await new Promise(r => setTimeout(r, 20));
    expect(posts).toEqual([]);
    expect(screen.queryByRole('combobox', { name: 'Speaker name' })).not.toBeInTheDocument();
  });

  it('shows why a rename failed and unlocks the badge', async () => {
    server.use(http.post('/speakers/label', () => HttpResponse.json({ ok: false, error: 'locked' }, { status: 409 })));
    renderTranscript();
    await rename(0, 'Zed');
    expect(await screen.findByRole('alert')).toHaveTextContent("That didn't work: locked");
    expect(screen.getByRole('button', { name: 'Why Dana' })).toBeEnabled();
  });

  it('asks who it is with Not <name>, as Review does, and can take the turn off them', async () => {
    const corrected: unknown[] = [];
    server.use(
      http.post('/people/correct', async ({ request }) => {
        corrected.push(await request.json());
        return HttpResponse.json({ ok: true, undo: 'u1' });
      }),
    );
    renderTranscript();
    await userEvent.click(await screen.findByRole('button', { name: 'Not Dana' }));
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    expect(within(who).getByText("It's not Dana in this line. Who is it?")).toBeInTheDocument();
    await userEvent.click(within(who).getByRole('button', { name: 'Not sure, just remove it from Dana' }));
    await vi.waitFor(() =>
      expect(corrected).toEqual([{ name: 'Dana', kind: 'voice', id: `mic@${T0}.000`, action: 'remove' }]),
    );
    expect(await screen.findByRole('button', { name: 'Undo' })).toBeInTheDocument();
  });

  it('leaves the tagging state once the tagged lines come back named', async () => {
    const { client } = renderTranscript([LINES[2] as TranscriptEntry]);
    await rename(0, 'Erin');
    expect(await screen.findByText(/Tagging as Erin/)).toBeInTheDocument();
    act(() => {
      client.setQueryData(
        liveKey('transcript'),
        reduce('transcript', client.getQueryData(liveKey('transcript')), {
          entries: [{ ...LINES[2], speaker: 'Erin', speaker_id: null, tagged: true }],
          replace: true,
        }),
      );
    });
    await vi.waitFor(() => expect(screen.queryByText(/Tagging as/)).not.toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Why Erin' })).toHaveTextContent('Erin'); // named now: it says why
  });
});

describe('why this name', () => {
  const TURN = [
    LINES[0] as TranscriptEntry,
    {
      ...(LINES[0] as TranscriptEntry),
      ts_start: T0 + 3,
      ts_end: T0 + 4,
      text: 'and you?',
      line: 'kitchen@1793000003.000',
      device: 'Kitchen',
    },
  ];

  it("says why a named voice's turn has its name, from the turn's first line", async () => {
    const { client } = renderTranscript(TURN);
    await userEvent.click(await screen.findByRole('button', { name: 'Why Dana' }));
    const why = await screen.findByRole('dialog', { name: 'Why Dana' });
    expect(await within(why).findByText(/^Its voice was named Dana on /)).toBeInTheDocument();
    expect(within(why).getByText('Voice match 0.71 (a match needs 0.55).')).toBeInTheDocument();
    expect(asked).toEqual([`mic@${T0}.000`]); // where the turn's name comes from
    act(() => {
      client.setQueryData(
        liveKey('transcript'),
        reduce('transcript', client.getQueryData(liveKey('transcript')), {
          entries: [{ ts_start: T0 + 6, ts_end: T0 + 7, text: 'one more', speaker_id: 1, speaker: 'Dana' }],
        }),
      );
    });
    await screen.findByText('one more');
    expect(asked).toEqual([`mic@${T0}.000`]); // a new line in the turn doesn't change what it explains
  });

  it('marks the name right without changing it', async () => {
    renderTranscript(TURN);
    await userEvent.click(await screen.findByRole('button', { name: 'Why Dana' }));
    const why = await screen.findByRole('dialog', { name: 'Why Dana' });
    await userEvent.click(await within(why).findByRole('button', { name: 'Yes' }));
    expect(await within(why).findByRole('status')).toHaveTextContent('Marked right');
    expect(posts).toEqual([['measurement/mark', { ...WHY.mark, shown: 'Dana', right: true }]]);
  });

  it('asks who it is from Why this name too, about every line of the turn', async () => {
    renderTranscript(TURN);
    await userEvent.click(await screen.findByRole('button', { name: 'Why Dana' }));
    const why = await screen.findByRole('dialog', { name: 'Why Dana' });
    await userEvent.click(await within(why).findByRole('button', { name: 'Not Dana' }));
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    expect(within(who).getByText("It's not Dana in this turn. Who is it?")).toBeInTheDocument();
  });

  it('closes with Escape, back on the name', async () => {
    renderTranscript(TURN);
    const name = await screen.findByRole('button', { name: 'Why Dana' });
    await userEvent.click(name);
    await screen.findByRole('dialog', { name: 'Why Dana' });
    await userEvent.keyboard('{Escape}');
    await vi.waitFor(() => expect(screen.queryByRole('dialog', { name: 'Why Dana' })).not.toBeInTheDocument());
    expect(name).toHaveFocus();
  });
});

describe('voices nobody has named', () => {
  const turnOf = (text: string) => screen.getByText(text).closest('article') as HTMLElement;

  it('shows a voice no person matched as Unmatched with "?", never its placeholder', async () => {
    renderTranscript();
    await screen.findByText('hello there');
    for (const text of ['sunny weather', 'who is this']) {
      const turn = turnOf(text);
      expect(within(turn).getByRole('button', { name: 'Unmatched: say who this is' })).toHaveTextContent(/^Unmatched$/);
      expect(turn.querySelector('[aria-hidden]')).toHaveTextContent(/^\?$/); // the avatar
      expect(turn).toHaveAttribute('data-unmatched');
      expect(turn).not.toHaveTextContent(/Speaker 2|unknown/i);
    }
    expect(within(turnOf('hello there')).getByRole('button', { name: 'Why Dana' })).toHaveTextContent('Dana');
    expect(turnOf('hello there')).not.toHaveAttribute('data-unmatched');
    // A voice is named from its name: no separate Name this voice button.
    expect(screen.queryByRole('button', { name: 'Name this voice' })).not.toBeInTheDocument();
  });

  it('puts Not this voice first in the quiet actions, and it takes the turn off that voice', async () => {
    renderTranscript();
    await screen.findByText('hello there');
    await vi.waitFor(() => expect(posts).toEqual([]));
    const not = within(turnOf('sunny weather')).getByRole('button', { name: 'Not this voice' });
    expect(not.previousElementSibling).toBeNull(); // first in its group
    expect(not.nextElementSibling).toHaveAccessibleName('Not transcribed correctly');
    // A line no voice was matched to has no voice to take it off, but can still be rejected as a bad transcription.
    const unknown = within(turnOf('who is this')).getByRole('button', { name: 'Not transcribed correctly' });
    expect(unknown.previousElementSibling).toBeNull();
    await userEvent.click(not);
    // The same Who is this? as a person's, by the voice's label, with an Undo.
    const who = await screen.findByRole('dialog', { name: 'Who is this?' });
    expect(within(who).getByText("It's not the unmatched voice in this line. Who is it?")).toBeInTheDocument();
    await justRemove('the unmatched voice');
    await vi.waitFor(() => expect(posts).toEqual(removed('Speaker 2', `mic@${T0 + 10}.000`)));
    expect(await screen.findByText('Removed from the unmatched voice')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Undo' })).toBeInTheDocument();
  });

  it("puts the voice's actions beside its name, before the lines, as quiet icons", async () => {
    renderTranscript();
    await screen.findByText('hello there');
    const turn = turnOf('sunny weather');
    const head = within(turn).getByRole('banner');
    const act = within(head).getByRole('button', { name: 'Not transcribed correctly' });
    expect(act.parentElement).toContainElement(within(head).getByRole('button', { name: 'Not this voice' })); // one quiet group
    expect(act.textContent).toBe(''); // an icon, its label in aria-label and its tooltip
    // in the header after the time, so it comes before the lines in reading and tab order
    expect(head.querySelector('time')?.compareDocumentPosition(act)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it("doesn't repeat Not <voice> on a one-line turn's line: the header covers it", async () => {
    renderTranscript();
    await screen.findByText('hello there');
    const turn = turnOf('sunny weather'); // Speaker 2, one line
    expect(
      within(turn)
        .getAllByRole('button', { name: /^Not this voice$/ })
        .map(b => b.getAttribute('aria-label') ?? b.textContent),
    ).toEqual(['Not this voice']);
  });

  it('keeps actions in the header and removes every visible line from that voice', async () => {
    renderTranscript([
      { ts_start: T0 + 10, ts_end: T0 + 11, text: 'first', speaker_id: 2, speaker: 'Speaker 2', line: 'mic@first' },
      { ts_start: T0 + 12, ts_end: T0 + 13, text: 'second', speaker_id: 2, speaker: 'Speaker 2', line: 'mic@second' },
    ]);
    const turn = (await screen.findByText('first')).closest('article') as HTMLElement;
    expect(within(turn).getAllByRole('button', { name: /^Not this voice$/ })).toHaveLength(1);
    expect(within(turn).getAllByRole('button', { name: /^Not this voice in the line at / })).toHaveLength(2);
    expect(within(turn).queryByRole('button', { name: /Confirm/ })).not.toBeInTheDocument();
    await userEvent.click(within(turn).getByRole('button', { name: 'Not this voice' }));
    await justRemove('the unmatched voice');
    await vi.waitFor(() => expect(posts).toEqual(removed('Speaker 2', 'mic@first', 'mic@second')));
  });

  it('takes only the selected line off an unmatched voice, as off a person, with the person-minus beside its time', async () => {
    renderTranscript([
      { ts_start: T0 + 10, ts_end: T0 + 11, text: 'first', speaker_id: 2, speaker: 'Speaker 2', line: 'mic@first' },
      { ts_start: T0 + 12, ts_end: T0 + 13, text: 'second', speaker_id: 2, speaker: 'Speaker 2', line: 'mic@second' },
    ]);
    const line = (await screen.findByText('second')).closest('button')?.parentElement as HTMLElement;
    const action = within(line).getByRole('button', { name: /^Not this voice in the line at / });
    expect(action.previousElementSibling?.querySelector('time')).not.toBeNull(); // right after the line and its time
    expect(action.querySelector('.tabler-icon-user-minus')).not.toBeNull(); // the one Not <name> icon
    await userEvent.click(action);
    await justRemove('the unmatched voice');
    await vi.waitFor(() => expect(posts).toEqual(removed('Speaker 2', 'mic@second')));
    const first = screen.getByText('first').closest('article') as HTMLElement;
    expect(within(first).getByRole('button', { name: 'Not this voice' })).toBeInTheDocument();
  });

  it('opens an empty name box from an unmatched name, and tags the line with the name given', async () => {
    renderTranscript();
    await screen.findByText('hello there');
    await vi.waitFor(() => expect(posts).toEqual([]));
    await userEvent.click(within(turnOf('who is this')).getByRole('button', { name: 'Unmatched: say who this is' }));
    const box = await screen.findByRole('combobox', { name: 'Speaker name' });
    expect([box, (box as HTMLInputElement).value]).toEqual([document.activeElement, '']);
    await userEvent.type(box, 'Dana{Enter}');
    await vi.waitFor(() => expect(posts).toEqual([['speakers/tag', { lines: [`mic@${T0 + 20}.000`], name: 'Dana' }]]));
  });

  it('shows each turn at its minute', async () => {
    renderTranscript();
    await screen.findByText('hello there');
    const hm = new Date(T0 * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
    expect(within(turnOf('hello there')).getAllByText(hm)).not.toEqual([]);
  });
});

describe('two microphones in one millisecond', () => {
  const TWINS: TranscriptEntry[] = [
    { ts_start: T0, text: 'from a', speaker_id: 1, speaker: 'Dana', line: `mic-a@${T0}.000` },
    { ts_start: T0, text: 'from b', speaker_id: null, speaker: 'unknown', line: `mic-b@${T0}.000` },
  ];

  it('takes just the one line off its voice, by its key', async () => {
    renderTranscript(TWINS);
    const line = (await screen.findByText('from a')).closest('button')?.parentElement as HTMLElement;
    await userEvent.click(within(line).getByRole('button', { name: /^Not Dana in the line at / }));
    await justRemove('Dana');
    await vi.waitFor(() => expect(posts).toEqual(removed('Dana', `mic-a@${T0}.000`)));
  });

  it('tags the unknown line by its key', async () => {
    renderTranscript(TWINS);
    await screen.findByText('from b');
    await vi.waitFor(() => expect(posts).toEqual([]));
    await userEvent.click(
      within(screen.getByText('from b').closest('article') as HTMLElement).getByRole('button', {
        name: 'Unmatched: say who this is',
      }),
    );
    await userEvent.type(await screen.findByRole('combobox', { name: 'Speaker name' }), 'Dana{Enter}');
    await vi.waitFor(() => expect(posts).toEqual([['speakers/tag', { lines: [`mic-b@${T0}.000`], name: 'Dana' }]]));
  });
});

describe('the feed', () => {
  it('follows the newest lines again after its new-lines pill, as Jump to latest did', async () => {
    const { client } = renderTranscript();
    const feed = await screen.findByRole('region', { name: 'Recent conversation' });
    // jsdom has no layout: a list 1000 px tall in a 300 px box, read at its top, is away from the end
    Object.defineProperty(feed, 'scrollHeight', { configurable: true, value: 1000 });
    Object.defineProperty(feed, 'clientHeight', { configurable: true, value: 300 });
    feed.scrollTop = 0;
    fireEvent.scroll(feed);
    expect(await screen.findByRole('button', { name: 'Follow latest' })).toBeInTheDocument();
    act(() => {
      client.setQueryData(
        liveKey('transcript'),
        reduce('transcript', client.getQueryData(liveKey('transcript')), {
          entries: [
            { ts_start: T0 + 30, ts_end: T0 + 31, text: 'one more', speaker_id: 1, speaker: 'Dana', line: 'mic@more' },
          ],
        }),
      );
    });
    await userEvent.click(await screen.findByRole('button', { name: '1 new line' }));
    expect(await screen.findByRole('button', { name: 'Following latest' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '1 new line' })).not.toBeInTheDocument();
  });

  it('dims the turns without a match, highlights the matches, and says how many match', async () => {
    renderTranscript();
    await userEvent.type(await screen.findByRole('searchbox', { name: 'Search recent conversation' }), 'SUNNY');
    expect(screen.getByRole('status')).toHaveTextContent(/^1 match$/);
    // A highlight splits the words into parts, so a turn is found by its whole text.
    const turn = (text: string) =>
      [...document.querySelectorAll('article')].find(a => a.textContent?.includes(text)) as HTMLElement;
    expect(turn('hello there')).toHaveAttribute('data-missed'); // kept in place, dimmed
    expect(turn('sunny weather')).not.toHaveAttribute('data-missed');
    expect(turn('sunny weather').querySelector('mark')).toHaveTextContent(/^sunny$/);
    expect(screen.getByRole('button', { name: 'Following latest' })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Clear search' }));
    expect(turn('hello there')).not.toHaveAttribute('data-missed');
    expect(document.querySelector('mark')).toBeNull();
  });

  it('matches the name a turn shows, and never a placeholder', async () => {
    renderTranscript();
    const box = await screen.findByRole('searchbox', { name: 'Search recent conversation' });
    await userEvent.type(box, 'unmatched');
    expect(screen.getByRole('status')).toHaveTextContent(/^2 matches$/);
    await userEvent.clear(box);
    await userEvent.type(box, 'speaker 2');
    expect(screen.getByRole('status')).toHaveTextContent(/^No matches$/);
  });

  it('says so when nothing matches', async () => {
    renderTranscript();
    await userEvent.type(await screen.findByRole('searchbox'), 'zebra');
    expect(screen.getByRole('status')).toHaveTextContent(/^No matches$/);
  });

  it('stops and resumes following on request', async () => {
    renderTranscript();
    await userEvent.click(await screen.findByRole('button', { name: 'Following latest' }));
    expect(screen.getByRole('button', { name: 'Follow latest' })).toHaveAttribute('aria-pressed', 'false');
    await userEvent.click(screen.getByRole('button', { name: 'Follow latest' }));
    expect(screen.getByRole('button', { name: 'Following latest' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('plays a line when it is clicked, and stops it on a second click', async () => {
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
    renderTranscript();
    const line = (await screen.findByText('hello there')).closest('button') as HTMLElement;
    await userEvent.click(line);
    expect(usePlayer.getState().ts).toBe(T0);
    expect(line).toHaveAttribute('data-playing');
    await userEvent.click(line);
    expect(usePlayer.getState().ts).toBeNull();
  });

  it('links to Review for everything older', async () => {
    renderTranscript();
    expect(await screen.findByRole('link', { name: 'All recordings →' })).toHaveAttribute('href', '/review');
  });
});

describe('sameTurn (a turn renders again only when its lines change)', () => {
  const a = LINES[0] as TranscriptEntry;
  const b = LINES[1] as TranscriptEntry;
  const turn = (lines: TranscriptEntry[], id: number | null = 1): Turn => ({
    id,
    lines: lines.map((e, i) => ({ e, i })),
    end: 0,
  });
  it.each([
    ['the same line objects in a rebuilt turn', turn([a]), turn([a]), true],
    ['a line added', turn([a]), turn([a, b]), false],
    ['a line replaced by a relabel or a faded highlight (a new object)', turn([a]), turn([{ ...a }]), false],
    ['another voice', turn([a], 1), turn([a], 2), false],
  ])('%s', (_why, x, y, same) => {
    expect(sameTurn(x, y)).toBe(same);
  });
});

it('confirms exactly the stable keys of the visible turn after claiming its voice', async () => {
  const first = { ...LINES[1], line: 'mic-a@1793000010.000' } as TranscriptEntry;
  const second = { ...first, line: 'mic-a@1793000013.000', ts_start: T0 + 13, ts_end: T0 + 15, text: 'A second line.' };
  renderTranscript([first, second]);
  await rename(0, 'Eli');
  await vi.waitFor(() =>
    expect(posts).toEqual([
      ['people/claim', { kind: 'voice', id: 2, name: 'Eli' }],
      ['speakers/tag', { lines: ['mic-a@1793000010.000', 'mic-a@1793000013.000'], name: 'Eli' }],
    ]),
  );
});

it('keeps a successful voice rename when its line-confirmation queue fails and reports the partial result', async () => {
  server.use(
    http.post('/speakers/label', async ({ request }) => {
      posts.push(['speakers/label', await request.json()]);
      return HttpResponse.json({ ok: true, speakers: { speakers: [{ id: 1, label: 'Zed' }], merged: {} } });
    }),
    http.post('/speakers/tag', async ({ request }) => {
      posts.push(['speakers/tag', await request.json()]);
      return HttpResponse.json({ ok: false, error: 'The confirmation queue is unavailable.' }, { status: 503 });
    }),
  );
  renderTranscript([LINES[0] as TranscriptEntry]);
  await rename(0, 'Zed');
  expect(await screen.findByRole('alert')).toHaveTextContent(
    'The voice was named Zed, but its lines could not be queued for confirmation: The confirmation queue is unavailable.',
  );
  expect(screen.getByRole('button', { name: 'Why Zed' })).toBeEnabled();
  expect(posts).toEqual([
    ['speakers/label', { id: 1, name: 'Zed' }],
    ['speakers/tag', { lines: [`mic@${T0}.000`], name: 'Zed' }],
  ]);
});

it('ends a named turn’s queued state when its lines are confirmed even though its cached name already changed', async () => {
  server.use(
    http.post('/speakers/label', () =>
      HttpResponse.json({ ok: true, speakers: { speakers: [{ id: 1, label: 'Zed' }], merged: {} } }),
    ),
  );
  const { client } = renderTranscript([LINES[0] as TranscriptEntry]);
  await rename(0, 'Zed');
  expect(await screen.findByText('Tagging as Zed')).toBeInTheDocument();
  act(() => {
    client.setQueryData(
      liveKey('transcript'),
      reduce('transcript', client.getQueryData(liveKey('transcript')), {
        entries: [{ ...LINES[0], speaker: 'Zed', tagged: true }],
        replace: true,
      }),
    );
  });
  await vi.waitFor(() => expect(screen.queryByText('Tagging as Zed')).not.toBeInTheDocument());
});

describe('incorrect transcript turns', () => {
  const lines = [
    { ts_start: T0, line: `mic-a@${T0}.000`, text: 'first misheard line', speaker_id: 1, speaker: 'Dana' },
    { ts_start: T0 + 2, line: `mic-a@${T0 + 2}.000`, text: 'second misheard line', speaker_id: 1, speaker: 'Dana' },
    { ts_start: T0 + 2, line: `mic-b@${T0 + 2}.000`, text: 'other microphone', speaker_id: 2, speaker: 'Speaker 2' },
  ];
  it('rejects the visible turn by exact keys and offers one Undo after the turn disappears', async () => {
    let rejected = false;
    server.use(
      http.post('/transcript/reject', async ({ request }) => {
        posts.push(['transcript/reject', await request.json()]);
        rejected = true;
        return HttpResponse.json({ ok: true, undo: 'batch-one' });
      }),
      http.post('/transcript/undo', async ({ request }) => {
        posts.push(['transcript/undo', await request.json()]);
        rejected = false;
        return HttpResponse.json({ ok: true });
      }),
      http.get('/transcript', () => HttpResponse.json({ entries: rejected ? [lines[2]] : lines, reset: true })),
    );
    renderTranscript(lines);
    const turn = screen.getByText('first misheard line').closest('article') as HTMLElement;
    await userEvent.click(within(turn).getByRole('button', { name: 'Not transcribed correctly' }));
    await vi.waitFor(() => expect(screen.queryByText('first misheard line')).not.toBeInTheDocument());
    expect(screen.queryByText('second misheard line')).not.toBeInTheDocument();
    expect(screen.getByText('other microphone')).toBeInTheDocument();
    expect(posts).toEqual([['transcript/reject', { lines: [`mic-a@${T0}.000`, `mic-a@${T0 + 2}.000`] }]]);
    expect(screen.getAllByRole('button', { name: 'Undo' })).toHaveLength(1);
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(await screen.findByText('first misheard line')).toBeInTheDocument();
    expect(screen.getByText('second misheard line')).toBeInTheDocument();
    expect(posts.at(-1)).toEqual(['transcript/undo', { undo: 'batch-one' }]);
  });

  it('marks the visible turn as TV, keeps it greyed with a TV tag, and offers an Undo', async () => {
    let marked = false;
    server.use(
      http.post('/transcript/media', async ({ request }) => {
        posts.push(['transcript/media', await request.json()]);
        marked = true;
        return HttpResponse.json({ ok: true, undo: 'tv-one' });
      }),
      http.post('/transcript/undo', async ({ request }) => {
        posts.push(['transcript/undo', await request.json()]);
        marked = false;
        return HttpResponse.json({ ok: true });
      }),
      http.get('/transcript', () =>
        HttpResponse.json({
          entries: marked ? [{ ...lines[0], media: true }, { ...lines[1], media: true }, lines[2]] : lines,
          reset: true,
        }),
      ),
    );
    renderTranscript(lines);
    const turn = screen.getByText('first misheard line').closest('article') as HTMLElement;
    await userEvent.click(within(turn).getByRole('button', { name: 'This is TV' }));
    expect(posts).toEqual([['transcript/media', { lines: [`mic-a@${T0}.000`, `mic-a@${T0 + 2}.000`] }]]);
    const tvTurn = () => screen.getByText('first misheard line').closest('article') as HTMLElement;
    // Two TV tags, and the turn goes by TV: every line a TV's, so there's no voice to name.
    await vi.waitFor(() => expect(within(tvTurn()).getAllByText('TV')).toHaveLength(3));
    expect(within(tvTurn()).queryByRole('button', { name: /^Not Dana/ })).not.toBeInTheDocument();
    expect(within(tvTurn()).getByRole('button', { name: 'Not transcribed correctly' })).toBeInTheDocument();
    expect(screen.getByText('first misheard line')).toBeInTheDocument(); // kept, not hidden
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(posts.at(-1)).toEqual(['transcript/undo', { undo: 'tv-one' }]);
    await vi.waitFor(() => expect(screen.queryByText('TV')).not.toBeInTheDocument());
  });

  it("says who marked a line as TV, and Not TV takes the marks back, a turn's or one line's", async () => {
    server.use(
      http.post('/transcript/not-media', async ({ request }) => {
        posts.push(['transcript/not-media', await request.json()]);
        return HttpResponse.json({ ok: true, lines: 1, undo: `off${posts.length}` });
      }),
      http.post('/transcript/undo', async ({ request }) => {
        posts.push(['transcript/undo', await request.json()]);
        return HttpResponse.json({ ok: true });
      }),
    );
    const by = ['classifier', 'owner'] as const; // the first line marked by ear, the second by the owner
    const marked = lines.map((e, k): TranscriptEntry => (by[k] ? { ...e, media: true, media_by: by[k] } : e));
    // A correction reads the lines again: these, not the default feed's.
    server.use(http.get('/transcript', () => HttpResponse.json({ entries: marked, reset: true })));
    renderTranscript(marked);
    const turn = screen.getByText('first misheard line').closest('article') as HTMLElement;
    expect(within(turn).getByTitle(/^Marked as TV by ear/)).toHaveTextContent(/^TV by ear$/);
    expect(within(turn).getByTitle(/^Marked as TV\. /)).toHaveTextContent(/^TV$/);
    await userEvent.click(within(turn).getByRole('button', { name: 'Not TV' }));
    expect(posts).toEqual([['transcript/not-media', { lines: [`mic-a@${T0}.000`, `mic-a@${T0 + 2}.000`] }]]);
    // Not TV has an Undo.
    await userEvent.click(await screen.findByRole('button', { name: 'Undo' }));
    await vi.waitFor(() => expect(posts.at(-1)).toEqual(['transcript/undo', { undo: 'off1' }]));
    // A line's Not TV is its TV tag, at the start of the line, not where Not <name> sits at its end.
    const second = screen.getByText('second misheard line').closest('button')?.parentElement as HTMLElement;
    const chip = within(second).getByRole('button', { name: /^Not TV: the line at / });
    expect(chip).toHaveTextContent(/^TV$/);
    expect(chip.nextElementSibling).toContainElement(screen.getByText('second misheard line'));
    await userEvent.click(chip);
    expect(posts.at(-1)).toEqual(['transcript/not-media', { lines: [`mic-a@${T0 + 2}.000`] }]);
    // The other microphone's line isn't a TV's: it has no Not TV.
    const other = screen.getByText('other microphone').closest('article') as HTMLElement;
    expect(within(other).queryByRole('button', { name: /^Not TV/ })).not.toBeInTheDocument();
  });

  it('keeps the transcript visible and explains a failed correction', async () => {
    server.use(
      http.post('/transcript/reject', () => HttpResponse.json({ ok: false, error: 'Please refresh' }, { status: 409 })),
    );
    renderTranscript(lines);
    const turn = screen.getByText('first misheard line').closest('article') as HTMLElement;
    await userEvent.click(within(turn).getByRole('button', { name: 'Not transcribed correctly' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Please refresh');
    expect(screen.getByText('first misheard line')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Undo' })).not.toBeInTheDocument();
    expect(within(turn).getByRole('button', { name: 'Not transcribed correctly' })).toBeEnabled();
  });
});

// Naming a line makes the page read the tail again (livetab/api.ts's refreshNames, 4 s after a tag), and that read
// answers with reset: true and up to 300 lines: more history than the tab opened with. jsdom has no layout, so here a
// turn is its header and then a LINE_PX row per line (the list's own estimate), each where the list's translateY puts
// it, in a box VIEW px tall; a scroll the list sets fires its scroll event, as a browser's does.
// (The same fake layout as review/Transcript.test.tsx.)
describe('the list holds its place when the tail is read again', () => {
  const TURN = HEAD_PX + LINE_PX; // a turn per line
  const VIEW = 300;
  const LIST = 'Recent conversation';
  const scrolled = new WeakMap<Element, number>();
  const isList = (el: Element) => el.getAttribute('aria-label') === LIST;
  const listOf = (el: Element) => el.closest(`[aria-label="${LIST}"]`);
  const height = (list: Element) =>
    Number.parseFloat(list.querySelector<HTMLElement>('[data-turns]')?.style.height ?? '0') || 0;
  const slotTop = (slot: Element | null) =>
    Number.parseFloat(
      /translateY\((-?[\d.]+)px\)/.exec((slot as HTMLElement | null)?.style.transform ?? '')?.[1] ?? '0',
    );

  const lineAt = (i: number): TranscriptEntry => ({
    ts_start: T0 + i * 5,
    ts_end: T0 + i * 5 + 2,
    line: `mic@${T0 + i * 5}.000`,
    text: `line ${i}`,
    speaker_id: (i & 1) + 1, // a turn per line: alternating voices
  });
  const from = (first: number, to: number) => Array.from({ length: to - first }, (_, n) => lineAt(first + n));

  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function (this: HTMLElement) {
      return isList(this) ? VIEW : 0;
    });
    vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(function (this: HTMLElement) {
      return isList(this) ? height(this) : 0;
    });
    vi.spyOn(Element.prototype, 'scrollTop', 'get').mockImplementation(function (this: Element) {
      return scrolled.get(this) ?? 0;
    });
    vi.spyOn(Element.prototype, 'scrollTop', 'set').mockImplementation(function (this: Element, v: number) {
      const to = Math.max(0, Math.min(v, height(this) - VIEW));
      if (to === (scrolled.get(this) ?? 0)) return;
      scrolled.set(this, to);
      queueMicrotask(() => this.dispatchEvent(new Event('scroll')));
    });
    // The list measures each turn as it's observed: as tall as its lines.
    vi.stubGlobal(
      'ResizeObserver',
      class {
        readonly cb: ResizeObserverCallback;
        constructor(cb: ResizeObserverCallback) {
          this.cb = cb;
        }
        observe(el: Element) {
          if (!el.hasAttribute('data-index')) return;
          const blockSize = HEAD_PX + el.querySelectorAll('[data-key]').length * LINE_PX;
          const entry = {
            target: el,
            borderBoxSize: [{ blockSize, inlineSize: 300 }],
          } as unknown as ResizeObserverEntry;
          queueMicrotask(() => this.cb([entry], this as unknown as ResizeObserver));
        }
        unobserve() {}
        disconnect() {}
      },
    );
    vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockImplementation(function (this: HTMLElement) {
      return this.hasAttribute('data-index') ? HEAD_PX + this.querySelectorAll('[data-key]').length * LINE_PX : 0;
    });
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
      const list = listOf(this);
      if (!list || isList(this)) return DOMRect.fromRect({ x: 0, y: 0, width: 300, height: VIEW });
      const slot = this.closest('[data-index]');
      const at = slot ? [...slot.querySelectorAll('[data-key]')].indexOf(this) : 0;
      const y = slotTop(slot) + HEAD_PX + at * LINE_PX - (scrolled.get(list) ?? 0);
      return DOMRect.fromRect({ x: 0, y, width: 300, height: LINE_PX });
    });
    return () => {
      vi.restoreAllMocks();
      vi.unstubAllGlobals();
    };
  });

  /** Let queued scroll events, and the renders they cause, run. */
  const settle = () => act(async () => {});

  /** Name the unmatched line at index i, as a reader does: its name badge, then a name. */
  async function nameLine(i: number) {
    const turn = screen.getByText(`line ${i}`).closest('article') as HTMLElement;
    await userEvent.click(within(turn).getByRole('button', { name: 'Unmatched: say who this is' }));
    await userEvent.type(await screen.findByRole('combobox', { name: 'Speaker name' }), 'Dana{Enter}');
    const key = `mic@${T0 + i * 5}.000`;
    await vi.waitFor(() => expect(posts).toEqual([['speakers/tag', { lines: [key], name: 'Dana' }]]));
  }

  it('holds the line a reader named where it was while older lines land above it', async () => {
    const shown = from(100, 120).map((e, n) => (n === 5 ? { ...e, speaker_id: null, speaker: 'unknown' } : e));
    const { client } = renderTranscript(shown);
    const list = await screen.findByRole('region', { name: LIST });
    await settle();
    // The reader scrolled down to a line they are naming: line 105 is the top row, its header LINE_PX below the box top.
    list.scrollTop = 5 * TURN;
    fireEvent.scroll(list);
    await settle();
    expect(screen.getByRole('button', { name: 'Follow latest' })).toBeInTheDocument(); // away from the end
    // The re-read after the naming answers with the whole tail: 80 lines older than the ones the tab opened with.
    server.use(http.get('/transcript', () => HttpResponse.json({ entries: from(20, 120), reset: true })));
    await nameLine(105);
    act(() => {
      void refreshNames(client, 0);
    });
    await vi.waitFor(() => expect(height(list)).toBe(100 * TURN));
    await settle();
    // The line acted on keeps its place on screen: the same distance from the box's top, not moved, and not gone.
    expect(list.scrollTop).toBe(85 * TURN);
    expect(screen.getByText('line 105').closest('[data-key]')?.getBoundingClientRect().top).toBe(HEAD_PX);
    expect(screen.getByRole('button', { name: 'Follow latest' })).toBeInTheDocument(); // still away from the end
    expect(screen.queryByRole('button', { name: /new line/ })).toBeNull(); // nothing arrived below it
  });

  it('holds the line a reader is on when the read brings fewer lines, and no older ones', async () => {
    const shown = from(100, 120);
    const { client } = renderTranscript(shown);
    const list = await screen.findByRole('region', { name: LIST });
    await settle();
    list.scrollTop = 5 * TURN; // line 105, the same place as the naming above
    fireEvent.scroll(list);
    await settle();
    // What the read after a correction answers: a tail with the lines purged since left out.
    act(() => {
      client.setQueryData(
        liveKey('transcript'),
        reduce('transcript', client.getQueryData(liveKey('transcript')), { entries: from(60, 120), reset: true }),
      );
    });
    await vi.waitFor(() => expect(height(list)).toBe(60 * TURN));
    await settle();
    expect(screen.getByText('line 105').closest('[data-key]')?.getBoundingClientRect().top).toBe(HEAD_PX);
    expect(list.scrollTop).toBe(45 * TURN);
  });

  it('stays at the newest line when the tail is read again while following it', async () => {
    const shown = from(100, 120);
    const { client } = renderTranscript(shown);
    const list = await screen.findByRole('region', { name: LIST });
    await settle();
    list.scrollTop = 20 * TURN - VIEW; // the end
    fireEvent.scroll(list);
    await settle();
    expect(screen.getByRole('button', { name: 'Following latest' })).toBeInTheDocument();
    server.use(http.get('/transcript', () => HttpResponse.json({ entries: from(20, 120), reset: true })));
    act(() => {
      void refreshNames(client, 0);
    });
    await vi.waitFor(() => expect(height(list)).toBe(100 * TURN));
    await settle();
    expect(list.scrollTop).toBe(100 * TURN - VIEW);
    expect(screen.getByRole('button', { name: 'Following latest' })).toBeInTheDocument();
  });
});
