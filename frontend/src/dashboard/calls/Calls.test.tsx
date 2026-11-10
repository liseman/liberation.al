import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { stop, usePlayer } from '../lib/player';
import type { Call, Person } from '../people/api';
import { Calls } from './Calls';

const CALLS: Call[] = [
  {
    day: '2026-10-27',
    dir: 'call-1',
    from: '+14155550123',
    durationSeconds: 61.6,
    turns: 3,
    speakers: { Dana: {}, Eli: {} },
  },
  { day: '2026-10-26', dir: 'call-2', durationSeconds: 4, turns: 1 },
];

const person = (name: string, number: string | null = null): Person => ({ name, voices: [], number });

function serve(calls: Call[] = CALLS) {
  server.use(http.get('/calls', () => HttpResponse.json(calls)));
}

/** A call's row by the start of its name ("+14155550123", "Called Dana"): its name goes on with when and how long. */
const named = (start: string) => new RegExp(`^${start.replace(/[+()]/g, '\\$&')}`);
const opener = (start: string) => screen.getByRole('button', { name: named(start) });
const row = (start: string) => {
  const item = opener(start).closest('.mantine-Accordion-item');
  if (!item) throw new Error(`no row for ${start}`);
  return item as HTMLElement;
};

beforeEach(() => {
  (globalThis as { NAMES?: object }).NAMES = { assistant: 'Ada' };
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
});
afterEach(() => {
  stop();
  delete (globalThis as { NAMES?: object }).NAMES;
  vi.useRealTimers();
});

describe('the calls list', () => {
  it('puts the call log directly after the call action form', async () => {
    serve();
    renderPage(<Calls people={[person('Dana', '+14155550123')]} />);
    await screen.findByRole('button', { name: named('Unknown caller') });
    const formHeading = screen.getByRole('heading', { name: 'Have Ada call someone', hidden: true }); // folded
    const log = screen.getByRole('heading', { name: 'Call log' });
    expect(formHeading.closest('form')?.nextElementSibling).toBe(log);
    expect(log.compareDocumentPosition(opener('Unknown caller')) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
  it('lists each call with who, how long, and who spoke, in minutes and not turns', async () => {
    serve();
    renderPage(<Calls people={[]} />);
    expect(screen.getByRole('heading', { name: 'Calls' })).toBeInTheDocument();
    expect(screen.getByLabelText('Loading the call log')).toBeInTheDocument();
    expect(await screen.findByRole('button', { name: named('+14155550123') })).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    expect(row('+14155550123')).toHaveTextContent('1 min · with Dana, Eli');
    expect(row('Unknown caller')).toHaveTextContent('4 s');
    expect(row('Unknown caller')).not.toHaveTextContent(/with|turn/);
  });

  it('lists calls under their day, with the time in 24 hours', async () => {
    const started = '2026-10-27T15:04:00';
    serve([{ ...CALLS[1], started } as Call]);
    renderPage(<Calls people={[]} />);
    expect(await screen.findByText('15:04 · 4 s')).toBeInTheDocument();
    // under its day's heading: the day in the locale's words (with 27), and no year, since it's this year's
    const day = opener('Unknown caller').closest('section');
    expect(day).toHaveAccessibleName(/27/);
    expect(day).not.toHaveAccessibleName(/2026/);
  });

  it('says which way a call went and with whom, what the assistant was asked, and what came of it (#2684)', async () => {
    serve([
      {
        day: '2026-10-27',
        dir: 'out-1',
        direction: 'out',
        to: { name: null, number: '+1 415 555 0123' }, // a typed number that's someone's in People
        from: '+1 415 555 0123',
        instructions: 'Ask whether she is free for dinner on Friday.',
        outcome: 'She is free on Friday after 7.',
        durationSeconds: 182,
      },
      { day: '2026-10-27', dir: 'in-1', direction: 'in', from: '+14155550199', durationSeconds: 40 },
      { day: '2026-10-27', dir: 'old-1', direction: 'unknown', from: '+14155550188', durationSeconds: 9 },
    ]);
    renderPage(<Calls people={[person('Dana', '+14155550123'), person('Fay', '+14155550199')]} />);
    expect(await screen.findByRole('button', { name: named('Called Dana') })).toBeInTheDocument();
    expect(row('Called Dana')).toHaveTextContent('3 min');
    expect(row('Called Dana')).toHaveTextContent('Asked: Ask whether she is free for dinner on Friday.');
    expect(row('Called Dana')).toHaveTextContent('She is free on Friday after 7.');
    expect(row('Fay called')).toHaveTextContent('40 s');
    expect(row('Fay called')).not.toHaveTextContent('Asked');
    expect(row('+14155550188')).toHaveTextContent('9 s'); // an older call: who it was with, no way
  });

  it('shows at most 20 calls', async () => {
    serve(Array.from({ length: 25 }, (_, i) => ({ day: '2026-10-27', dir: `c${i}`, from: `Caller ${i}` })));
    renderPage(<Calls people={[]} />);
    await screen.findByRole('button', { name: named('Caller 0') });
    expect(screen.getAllByRole('button', { name: /^Caller \d+/ })).toHaveLength(20);
    expect(screen.queryByRole('button', { name: named('Caller 20') })).not.toBeInTheDocument();
  });

  it('opens a transcript once, and closes it again', async () => {
    serve();
    let asked = 0;
    let sent = '';
    server.use(
      http.get('/calls/detail', ({ request }) => {
        asked++;
        sent = new URL(request.url).search;
        return HttpResponse.json({
          transcript: [
            { who: 'Ada', t: 0.4, text: 'Hello, this is Ada.' },
            { t: 3.6, text: 'Who is this?' },
          ],
        });
      }),
    );
    renderPage(<Calls people={[]} />);
    const who = await screen.findByRole('button', { name: named('+14155550123') });
    await userEvent.click(who);
    expect(who).toHaveAttribute('aria-expanded', 'true');
    expect(await within(row('+14155550123')).findByText('Hello, this is Ada.')).toBeInTheDocument();
    expect(sent).toBe('?day=2026-10-27&dir=call-1');
    expect(row('+14155550123')).toHaveTextContent('Ada · 0:00');
    expect(row('+14155550123')).toHaveTextContent('Unknown · 0:04Who is this?');

    await userEvent.click(who);
    expect(who).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByText('Hello, this is Ada.')).not.toBeInTheDocument();
    await userEvent.click(who);
    expect(screen.getByText('Hello, this is Ada.')).toBeInTheDocument();
    expect(asked).toBe(1);
  });

  it('keeps independently opened transcripts expanded while other calls open or play', async () => {
    serve();
    server.use(http.get('/calls/detail', () => HttpResponse.json({ transcript: [] })));
    renderPage(<Calls people={[]} />);
    const first = await screen.findByRole('button', { name: named('+14155550123') });
    const second = opener('Unknown caller');
    await userEvent.click(first);
    await userEvent.click(second);
    expect(first).toHaveAttribute('aria-expanded', 'true');
    expect(second).toHaveAttribute('aria-expanded', 'true');
    await userEvent.click(within(row('+14155550123')).getByRole('button', { name: 'Play the call' }));
    expect(first).toHaveAttribute('aria-expanded', 'true');
    expect(second).toHaveAttribute('aria-expanded', 'true');
    await userEvent.click(first);
    expect(second).toHaveAttribute('aria-expanded', 'true');
  });

  it("says when a call's transcript is missing", async () => {
    serve();
    let answer = () => {};
    const held = new Promise<void>(r => {
      answer = r;
    });
    server.use(
      http.get('/calls/detail', async () => {
        await held;
        return HttpResponse.json({ error: 'no such call' }, { status: 404 });
      }),
    );
    renderPage(<Calls people={[]} />);
    await userEvent.click(await screen.findByRole('button', { name: named('+14155550123') }));
    expect(await within(row('+14155550123')).findByLabelText('Loading the transcript')).toBeInTheDocument();
    answer();
    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't load this call's transcript");
  });

  it('plays a call from its opened row through the shared player, and pauses it on a second press', async () => {
    serve();
    server.use(http.get('/calls/detail', () => HttpResponse.json({ transcript: [] })));
    renderPage(<Calls people={[]} />);
    await userEvent.click(await screen.findByRole('button', { name: named('+14155550123') }));
    const where = within(row('+14155550123')).getByRole('slider', { name: 'Where in the call' });
    expect(where).toHaveAttribute('data-disabled'); // nothing to move through until it plays
    await userEvent.click(within(row('+14155550123')).getByRole('button', { name: 'Play the call' }));
    expect(usePlayer.getState().key).toBe('call:call-1');
    expect(document.querySelector('audio')?.getAttribute('src')).toBe(
      '/calls/audio?day=2026-10-27&dir=call-1&which=mixed',
    );
    expect(row('+14155550123')).toHaveTextContent('0:00 / 1:02');
    const audio = document.querySelector('audio') as HTMLAudioElement;
    Object.defineProperty(audio, 'paused', { value: false, configurable: true }); // jsdom's audio never starts
    await userEvent.click(within(row('+14155550123')).getByRole('button', { name: 'Pause the call' }));
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled();
    expect(usePlayer.getState().key).toBe('call:call-1'); // paused where it was, not stopped
    delete (audio as { paused?: boolean }).paused;
  });
});

describe('empty states', () => {
  it('keeps manual calling available without contacts or call history', async () => {
    serve([]);
    renderPage(<Calls people={[]} />);
    await userEvent.click(screen.getByRole('radio', { name: 'Number' }));
    expect(screen.getByLabelText('Phone number with country code')).toBeInTheDocument();
  });

  it('keeps the form when someone has a number', async () => {
    serve([]);
    renderPage(<Calls people={[person('Dana', '+14155550123')]} />);
    expect(await screen.findByText('No calls yet')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Have Ada call someone' })).toBeInTheDocument();
  });

  it('says the call log failed instead of saying there are no calls, and retries', async () => {
    let fail = true;
    server.use(
      http.get('/calls', () =>
        fail ? HttpResponse.json({ error: 'disk' }, { status: 500 }) : HttpResponse.json(CALLS),
      ),
    );
    renderPage(<Calls people={[]} />);
    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't load the call log");
    expect(screen.queryByText('No calls yet')).not.toBeInTheDocument();
    // Nobody has a number, so the form offers a number to type, and Call waits for one
    expect(screen.getByLabelText('Phone number with country code')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Call' })).toBeDisabled();
    fail = false;
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByRole('button', { name: named('+14155550123') })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

describe('placing a call', () => {
  const people = [person('Dana', '+14155550123'), person('Eli'), person('Fay', '+14155550199')];

  it('folds the form to "Call someone" once there are calls, and opens it in place', async () => {
    serve();
    renderPage(<Calls people={people} />);
    const open = await screen.findByRole('button', { name: 'Call someone' });
    expect(open).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('heading', { name: 'Have Ada call someone' })).not.toBeInTheDocument();
    await userEvent.click(open);
    expect(screen.getByRole('heading', { name: 'Have Ada call someone' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Close' })).toHaveAttribute('aria-expanded', 'true');
  });

  it('offers only people with a number', async () => {
    serve([]);
    renderPage(<Calls people={people} />);
    const pick = await screen.findByRole('combobox', { name: 'Who to call' });
    expect(
      within(pick)
        .getAllByRole('option')
        .map(o => o.textContent),
    ).toEqual(['Pick someone', 'Dana, +14155550123', 'Fay, +14155550199']);
    expect(
      screen.getByText(
        'Ada calls on your behalf, follows your instructions as the call goes, and records and transcribes it.',
      ),
    ).toBeInTheDocument();
  });

  it('waits for someone to call before Call can be pressed (#2668)', async () => {
    serve([]);
    renderPage(<Calls people={people} />);
    expect(await screen.findByRole('button', { name: 'Call' })).toBeDisabled();
    await userEvent.selectOptions(screen.getByLabelText('Who to call'), 'Dana');
    expect(screen.getByRole('button', { name: 'Call Dana' })).toBeEnabled();
  });

  it('opens on Number while nobody has a number, and says where numbers come from (#2668)', async () => {
    serve([]);
    renderPage(<Calls people={[person('Eli')]} />);
    expect(screen.getByRole('radio', { name: 'Number' })).toBeChecked();
    expect(screen.getByRole('button', { name: 'Call' })).toBeDisabled();
    await userEvent.type(screen.getByLabelText('Phone number with country code'), '+14155550123');
    expect(screen.getByRole('button', { name: 'Call +14155550123' })).toBeEnabled();
    await userEvent.click(screen.getByRole('radio', { name: 'Person' }));
    expect(screen.getByLabelText('Who to call')).toBeDisabled();
    expect(screen.getByRole('link', { name: 'People' })).toHaveAttribute('href', '/people');
    expect(screen.getByRole('button', { name: 'Call' })).toBeDisabled();
  });

  it('stays on Person while the contacts load', () => {
    serve([]);
    renderPage(<Calls people={undefined} />);
    expect(screen.getByRole('radio', { name: 'Person' })).toBeChecked();
    expect(screen.queryByRole('link', { name: 'People' })).not.toBeInTheDocument();
  });

  it('asks first, posts nothing on Cancel, then places the call and looks again later', async () => {
    let gets = 0;
    server.use(
      http.get('/calls', () => {
        gets++;
        return HttpResponse.json([]);
      }),
    );
    const sent: unknown[] = [];
    server.use(
      http.post('/calls/place', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ok: true });
      }),
    );
    renderPage(<Calls people={people} />);
    await userEvent.selectOptions(await screen.findByRole('combobox', { name: 'Who to call' }), 'Dana');
    await userEvent.type(screen.getByRole('textbox', { name: 'What should Ada do?' }), '  Dinner is at 7  ');

    await userEvent.click(screen.getByRole('button', { name: 'Call Dana' }));
    let ask = await screen.findByRole('dialog', { name: 'Place a call' });
    expect(ask).toHaveTextContent('Have Ada call Dana (+14155550123) now with instructions: “Dinner is at 7”?');
    await userEvent.click(within(ask).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(sent).toEqual([]);

    await userEvent.click(screen.getByRole('button', { name: 'Call Dana' }));
    ask = await screen.findByRole('dialog', { name: 'Place a call' });
    // The clock moves on its own too, so the network and waitFor still run.
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
    await userEvent
      .setup({ advanceTimers: vi.advanceTimersByTime })
      .click(within(ask).getByRole('button', { name: 'Call' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Calling Dana.'));
    expect(sent).toEqual([{ name: 'Dana', message: 'Dinner is at 7', number: '+14155550123', confirmed: true }]);

    const before = gets;
    await vi.advanceTimersByTimeAsync(19000);
    expect(gets).toBe(before);
    await vi.advanceTimersByTimeAsync(1000);
    await waitFor(() => expect(gets).toBe(before + 1));
  });

  it("shows the server's word on a placed call, and asks plainly without a message", async () => {
    serve([]);
    server.use(http.post('/calls/place', () => HttpResponse.json({ ok: true, detail: 'Dialing Fay now.' })));
    renderPage(<Calls people={people} />);
    await userEvent.selectOptions(await screen.findByRole('combobox', { name: 'Who to call' }), 'Fay');
    await userEvent.click(screen.getByRole('button', { name: 'Call Fay' }));
    const ask = await screen.findByRole('dialog', { name: 'Place a call' });
    expect(ask).toHaveTextContent(/^.*Have Ada call Fay \(\+14155550199\) now\?/);
    await userEvent.click(within(ask).getByRole('button', { name: 'Call' }));
    expect(await screen.findByText('Dialing Fay now.')).toBeInTheDocument();
  });

  it("shows the server's reason when the call is refused", async () => {
    serve([]);
    server.use(
      http.post('/calls/place', () =>
        HttpResponse.json({ ok: false, detail: 'Twilio is not configured' }, { status: 409 }),
      ),
    );
    renderPage(<Calls people={people} />);
    await userEvent.selectOptions(await screen.findByRole('combobox', { name: 'Who to call' }), 'Dana');
    await userEvent.click(screen.getByRole('button', { name: 'Call Dana' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Place a call' })).getByRole('button', { name: 'Call' }),
    );
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/^Twilio is not configured$/));
  });

  it('says the call failed when the server cannot be reached', async () => {
    serve([]);
    server.use(http.post('/calls/place', () => HttpResponse.error()));
    renderPage(<Calls people={people} />);
    await userEvent.selectOptions(await screen.findByRole('combobox', { name: 'Who to call' }), 'Dana');
    await userEvent.click(screen.getByRole('button', { name: 'Call Dana' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Place a call' })).getByRole('button', { name: 'Call' }),
    );
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/^The call could not be placed: /));
  });
});

describe('manual call targets', () => {
  it('normalizes and confirms the exact target and preserves instructions when switching', async () => {
    serve([]);
    const sent: unknown[] = [];
    server.use(
      http.post('/calls/place', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ok: true });
      }),
    );
    renderPage(<Calls people={[person('Dana', '+14155550199')]} />);
    await userEvent.type(screen.getByLabelText('What should Ada do?'), 'Arrange dinner');
    await userEvent.selectOptions(screen.getByLabelText('Who to call'), 'Dana');
    await userEvent.click(screen.getByRole('radio', { name: 'Number' }));
    await userEvent.type(screen.getByLabelText('Phone number with country code'), '+44 20 7946 0123');
    await userEvent.click(screen.getByRole('button', { name: 'Call +442079460123' }));
    const dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveTextContent('+442079460123');
    expect(dialog).toHaveTextContent('Arrange dinner');
    expect(sent).toEqual([]);
    await userEvent.click(within(dialog).getByRole('button', { name: 'Call' }));
    await waitFor(() =>
      expect(sent).toEqual([{ name: '', number: '+442079460123', message: 'Arrange dinner', confirmed: true }]),
    );
    await userEvent.click(screen.getByRole('radio', { name: 'Person' }));
    expect(screen.getByLabelText('Who to call')).toHaveValue('');
    expect(screen.getByLabelText('What should Ada do?')).toHaveValue('Arrange dinner');
  });
  it.each(['', '4155550123', '+44 (0)20 7946 0123', '+123', '+1abc5550123'])(
    'refuses ambiguous or invalid target %s before confirmation',
    async number => {
      serve([]);
      renderPage(<Calls people={[]} />);
      const field = screen.getByLabelText('Phone number with country code');
      if (number) await userEvent.type(field, number);
      await userEvent.tab(); // leave the field: an incomplete number now says why Call waits
      expect(screen.getByRole('button', { name: 'Call' })).toBeDisabled();
      // Mantine ties the error to the field by aria-describedby and aria-invalid
      if (number) {
        expect(field).toHaveAttribute('aria-invalid', 'true');
        expect(field).toHaveAccessibleDescription(/international number/);
      } else expect(field).not.toHaveAttribute('aria-invalid', 'true');
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    },
  );
});

describe('a call in progress (#2699)', () => {
  const live = (more: object = {}) => ({
    direction: 'out',
    number: '+14155550123',
    name: null,
    since: Date.now() / 1000 - 134,
    day: '2026-10-28',
    dir: 'call-live',
    ...more,
  });

  it('pins "On a call with" who it is, matched from the number, with how long so far', async () => {
    serve([]);
    server.use(http.get('/calls/live', () => HttpResponse.json({ live: live(), count: 1 })));
    renderPage(<Calls people={[person('Dana', '+1 415 555 0123')]} />);
    expect(await screen.findByText('On a call with Dana')).toHaveAttribute('role', 'status');
    expect(screen.getByText('2:14')).toBeInTheDocument();
  });

  it('shows nothing when no call is on', async () => {
    serve();
    renderPage(<Calls people={[]} />);
    await screen.findByRole('button', { name: named('+14155550123') });
    expect(screen.queryByText(/On a call/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Hang up' })).not.toBeInTheDocument();
  });

  it('ends the clicked record once, keeps it while pending, and refreshes live and history after success', async () => {
    let on = true;
    let gets = 0;
    let liveGets = 0;
    const sent: unknown[] = [];
    let answer = () => {};
    const held = new Promise<void>(resolve => {
      answer = resolve;
    });
    server.use(
      http.get('/calls', () => {
        gets++;
        return HttpResponse.json(CALLS);
      }),
      http.get('/calls/live', () => {
        liveGets++;
        return HttpResponse.json({ live: on ? live({ name: 'Dana' }) : null, count: on ? 1 : 0 });
      }),
      http.post('/calls/hangup', async ({ request }) => {
        sent.push(await request.json());
        await held;
        on = false;
        return HttpResponse.json({ ok: true, detail: 'Call ended.' });
      }),
    );
    renderPage(<Calls people={[]} />);
    await userEvent.click(await screen.findByRole('button', { name: 'Hang up' }));
    const busy = screen.getByRole('button', { name: 'Hanging up…' });
    expect(busy).toBeDisabled();
    expect(screen.getByText('On a call with Dana')).toBeInTheDocument();
    await userEvent.click(busy);
    expect(sent).toEqual([{ day: '2026-10-28', dir: 'call-live', confirmed: true }]);
    const before = gets;
    const liveBefore = liveGets;
    answer();
    expect(await screen.findByText('Call ended.')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('On a call with Dana')).not.toBeInTheDocument());
    expect(gets).toBeGreaterThan(before);
    expect(liveGets).toBeGreaterThan(liveBefore);
  });

  it.each([
    [400, 'Select a call and confirm.'],
    [409, 'The selected call is no longer active.'],
    [502, 'Could not connect to the call provider.'],
  ])('keeps the row and allows an explicit safe retry after HTTP %s', async (status, detail) => {
    serve();
    const sent: unknown[] = [];
    server.use(
      http.get('/calls/live', () => HttpResponse.json({ live: live({ name: 'Dana' }), count: 1 })),
      http.post('/calls/hangup', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ok: false, detail }, { status });
      }),
    );
    renderPage(<Calls people={[]} />);
    await userEvent.click(await screen.findByRole('button', { name: 'Hang up' }));
    expect(await screen.findByText(`Could not hang up: ${detail}`)).toBeInTheDocument();
    expect(screen.getByText('On a call with Dana')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Hang up' })).toBeEnabled();
    await userEvent.click(screen.getByRole('button', { name: 'Hang up' }));
    await waitFor(() =>
      expect(sent).toEqual([
        { day: '2026-10-28', dir: 'call-live', confirmed: true },
        { day: '2026-10-28', dir: 'call-live', confirmed: true },
      ]),
    );
  });

  it('when the call ends, it goes, and the log is read again with it as the newest row', async () => {
    let on = true;
    let saved = false;
    server.use(
      http.get('/calls/live', () =>
        HttpResponse.json(on ? { live: live({ name: 'Dana' }), count: 1 } : { live: null, count: 0 }),
      ),
      http.get('/calls', () =>
        HttpResponse.json(
          saved
            ? [
                { day: '2026-10-28', dir: 'call-live', direction: 'out', to: { name: 'Dana', number: '+14155550123' } },
                ...CALLS,
              ]
            : CALLS,
        ),
      ),
    );
    vi.useFakeTimers({
      toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'],
      shouldAdvanceTime: true,
    });
    renderPage(<Calls people={[]} />);
    expect(await screen.findByText('On a call with Dana')).toBeInTheDocument();
    on = false;
    saved = true;
    await vi.advanceTimersByTimeAsync(3000); // the next look for a call on
    await waitFor(() => expect(screen.queryByText(/On a call/)).not.toBeInTheDocument());
    expect(await screen.findByRole('button', { name: named('Called Dana') })).toBeInTheDocument();
  });
});
