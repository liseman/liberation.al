import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { stop, usePlayer } from '../../lib/player';
import { spkColor } from '../../lib/speakers';
import { CLONES_KEY, type Clone } from '../api';
import { Clones } from './Clones';
import { CLONE_TEXT_ID, useClones } from './store';

const CLONES: Clone[] = [
  { id: 3, label: 'Dana', ready: true, built: true, clean_s: 14 },
  { id: 5, label: 'Speaker 5', ready: true, built: false, clean_s: 20.5, usable_s: 14.2, needs_s: 12 },
  { id: 7, label: 'Speaker 7', ready: false, built: false, clean_s: 2, usable_s: 2, needs_s: 12 },
];

function serve(clones: Clone[] = CLONES, error: string | null = null) {
  let gets = 0;
  server.use(
    http.get('/clones', () => {
      gets++;
      return HttpResponse.json({ clones, error });
    }),
  );
  return { gets: () => gets };
}

const voice = (name: RegExp) => screen.getByRole('radio', { name });
const textBox = () => screen.getByRole('textbox', { name: 'Text to say' });
const status = () => screen.getByRole('status');

async function expectTooltip(target: HTMLElement, text: string) {
  await userEvent.hover(target);
  expect(await screen.findByText(text, { selector: '[role="tooltip"]' })).toBeInTheDocument();
  await userEvent.unhover(target);
  await waitFor(() => expect(screen.queryByText(text, { selector: '[role="tooltip"]' })).not.toBeInTheDocument());
}

beforeEach(() => {
  useClones.setState({ sel: null, busy: null });
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
});
afterEach(() => stop());

describe("Settings' voice clones panel", () => {
  it('waits for the first list before showing an empty answer or movable controls', async () => {
    let reply = () => {};
    const held = new Promise<void>(resolve => {
      reply = resolve;
    });
    let requested = false;
    server.use(
      http.get('/clones', async () => {
        requested = true;
        await held;
        return HttpResponse.json({ clones: CLONES });
      }),
    );
    renderPage(<Clones />);
    try {
      await waitFor(() => expect(requested).toBe(true));
      expect(screen.queryByText(/No voices heard yet/)).not.toBeInTheDocument();
      expect(screen.queryByRole('textbox', { name: 'Text to say' })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Record a new voice' })).not.toBeInTheDocument();
      expect(screen.getByLabelText('Loading voice clones')).toBeInTheDocument();
    } finally {
      reply();
    }
    expect(await screen.findByRole('radio', { name: /^Dana/ })).toBeInTheDocument();
    expect(textBox()).toBeInTheDocument();
    expect(screen.queryByLabelText('Loading voice clones')).not.toBeInTheDocument();
  });

  it('keeps cached choices and the text draft mounted while refreshing', async () => {
    serve();
    const { client } = renderPage(<Clones />);
    await userEvent.click(await screen.findByRole('radio', { name: /^Dana/ }));
    const field = textBox();
    await userEvent.type(field, 'Draft to keep');
    let reply = () => {};
    const held = new Promise<void>(resolve => {
      reply = resolve;
    });
    let requested = false;
    server.use(
      http.get('/clones', async () => {
        requested = true;
        await held;
        return HttpResponse.json({ clones: CLONES });
      }),
    );
    act(() => {
      void client.invalidateQueries({ queryKey: CLONES_KEY });
    });
    try {
      await waitFor(() => expect(requested).toBe(true));
      expect(screen.queryByLabelText('Loading voice clones')).not.toBeInTheDocument();
      expect(textBox()).toBe(field);
      expect(field).toHaveValue('Draft to keep');
      expect(voice(/^Dana/)).toHaveAttribute('aria-checked', 'true');
    } finally {
      reply();
    }
    await waitFor(() => expect(client.isFetching({ queryKey: CLONES_KEY })).toBe(0));
    expect(textBox()).toBe(field);
  });

  it('offers Retry after the initial list fails without claiming there are no voices', async () => {
    server.use(http.get('/clones', () => HttpResponse.json({ error: 'temporarily unavailable' }, { status: 503 })));
    renderPage(<Clones />);
    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't load voice clones");
    expect(screen.queryByText(/No voices heard yet/)).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Text to say' })).not.toBeInTheDocument();
    serve();
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByRole('radio', { name: /^Dana/ })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('lists the voices, with only the ready ones to pick', async () => {
    serve();
    renderPage(<Clones />);
    const group = await screen.findByRole('radiogroup', { name: 'Voice to clone' });
    const radios = within(group).getAllByRole('radio');
    expect(radios.map(r => r.textContent)).toEqual([
      'Danacloned',
      'Speaker 514 of 21 s usable',
      'Speaker 7not enough usable speech yet',
    ]);
    expect(voice(/^Dana/)).toBeEnabled();
    await expectTooltip(voice(/^Dana/), 'Reference clip ready');
    await expectTooltip(voice(/^Speaker 5/), '21 s heard, 14 s usable for cloning, needs 12 s');
    expect(voice(/^Speaker 7/)).toBeDisabled();
    // The wrapper receives hover events even while this voice is disabled.
    await expectTooltip(
      voice(/^Speaker 7/).parentElement as HTMLElement,
      '2 s heard, 2 s usable for cloning, needs 12 s',
    );
    expect(voice(/^Speaker 5/).style.getPropertyValue('--who')).toBe(spkColor(5));
    expect(textBox()).toHaveAttribute('id', CLONE_TEXT_ID);
    expect(textBox()).toHaveAttribute('placeholder', 'Pick a voice first');
    expect(screen.getByRole('button', { name: 'Say it' })).toBeDisabled();
  });

  it('says when no voice can be cloned yet', async () => {
    serve([]);
    renderPage(<Clones />);
    expect(
      await screen.findByText('No voices heard yet. Talk near the microphone for a while, or record a voice.'),
    ).toBeInTheDocument();
    expect(textBox()).toHaveAttribute('placeholder', 'Text to say');
  });

  it("shows the list's error in the status line", async () => {
    serve([], 'the voice store is locked');
    renderPage(<Clones />);
    await waitFor(() => expect(status()).toHaveTextContent('the voice store is locked'));
  });

  it('picks a voice and puts the cursor in its text box', async () => {
    serve();
    renderPage(<Clones />);
    await userEvent.click(await screen.findByRole('radio', { name: /^Speaker 5/ }));
    expect(voice(/^Speaker 5/)).toHaveAttribute('aria-checked', 'true');
    expect(voice(/^Dana/)).toHaveAttribute('aria-checked', 'false');
    expect(textBox()).toHaveAttribute('placeholder', 'What should Speaker 5 say?');
    await waitFor(() => expect(textBox()).toHaveFocus());
    expect(screen.getByRole('button', { name: 'Build the reference' })).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Remove clone' })).not.toBeInTheDocument();
  });

  it('says the text in the picked voice and plays it', async () => {
    const list = serve();
    let sent: unknown;
    server.use(
      http.post('/clones/say', async ({ request }) => {
        sent = await request.json();
        return HttpResponse.json({ ok: true, url: '/clones/render/abc.wav' });
      }),
    );
    renderPage(<Clones />);
    await userEvent.click(await screen.findByRole('radio', { name: /^Dana/ }));
    await userEvent.type(textBox(), ' Dinner is ready ');
    const before = list.gets();
    await userEvent.click(screen.getByRole('button', { name: 'Say it' }));
    await waitFor(() => expect(status()).toHaveTextContent(/^Spoken in \d+\.\d s$/));
    expect(sent).toEqual({ id: 3, text: 'Dinner is ready' });
    expect(usePlayer.getState().key).toMatch(/^say:\d+$/);
    expect(document.querySelector('audio')?.getAttribute('src')).toBe('/clones/render/abc.wav');
    await waitFor(() => expect(list.gets()).toBe(before + 1));
  });

  it('says the first time is slow, and shows why speaking failed', async () => {
    serve();
    let reply = () => {};
    const held = new Promise<void>(r => {
      reply = r;
    });
    server.use(
      http.post('/clones/say', async () => {
        await held;
        return HttpResponse.json({ ok: false, error: 'the voice model is not loaded' }, { status: 503 });
      }),
    );
    renderPage(<Clones />);
    await userEvent.click(await screen.findByRole('radio', { name: /^Speaker 5/ }));
    await userEvent.type(textBox(), 'Hello');
    await userEvent.click(screen.getByRole('button', { name: 'Say it' }));
    expect(status()).toHaveTextContent('Speaking (the first time takes 5 to 20 s)');
    expect(screen.getByRole('button', { name: 'Say it' })).toBeDisabled();
    reply();
    await waitFor(() => expect(status()).toHaveTextContent(/^the voice model is not loaded$/));
    expect(screen.getByRole('button', { name: 'Say it' })).toBeEnabled();
  });

  it('plays the reference of a built voice', async () => {
    serve();
    renderPage(<Clones />);
    await userEvent.click(await screen.findByRole('radio', { name: /^Dana/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Hear the reference' }));
    expect(usePlayer.getState().key).toBe('ref:3');
    expect(document.querySelector('audio')?.getAttribute('src')).toMatch(/^\/clones\/audio\?id=3&t=\d+$/);
  });

  it('builds the reference of a voice that has none', async () => {
    serve();
    let sent: unknown;
    server.use(
      http.post('/clones/build', async ({ request }) => {
        sent = await request.json();
        return HttpResponse.json({ ok: true, utterances: 1, seconds: 13.2 });
      }),
    );
    renderPage(<Clones />);
    await userEvent.click(await screen.findByRole('radio', { name: /^Speaker 5/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Build the reference' }));
    await waitFor(() => expect(status()).toHaveTextContent('Reference built from 1 line, 13.2 s'));
    expect(sent).toEqual({ id: 5 });
  });

  it('asks before removing a clone', async () => {
    serve();
    const sent: unknown[] = [];
    server.use(
      http.post('/clones/remove', async ({ request }) => {
        sent.push(await request.json());
        return HttpResponse.json({ ok: true, removed: ['ref.wav', 'ref.txt'] });
      }),
    );
    renderPage(<Clones />);
    await userEvent.click(await screen.findByRole('radio', { name: /^Dana/ }));
    const button = screen.getByRole('button', { name: 'Remove clone' });
    await expectTooltip(button, "Delete this voice's clone reference and renders");

    await userEvent.click(button);
    let ask = await screen.findByRole('dialog', { name: 'Remove clone' });
    expect(ask).toHaveTextContent(
      'Remove the cloned voice for Dana? The speaker and their lines stay; the clone can be rebuilt later.',
    );
    await userEvent.click(within(ask).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(sent).toEqual([]);

    await userEvent.click(button);
    ask = await screen.findByRole('dialog', { name: 'Remove clone' });
    await userEvent.click(within(ask).getByRole('button', { name: 'Remove' }));
    await waitFor(() => expect(status()).toHaveTextContent('Clone removed (2 files)'));
    expect(sent).toEqual([{ id: 3 }]);
  });
});
