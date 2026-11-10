import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import type { MicRecorder } from '../../lib/mic';
import { liveKey } from '../../live/feed';
import { CLONES_KEY } from '../api';
import { Clones } from './Clones';
import { useClones } from './store';

// The microphone is faked: each test sets what the recorder hook reports.
const mic = vi.hoisted(() => ({ current: null as MicRecorder | null }));
vi.mock('../../lib/mic', () => ({ useMicRecorder: () => mic.current }));

const WAV = new Blob([new Uint8Array([82, 73, 70, 70])], { type: 'audio/wav' });

function recorder(over: Partial<MicRecorder> = {}): MicRecorder {
  return {
    phase: 'idle',
    secs: 0,
    start: vi.fn(async () => {}),
    stop: vi.fn(),
    wav: null,
    seconds: 0,
    voiced: 0,
    error: null,
    ...over,
  };
}

const done = (seconds: number, voiced = seconds) => recorder({ phase: 'done', wav: WAV, seconds, voiced });

beforeEach(() => {
  useClones.setState({ sel: null, busy: null });
  mic.current = recorder();
  server.use(
    http.get('/clones', () => HttpResponse.json({ clones: [{ id: 3, label: 'Dana', ready: true, built: true }] })),
  );
});

async function openRecorder() {
  renderPage(<Clones />);
  await userEvent.click(await screen.findByRole('button', { name: 'Record a new voice' }));
  return screen.findByRole('dialog', { name: 'Record a new voice' });
}

describe('the recorder dialog', () => {
  it('takes focus, closes on Escape, and gives focus back', async () => {
    renderPage(<Clones />);
    const open = await screen.findByRole('button', { name: 'Record a new voice' });
    await userEvent.click(open);
    const dialog = await screen.findByRole('dialog', { name: 'Record a new voice' });
    await waitFor(() => expect(dialog).toContainElement(document.activeElement as HTMLElement));
    expect(dialog).toHaveTextContent(
      'Speak for at least 10 seconds in a quiet room (it stops by itself at 30). Liberational learns the voice from this recording and can clone it.',
    );
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    await waitFor(() => expect(open).toHaveFocus());
  });

  it('starts recording', async () => {
    const dialog = await openRecorder();
    await userEvent.click(within(dialog).getByRole('button', { name: 'Start recording' }));
    expect(mic.current?.start).toHaveBeenCalledOnce();
  });

  it('shows the time while recording, and Stop stops it', async () => {
    mic.current = recorder({ phase: 'rec', secs: 4.6 });
    const dialog = await openRecorder();
    const stopButton = within(dialog).getByRole('button', { name: 'Stop' });
    expect(stopButton).toHaveAttribute('aria-pressed', 'true');
    expect(dialog).toHaveTextContent('5 s');
    expect(within(dialog).getByRole('status')).toHaveTextContent('Recording. Press Stop when you are done.');
    await userEvent.click(stopButton);
    expect(mic.current?.stop).toHaveBeenCalledOnce();
  });

  it('refuses a recording that is too short or too quiet', async () => {
    mic.current = done(6.2);
    const dialog = await openRecorder();
    expect(within(dialog).getByRole('status')).toHaveTextContent('That was 6.2 s. Record at least 10 seconds.');
    expect(within(dialog).getByRole('button', { name: 'Record again' })).toBeInTheDocument();
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name for this voice' }), 'Dana');
    expect(within(dialog).getByRole('button', { name: 'Save voice' })).toBeDisabled();
  });

  it('says when the microphone picked up almost nothing', async () => {
    mic.current = done(12, 1.04);
    const dialog = await openRecorder();
    expect(within(dialog).getByRole('status')).toHaveTextContent(
      'The microphone picked up almost nothing (1.0 s of sound). Check the input level, or speak closer, and record again.',
    );
  });

  it("shows the microphone's error", async () => {
    mic.current = recorder({ error: 'The microphone is unavailable (NotAllowedError).' });
    const dialog = await openRecorder();
    expect(within(dialog).getByRole('status')).toHaveTextContent('The microphone is unavailable (NotAllowedError).');
  });

  it('saves the recording as WAV under its name, then reloads the voices', async () => {
    mic.current = done(14.25);
    let sent: { url: string; type: string | null; bytes: number } | null = null;
    server.use(
      http.post('/clones/record', async ({ request }) => {
        sent = {
          url: new URL(request.url).search,
          type: request.headers.get('Content-Type'),
          bytes: (await request.arrayBuffer()).byteLength,
        };
        return HttpResponse.json({ ok: true, name: 'Dana Smith', ref_text: `Hello there. ${'la '.repeat(60)}` });
      }),
    );
    const dialog = await openRecorder();
    expect(within(dialog).getByRole('status')).toHaveTextContent('14.3 s recorded. Name the voice and save it.');
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name for this voice' }), ' Dana Smith ');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Save voice' }));

    const heard = `Hello there. ${'la '.repeat(60)}`.slice(0, 120);
    await waitFor(() =>
      expect(within(dialog).getByRole('status')).toHaveTextContent(`Saved as Dana Smith. Heard: “${heard}”`),
    );
    expect(sent).toEqual({ url: '?name=Dana%20Smith', type: 'audio/wav', bytes: 4 });
    expect(within(dialog).getByRole('button', { name: 'Save voice' })).toBeDisabled();
  });

  it('reloads the voices and the live speakers after a save', async () => {
    mic.current = done(14);
    server.use(http.post('/clones/record', () => HttpResponse.json({ ok: true, name: 'Dana' })));
    const { client } = renderPage(<Clones />);
    const seen = vi.spyOn(client, 'invalidateQueries');
    await userEvent.click(await screen.findByRole('button', { name: 'Record a new voice' }));
    const dialog = await screen.findByRole('dialog', { name: 'Record a new voice' });
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name for this voice' }), 'Dana');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Save voice' }));
    await waitFor(() => expect(within(dialog).getByRole('status')).toHaveTextContent(/^Saved as Dana\.$/));
    expect(seen).toHaveBeenCalledWith({ queryKey: CLONES_KEY });
    expect(seen).toHaveBeenCalledWith({ queryKey: liveKey('speakers') });
  });

  it('says why saving failed, and lets it be tried again', async () => {
    mic.current = done(14);
    server.use(
      http.post('/clones/record', () =>
        HttpResponse.json({ ok: false, error: 'not enough voiced audio' }, { status: 400 }),
      ),
    );
    const dialog = await openRecorder();
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name for this voice' }), 'Dana');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Save voice' }));
    await waitFor(() =>
      expect(within(dialog).getByRole('status')).toHaveTextContent(/^Saving failed: not enough voiced audio$/),
    );
    expect(within(dialog).getByRole('button', { name: 'Save voice' })).toBeEnabled();
  });
});
