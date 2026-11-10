import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { MicRecorder } from '../lib/mic';
import { play, stop, usePlayer } from '../lib/player';
import { VoiceRef } from './VoiceRef';

const recorder = vi.hoisted(() => ({
  phase: 'idle' as 'idle' | 'rec' | 'done',
  secs: 0,
  seconds: 0,
  voiced: 0,
  wav: null as Blob | null,
  error: null as string | null,
  start: vi.fn(async () => {}),
  stop: vi.fn(),
}));
vi.mock('../lib/mic', () => ({ useMicRecorder: () => recorder satisfies MicRecorder }));
vi.mock('../lib/player', async importOriginal => ({
  ...(await importOriginal<typeof import('../lib/player')>()),
  play: vi.fn(),
  stop: vi.fn(),
}));
const reference = { path: 'voice.wav', exists: false, seconds: 0, prompt: 'Read this passage aloud.' };
const create = vi.fn(() => 'blob:voice-draft');
const revoke = vi.fn();

beforeEach(() => {
  Object.assign(recorder, { phase: 'idle', secs: 0, seconds: 0, voiced: 0, wav: null, error: null });
  vi.clearAllMocks();
  vi.stubGlobal(
    'URL',
    class extends URL {
      static createObjectURL = create;
      static revokeObjectURL = revoke;
    },
  );
  usePlayer.setState({ key: null, ts: null, paused: false, status: '' });
  server.use(http.get('/config/voice_ref', () => HttpResponse.json(reference)));
});
afterEach(() => vi.unstubAllGlobals());

it('shows the passage and starts raw recording, with no save before a take exists', async () => {
  renderPage(<VoiceRef readOnly={false} />);
  expect(await screen.findByText(reference.prompt)).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Save as my voice' })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Record' }));
  expect(recorder.start).toHaveBeenCalledOnce();
});

it('stops a recording and reports microphone errors', async () => {
  Object.assign(recorder, { phase: 'rec', secs: 12, error: 'The microphone is unavailable (NotAllowedError).' });
  renderPage(<VoiceRef readOnly={false} />);
  await userEvent.click(await screen.findByRole('button', { name: 'Stop (12 s)' }));
  expect(recorder.stop).toHaveBeenCalledOnce();
  expect(screen.getByRole('alert')).toHaveTextContent('NotAllowedError');
});

it('refuses a short take while offering another recording', async () => {
  Object.assign(recorder, { phase: 'done', seconds: 7.9, wav: new Blob(['short'], { type: 'audio/wav' }) });
  renderPage(<VoiceRef readOnly={false} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('at least 8 seconds');
  expect(screen.queryByRole('button', { name: 'Save as my voice' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Listen back' })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Record again' }));
  expect(recorder.start).toHaveBeenCalledOnce();
});

it('previews a take, uploads WAV directly, and refetches the saved reference', async () => {
  Object.assign(recorder, { phase: 'done', seconds: 8, wav: new Blob(['audio bytes'], { type: 'audio/wav' }) });
  let saved = false;
  let contentType = '';
  let body = '';
  server.use(
    http.get('/config/voice_ref', () =>
      HttpResponse.json(saved ? { ...reference, exists: true, seconds: 8 } : reference),
    ),
    http.post('/config/voice_ref', async ({ request }) => {
      contentType = request.headers.get('content-type') ?? '';
      body = await request.text();
      saved = true;
      return HttpResponse.json({ ok: true, seconds: 8, note: 'Voice enrolled.' });
    }),
  );
  const view = renderPage(<VoiceRef readOnly={false} />);
  await userEvent.click(await screen.findByRole('button', { name: 'Listen back' }));
  expect(play).toHaveBeenCalledWith('blob:voice-draft', 'voice-ref-take');
  await userEvent.click(screen.getByRole('button', { name: 'Save as my voice' }));
  expect(await screen.findByRole('status')).toHaveTextContent('Saved 8 s. Voice enrolled.');
  expect(contentType).toBe('audio/wav');
  expect(body).toBe('audio bytes');
  expect(await screen.findByRole('button', { name: 'Play reference' })).toBeVisible();
  usePlayer.setState({ key: 'voice-ref-take' });
  view.unmount();
  expect(revoke).toHaveBeenCalledWith('blob:voice-draft');
  expect(stop).toHaveBeenCalled();
});

it('keeps a failed upload available to retry and shows the server error', async () => {
  Object.assign(recorder, { phase: 'done', seconds: 12, wav: new Blob(['audio'], { type: 'audio/wav' }) });
  server.use(http.post('/config/voice_ref', () => HttpResponse.json({ error: 'Not enough speech' }, { status: 400 })));
  renderPage(<VoiceRef readOnly={false} />);
  await userEvent.click(await screen.findByRole('button', { name: 'Save as my voice' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Not enough speech');
  expect(screen.getByRole('button', { name: 'Save as my voice' })).toBeEnabled();
});

it('lets read-only viewers play the saved reference without recording or saving', async () => {
  server.use(http.get('/config/voice_ref', () => HttpResponse.json({ ...reference, exists: true, seconds: 10 })));
  renderPage(<VoiceRef readOnly />);
  await userEvent.click(await screen.findByRole('button', { name: 'Play reference' }));
  expect(play).toHaveBeenCalledWith(expect.stringMatching(/^\/config\/voice_ref\?audio=1&t=\d+$/), 'voice-ref');
  expect(screen.queryByRole('button', { name: /Record/ })).not.toBeInTheDocument();
  expect(screen.queryByText(reference.prompt)).not.toBeInTheDocument();
});

it('shows a load failure and retries the reference query', async () => {
  let calls = 0;
  server.use(
    http.get('/config/voice_ref', () =>
      ++calls === 1
        ? HttpResponse.json({ error: 'Reference unavailable' }, { status: 503 })
        : HttpResponse.json(reference),
    ),
  );
  renderPage(<VoiceRef readOnly={false} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Reference unavailable');
  await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
  expect(await screen.findByText(reference.prompt)).toBeVisible();
  await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
});
