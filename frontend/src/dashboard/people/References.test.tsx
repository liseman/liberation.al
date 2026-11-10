import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { attachedNote, References } from './References';

it('previews a URL, requires explicit target, persists and removes a reference', async () => {
  const item = { id: 'a'.repeat(32), kind: 'image', source: 'https://example.com/photo.png', person: null };
  let attached = false;
  const posts: unknown[] = [];
  server.use(
    http.get('/people/references', () => HttpResponse.json({ references: attached ? [item] : [] })),
    http.post('/people/reference-url', () => HttpResponse.json(item)),
    http.post('/people/reference-attach', async ({ request }) => {
      posts.push(await request.json());
      attached = true;
      return HttpResponse.json({ ...item, person: 'Dana', enrollment: 'face', face: { result: 'added' } });
    }),
    http.post('/people/reference-remove', () => {
      attached = false;
      return HttpResponse.json({ ok: true });
    }),
  );
  const user = userEvent.setup();
  renderPage(<References name="Dana" />);
  await user.click(screen.getByRole('button', { name: 'Reference photos, audio, and video' }));
  await user.type(screen.getByLabelText('Direct media URL'), item.source);
  await user.click(screen.getByRole('button', { name: 'Preview URL' }));
  expect(await screen.findByAltText('Selected reference preview')).toHaveAttribute(
    'src',
    `/people/reference-media/${item.id}`,
  );
  expect(posts).toEqual([]);
  expect(screen.getByRole('button', { name: 'Confirm reference for Dana' })).toBeDisabled();
  await user.selectOptions(screen.getByLabelText('Attach reference to'), 'Dana');
  await user.click(screen.getByRole('button', { name: 'Confirm reference for Dana' }));
  await waitFor(() => expect(posts).toEqual([{ id: item.id, person: 'Dana' }]));
  expect(
    await screen.findByText('Taught their face from this photo. Remove the reference to undo.'),
  ).toBeInTheDocument();
  await user.click(await screen.findByRole('button', { name: 'Remove reference' }));
  await waitFor(() => expect(screen.queryByAltText('Selected reference preview')).not.toBeInTheDocument());
});

it('shows unsupported link errors without claiming imported and retains URL for retry', async () => {
  server.use(
    http.get('/people/references', () => HttpResponse.json({ references: [] })),
    http.post('/people/reference-url', () =>
      HttpResponse.json({ error: 'Web pages unsupported; upload a file instead.' }, { status: 400 }),
    ),
  );
  const user = userEvent.setup();
  renderPage(<References name="Dana" />);
  await user.click(screen.getByRole('button', { name: 'Reference photos, audio, and video' }));
  await user.type(screen.getByLabelText('Direct media URL'), 'https://example.com/page');
  await user.click(screen.getByRole('button', { name: 'Preview URL' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Web pages unsupported');
  expect(screen.getByLabelText('Direct media URL')).toHaveValue('https://example.com/page');
  expect(screen.queryByRole('button', { name: 'Confirm reference for Dana' })).not.toBeInTheDocument();
});

it('uploads MP3 bytes but uses server-decoded media type and does not enroll', async () => {
  server.use(
    http.get('/people/references', () => HttpResponse.json({ references: [] })),
    http.post('/people/reference-upload', () =>
      HttpResponse.json({ id: 'b'.repeat(32), kind: 'audio', source: 'Local upload', person: null }),
    ),
  );
  const user = userEvent.setup();
  renderPage(<References name="Dana" />);
  await user.click(screen.getByRole('button', { name: 'Reference photos, audio, and video' }));
  await user.upload(
    screen.getByLabelText('Reference file'),
    new File(['synthetic'], 'voice.mp3', { type: 'audio/mpeg' }),
  );
  expect(await screen.findByLabelText('Attach reference to')).toHaveValue('');
  expect(screen.getByText('voice.mp3')).toBeInTheDocument();
  expect(
    screen.getByText(/Photos with several faces, audio, and video are stored as references only/),
  ).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Confirm reference for Dana' })).toBeDisabled();
});

describe('attachedNote', () => {
  const base = { id: 'r', kind: 'image' as const, source: 's', person: 'Dana' };
  it.each([
    ['added', 'Taught their face from this photo. Remove the reference to undo.'],
    ['same', 'Stored. This face was already one of theirs.'],
    ['several', 'Stored as a reference only: it shows several faces, so it trains no one.'],
    ['none', 'Stored as a reference only: no face was found in it.'],
  ] as const)('says what a %s photo did', (result, text) => {
    expect(attachedNote({ ...base, face: { result } })).toBe(text);
  });
  it('never claims audio or video trained anything', () => {
    expect(attachedNote({ ...base, kind: 'audio' })).toBe(
      "Stored as a reference. Audio and video don't train recognition.",
    );
  });
});
