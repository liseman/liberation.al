import type { QueryClient } from '@tanstack/react-query';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { UNKNOWN_GROUPS_KEY } from '../whosthis/api';
import { FacePicker } from './FacePicker';
import { type LiveFaceDraft, takeLiveFace } from './liveFace';
import { NewPerson } from './Roster';

const draft: LiveFaceDraft = {
  source: { camera: 'cam0', at: 100, track: 'run:cam0:7', group: 5 },
  image: 'data:image/jpeg;base64,cGljdHVyZQ==',
};
let posts: unknown[];
beforeEach(() => {
  posts = [];
  server.use(
    http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana' }, { name: 'Alex' }] })),
    http.post('/people/claim', async ({ request }) => {
      posts.push(await request.json());
      return HttpResponse.json({ name: 'Dana' });
    }),
    http.get('/transcript', () => HttpResponse.json({ entries: [] })),
  );
});
afterEach(() => {
  takeLiveFace();
  history.replaceState(null, '', '/');
});
it('claims an existing person only after explicit confirmation, retaining selected evidence', async () => {
  const close = vi.fn();
  renderPage(<FacePicker draft={draft} onClose={close} />);
  expect(screen.getByRole('img', { name: 'Selected face' })).toHaveAttribute('src', draft.image);
  await userEvent.click(await screen.findByRole('combobox', { name: 'Person' }));
  await userEvent.click(await screen.findByRole('option', { name: 'Dana' }));
  expect(posts).toEqual([]);
  await userEvent.click(screen.getByRole('button', { name: 'Use selected person' }));
  await waitFor(() =>
    expect(posts).toEqual([{ kind: 'face', id: 5, name: 'Dana', face_source: draft.source, face_image: draft.image }]),
  );
  expect(close).toHaveBeenCalledOnce();
  expect(location.pathname + location.hash).toBe('/');
});
it('More details in People transfers the same evidence without saving or guessing a name', async () => {
  renderPage(<FacePicker draft={draft} onClose={() => {}} />);
  await userEvent.click(screen.getByRole('link', { name: 'More details in People' }));
  expect(location.pathname + location.hash).toBe('/people#new-person');
  expect(takeLiveFace()).toEqual(draft);
  expect(posts).toEqual([]);
});
it('creates a new person only on an explicit Add choice and closes in place', async () => {
  const close = vi.fn();
  renderPage(<FacePicker draft={draft} onClose={close} />);
  const input = await screen.findByRole('combobox', { name: 'Person' });
  await userEvent.type(input, 'Da');
  expect(posts).toEqual([]);
  expect(screen.queryByRole('option', { name: 'Add Da' })).not.toBeInTheDocument();
  await userEvent.clear(input);
  await userEvent.type(input, 'Morgan');
  expect(posts).toEqual([]);
  await userEvent.click(await screen.findByRole('option', { name: 'Add “Morgan”' }));
  await waitFor(() => expect(close).toHaveBeenCalledOnce());
  expect(posts).toEqual([{ kind: 'face', id: 5, name: 'Morgan', face_source: draft.source, face_image: draft.image }]);
  expect(location.pathname + location.hash).toBe('/');
});
it('keeps selected pixels and name after a failed Add, then retries the same evidence', async () => {
  const close = vi.fn();
  server.use(http.post('/people/claim', () => HttpResponse.json({ error: 'Select the face again' }, { status: 409 })));
  renderPage(<FacePicker draft={draft} onClose={close} />);
  const input = await screen.findByRole('combobox', { name: 'Person' });
  await userEvent.type(input, 'Morgan');
  await userEvent.click(await screen.findByRole('option', { name: 'Add “Morgan”' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Select the face again');
  expect(close).not.toHaveBeenCalled();
  expect(input).toHaveValue('Morgan');
  expect(screen.getByRole('img', { name: 'Selected face' })).toHaveAttribute('src', draft.image);
  server.use(
    http.post('/people/claim', async ({ request }) => {
      posts.push(await request.json());
      return HttpResponse.json({ name: 'Morgan' });
    }),
  );
  await userEvent.click(input);
  await userEvent.click(await screen.findByRole('option', { name: 'Add “Morgan”' }));
  await waitFor(() => expect(close).toHaveBeenCalledOnce());
  expect(posts).toEqual([{ kind: 'face', id: 5, name: 'Morgan', face_source: draft.source, face_image: draft.image }]);
});
it('the People creation form keeps the selected crop, blank name, and enrolls only on explicit Add', async () => {
  renderPage(<NewPerson opened onClose={() => {}} onAdded={() => {}} faceDraft={draft} />);
  expect(screen.getByRole('textbox', { name: 'Name' })).toHaveValue('');
  expect(screen.getByRole('img', { name: 'Selected face' })).toHaveAttribute('src', draft.image);
  expect(posts).toEqual([]);
  await userEvent.type(screen.getByRole('textbox', { name: 'Name' }), 'Dana');
  await userEvent.click(screen.getByRole('button', { name: 'Add Dana' }));
  await waitFor(() =>
    expect(posts).toEqual([{ kind: 'face', id: 5, name: 'Dana', face_source: draft.source, face_image: draft.image }]),
  );
});
it('keeps source failures visible without navigating or creating anyone', async () => {
  server.use(http.post('/people/claim', () => HttpResponse.json({ error: 'Select the face again' }, { status: 409 })));
  renderPage(<NewPerson opened onClose={() => {}} onAdded={() => {}} faceDraft={draft} />);
  await userEvent.type(screen.getByRole('textbox', { name: 'Name' }), 'Dana');
  await userEvent.click(screen.getByRole('button', { name: 'Add Dana' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Select the face again');
  expect(screen.getByRole('textbox', { name: 'Name' })).toHaveValue('Dana');
  expect(posts).toEqual([]);
});

it('copies the selected source when making a clean displayed-face crop', async () => {
  const { prepareLiveFace } = await import('./liveFace');
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  const source = { camera: 'cam0', at: 100, track: 'run:cam0:7', group: 5 };
  const drawImage = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage,
  } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue(draft.image);
  const result = prepareLiveFace(
    {
      id: 'f7',
      kind: 'face',
      label: "Who's this?",
      tone: 'unknown',
      box: [0.25, 0.25, 0.5, 0.75],
      left: 0,
      top: 0,
      width: 100,
      height: 100,
      faceSource: source,
    },
    canvas,
  );
  source.group = 99;
  source.track = 'different';
  expect(result.source).toEqual({ camera: 'cam0', at: 100, track: 'run:cam0:7', group: 5 });
  expect(drawImage).toHaveBeenCalledWith(canvas, 160, 90, 160, 180, 0, 0, 160, 180);
  vi.restoreAllMocks();
});

// Naming a live face takes its group off the waiting list and the count at once, wherever it was named.
const WAITING = {
  groups: [
    { id: 5, label: 'Face 5' },
    { id: 6, label: 'Face 6' },
  ],
  count: 3,
  skipped: [],
};
const waitingNow = (client: QueryClient) => client.getQueryData<typeof WAITING>(UNKNOWN_GROUPS_KEY);

it('lowers the faces waiting once the picked person is saved', async () => {
  const { client } = renderPage(<FacePicker draft={draft} onClose={() => {}} />);
  client.setQueryData(UNKNOWN_GROUPS_KEY, WAITING);
  await userEvent.click(await screen.findByRole('combobox', { name: 'Person' }));
  await userEvent.click(await screen.findByRole('option', { name: 'Dana' }));
  await userEvent.click(screen.getByRole('button', { name: 'Use selected person' }));
  await waitFor(() => expect(waitingNow(client)?.count).toBe(2));
  expect(waitingNow(client)?.groups.map(g => g.id)).toEqual([6]);
});
it('lowers the faces waiting when People adds someone from a live face', async () => {
  const { client } = renderPage(<NewPerson opened onClose={() => {}} onAdded={() => {}} faceDraft={draft} />);
  client.setQueryData(UNKNOWN_GROUPS_KEY, WAITING);
  await userEvent.type(screen.getByRole('textbox', { name: 'Name' }), 'Dana');
  await userEvent.click(screen.getByRole('button', { name: 'Add Dana' }));
  await waitFor(() => expect(waitingNow(client)?.count).toBe(2));
  expect(waitingNow(client)?.groups.map(g => g.id)).toEqual([6]);
});
it('keeps the faces waiting when the server refuses the live face', async () => {
  server.use(http.post('/people/claim', () => HttpResponse.json({ error: 'Select the face again' }, { status: 409 })));
  const { client } = renderPage(<NewPerson opened onClose={() => {}} onAdded={() => {}} faceDraft={draft} />);
  client.setQueryData(UNKNOWN_GROUPS_KEY, WAITING);
  await userEvent.type(screen.getByRole('textbox', { name: 'Name' }), 'Dana');
  await userEvent.click(screen.getByRole('button', { name: 'Add Dana' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Select the face again');
  expect(waitingNow(client)).toEqual(WAITING);
});
