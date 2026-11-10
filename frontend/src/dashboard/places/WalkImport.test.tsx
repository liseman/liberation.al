import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { WalkImports, WalkJob } from './api';
import { refusal, resultText, stepText, WalkImport } from './WalkImport';

const NAMES = new Map([['p1', 'Home']]);
const opened: string[] = [];
const drop = (names = NAMES) => <WalkImport names={names} onOpen={id => opened.push(id)} />;

const job = (over: Partial<WalkJob> = {}): WalkJob => ({
  id: 'j1',
  name: 'walk.zip',
  kind: 'stray',
  status: 'importing',
  detail: '',
  error: '',
  result: {},
  retry: false,
  started: 1762500000,
  finished: null,
  ...over,
});

/** Serve GET /places/walk from `state`, which a test changes as its import moves on. */
function serve(state: { job: WalkJob | null }, over: Partial<WalkImports> = {}) {
  server.use(http.get('/places/walk', () => HttpResponse.json({ max_mb: 4096, video: true, ...over, ...state })));
}

const zip = (name = 'walk.zip') => new File(['zip bytes'], name, { type: 'application/zip' });
const chooser = () => screen.findByLabelText('Choose a walk to import');
const button = () => screen.getByRole('button', { name: 'Choose file' });

it.each([
  ['a zip', { name: 'Walk.ZIP', size: 5 }, true, null],
  ['a video where one is taken', { name: 'hall.MOV', size: 5 }, true, null],
  ['a video where none is taken', { name: 'hall.mov', size: 5 }, false, /plain video can't be imported/],
  ['some other file', { name: 'notes.txt', size: 5 }, true, /or a video of the walk/],
  ['some other file, no video', { name: 'notes.txt', size: 5 }, false, /^Drop the zip Stray Scanner exports\.$/],
  ['an empty file', { name: 'walk.zip', size: 0 }, true, /empty/],
  ['a file over the limit', { name: 'walk.zip', size: 2 * 2 ** 20 + 1 }, true, /larger than the 2 MB/],
] as [string, { name: string; size: number }, boolean, RegExp | null][])(
  'checks %s before sending it',
  (_, file, video, why) => {
    const said = refusal(file, file.size > 2 ** 20 ? 2 : 4096, video);
    if (why === null) expect(said).toBeNull();
    else expect(said).toMatch(why);
  },
);

it('says what a running import is doing, and what a finished one did', () => {
  expect(stepText(job({ status: 'extracting' }))).toBe('Unpacking walk.zip');
  expect(stepText(job({ detail: '12 frames from 40 posed frames' }))).toBe(
    'Importing walk.zip: 12 frames from 40 posed frames',
  );
  const into = job({ status: 'done', result: { final_place: 'p1', new_place: 'False', registered: 'map' } });
  expect(resultText(into, 'Home')).toBe("walk.zip went into Home: it fits that place's map.");
  expect(resultText(into, undefined)).toContain('went into a place you already have');
  const made = job({ status: 'done', result: { final_place: 'p9', new_place: 'True', registered: 'None' } });
  expect(resultText(made, 'Place 3')).toBe(
    'walk.zip matched no place you already have, so it became a new place: Place 3.',
  );
});

it('uploads a chosen zip as its body with no place named, then says which place the walk went into', async () => {
  const state: { job: WalkJob | null } = { job: null };
  const got: { place: string | null; name: string | null; type: string | null; body: string }[] = [];
  serve(state);
  server.use(
    http.post('/places/walk/upload', async ({ request }) => {
      const q = new URL(request.url).searchParams;
      got.push({
        place: q.get('place'), // none: the import works out where the walk belongs
        name: q.get('name'),
        type: request.headers.get('Content-Type'),
        body: await request.text(),
      });
      state.job = job({ status: 'extracting' });
      return HttpResponse.json({ job: state.job }, { status: 202 });
    }),
  );
  const { client } = renderPage(drop());
  client.setQueryData(['places'], { places: [] });
  await userEvent.upload(await chooser(), zip('my walk.zip'));
  expect(await screen.findByText('Unpacking walk.zip')).toBeInTheDocument();
  expect(got).toEqual([{ place: null, name: 'my walk.zip', type: 'application/octet-stream', body: 'zip bytes' }]);
  expect(button()).toBeDisabled(); // one import at a time
  state.job = job({
    status: 'done',
    result: { final_place: 'p1', new_place: 'False', registered: 'map' },
    finished: 1762500300,
  });
  expect(
    await screen.findByText("walk.zip went into Home: it fits that place's map.", {}, { timeout: 4000 }),
  ).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Open Home' }));
  expect(opened).toEqual(['p1']);
  // the places are marked to be read again, so the new walk shows
  await waitFor(() => expect(client.getQueryState(['places'])?.isInvalidated).toBe(true));
});

it('takes a file dropped on it', async () => {
  const state: { job: WalkJob | null } = { job: null };
  serve(state);
  server.use(
    http.post('/places/walk/upload', () => {
      state.job = job();
      return HttpResponse.json({ job: state.job }, { status: 202 });
    }),
  );
  renderPage(drop());
  await chooser();
  const dataTransfer = { files: [zip()], types: ['Files'] };
  fireEvent.drop(screen.getByRole('region', { name: 'Import a walk' }), { dataTransfer });
  expect(await screen.findByText('Importing walk.zip')).toBeInTheDocument();
});

it("shows the server's reason when an upload is refused, and sends nothing for a file that can't be a walk", async () => {
  let posts = 0;
  serve({ job: null });
  server.use(
    http.post('/places/walk/upload', () => {
      posts++;
      return HttpResponse.json(
        { error: "That zip isn't a whole Stray Scanner export. It is missing: depth/ (the LiDAR depth frames)." },
        { status: 400 },
      );
    }),
  );
  renderPage(drop());
  await userEvent.upload(await chooser(), zip());
  expect(await screen.findByRole('alert')).toHaveTextContent('It is missing: depth/ (the LiDAR depth frames).');
  fireEvent.drop(screen.getByRole('region', { name: 'Import a walk' }), {
    dataTransfer: { files: [new File(['x'], 'notes.txt')], types: ['Files'] },
  });
  expect(await screen.findByRole('alert')).toHaveTextContent('Drop the zip Stray Scanner exports');
  expect(posts).toBe(1);
});

it('shows an import still running after a reload, with its newest step', async () => {
  serve({ job: job({ detail: 'placing the walk by its owner plan' }) });
  renderPage(drop());
  expect(await screen.findByText('Importing walk.zip: placing the walk by its owner plan')).toBeInTheDocument();
  expect(button()).toBeDisabled();
});

it('shows why an import failed and runs it again from the files it kept', async () => {
  const state = { job: job({ status: 'failed', error: 'the GPU has 1.0 GB free; try later', retry: true }) };
  let retried = 0;
  serve(state);
  server.use(
    http.post('/places/walk/retry', () => {
      retried++;
      state.job = job({ status: 'importing' });
      return HttpResponse.json({ job: state.job }, { status: 202 });
    }),
  );
  renderPage(drop());
  expect(await screen.findByRole('alert')).toHaveTextContent('the GPU has 1.0 GB free; try later');
  await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
  expect(await screen.findByText('Importing walk.zip')).toBeInTheDocument();
  expect(retried).toBe(1);
});

it('offers no second try when a failed import kept nothing, and forgets it when dismissed', async () => {
  const state: { job: WalkJob | null } = { job: job({ status: 'failed', error: 'the zip was cut short' }) };
  serve(state);
  server.use(
    http.post('/places/walk/dismiss', () => {
      state.job = null;
      return HttpResponse.json({ job: null });
    }),
  );
  renderPage(drop());
  expect(await screen.findByRole('alert')).toHaveTextContent('the zip was cut short');
  expect(screen.queryByRole('button', { name: 'Try again' })).toBeNull();
  await userEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
  await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
  expect(button()).toBeEnabled();
});

it('says a walk that matched no place made a new one, and names it', async () => {
  serve({ job: job({ status: 'done', result: { final_place: 'p9', new_place: 'True', registered: 'None' } }) });
  renderPage(drop(new Map([...NAMES, ['p9', 'Place 2']])));
  expect(await screen.findByText('New place from a walk')).toBeInTheDocument();
  expect(
    screen.getByText('walk.zip matched no place you already have, so it became a new place: Place 2.'),
  ).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open Place 2' })).toBeInTheDocument();
});

it('offers no place to open until the places list has the new one', async () => {
  serve({ job: job({ status: 'done', result: { final_place: 'p9', new_place: 'True', registered: 'None' } }) });
  renderPage(drop());
  expect(
    await screen.findByText('walk.zip matched no place you already have, so it became a new place.'),
  ).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /^Open/ })).toBeNull();
});

it('offers a plain video only where the machine takes one', async () => {
  serve({ job: null }, { video: false });
  renderPage(drop());
  expect(await chooser()).toHaveAttribute('accept', '.zip');
  expect(screen.queryByText(/plain video/)).toBeNull();
});
