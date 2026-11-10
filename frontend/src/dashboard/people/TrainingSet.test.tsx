import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { seenAt } from '../lib/format';
import { captionOf, detailOf, PAGE, personTraining, petTraining, TrainingSet, unlikeKeys } from './TrainingSet';

const face = (key: string, more: object = {}) => ({
  key,
  kind: 'owner',
  source: 'group',
  at: 1_800_000_000,
  cam: 'cam0',
  ref: 'group:4',
  picture: true,
  similarity: 0.8,
  ...more,
});

it("lists a person's examples on request and removes the selected ones after asking", async () => {
  let rows = [
    face('a'.repeat(24), { similarity: 0.02 }),
    face('b'.repeat(24), { picture: false, at: null, source: 'owner' }),
    face('c'.repeat(24), { source: 'learned', kind: 'learned' }),
  ];
  const posts: unknown[] = [];
  let reads = 0;
  server.use(
    http.get('/people/examples', ({ request }) => {
      reads += 1;
      expect(new URL(request.url).searchParams.get('name')).toBe('Dana');
      return HttpResponse.json({ name: 'Dana', count: rows.length, examples: rows });
    }),
    http.post('/people/examples/remove', async ({ request }) => {
      const body = (await request.json()) as { keys: string[] };
      posts.push(body);
      rows = rows.filter(r => !body.keys.includes(r.key));
      return HttpResponse.json({ ok: true, removed: body.keys.length });
    }),
  );
  const user = userEvent.setup();
  let changed = 0;
  renderPage(
    <TrainingSet
      name="Dana"
      source={personTraining('Dana')}
      onRemoved={() => {
        changed += 1;
      }}
    />,
  );
  expect(reads).toBe(0); // closed: the card reads nothing until asked
  await user.click(screen.getByRole('button', { name: 'Show' }));
  expect(await screen.findByText(/^3 examples, the least like the rest first/)).toBeInTheDocument();
  const tiles = screen.getAllByRole('listitem') as [HTMLElement, HTMLElement, HTMLElement];
  // One short caption each; the camera and how it got in are in the tile's name and hover title.
  expect(within(tiles[0]).getByText(`2% alike · ${seenAt(1_800_000_000)}`)).toBeInTheDocument();
  expect(within(tiles[0]).getByRole('checkbox')).toHaveAccessibleName(
    expect.stringMatching(/^Select the example of Dana: .*, you named a face group, 2% like the rest$/),
  );
  expect(within(tiles[1]).getByText('No picture kept')).toBeInTheDocument();
  expect(within(tiles[1]).getByText('80% alike · date not kept')).toBeInTheDocument();
  expect(within(tiles[2]).getByRole('checkbox')).toHaveAccessibleName(expect.stringMatching(/learned automatically/));

  await user.click(within(tiles[0]).getByRole('checkbox'));
  await user.click(within(tiles[1]).getByRole('checkbox'));
  await user.click(screen.getByRole('button', { name: 'Remove 2 selected' }));
  const dialog = await screen.findByRole('dialog');
  expect(dialog).toHaveTextContent('Remove 2 training images from Dana?');
  expect(posts).toEqual([]); // nothing goes until confirmed
  await user.click(within(dialog).getByRole('button', { name: 'Remove' }));
  await waitFor(() => expect(posts).toEqual([{ name: 'Dana', keys: ['a'.repeat(24), 'b'.repeat(24)] }]));
  expect(await screen.findByText(/^1 example, the least like the rest first/)).toBeInTheDocument();
  expect(changed).toBe(1);
  expect(screen.queryByRole('button', { name: /selected$/ })).not.toBeInTheDocument();
});

it('keeps what was selected and says why when a removal fails', async () => {
  server.use(
    http.get('/people/examples', () => HttpResponse.json({ examples: [face('a'.repeat(24))] })),
    http.post('/people/examples/remove', () =>
      HttpResponse.json({ ok: false, error: 'people can only be taught from this machine' }, { status: 403 }),
    ),
  );
  const user = userEvent.setup();
  renderPage(<TrainingSet name="Dana" source={personTraining('Dana')} />);
  await user.click(screen.getByRole('button', { name: 'Show' }));
  await user.click(await screen.findByRole('checkbox'));
  await user.click(screen.getByRole('button', { name: 'Remove 1 selected' }));
  await user.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Remove' }));
  expect(await screen.findByRole('status')).toHaveTextContent(
    "Couldn't remove it: people can only be taught from this machine",
  );
  expect(await screen.findByRole('button', { name: 'Remove 1 selected' })).toBeInTheDocument();
});

it("shows a large pet set a page at a time, each example's photo from where it is kept", async () => {
  const examples = Array.from({ length: PAGE + 5 }, (_, k) => ({
    track_id: `run:dog:${k}`,
    cam: 'cam0',
    start: 1_800_000_000 + k,
    end: 1_800_000_005 + k,
    labeled: 1_800_000_100,
    source: k === 0 ? 'carried' : 'owner',
    picture: k === 0 ? 'kept' : k === 1 ? 'recorded' : null,
    example: k !== 2,
    similarity: k === 2 ? null : 0.5,
  }));
  server.use(http.get('/pets/examples', () => HttpResponse.json({ pet: 'pet-1', count: examples.length, examples })));
  const user = userEvent.setup();
  const { container } = renderPage(<TrainingSet name="Rex" source={petTraining('pet-1')} />);
  await user.click(screen.getByRole('button', { name: 'Show' }));
  await screen.findByText(new RegExp(`^${PAGE + 5} examples`));
  expect(screen.getAllByRole('listitem')).toHaveLength(PAGE);
  const sources = [...container.querySelectorAll('img')].map(i => i.getAttribute('src'));
  expect(sources.slice(0, 2)).toEqual([
    '/pets/examples/photo?track=run%3Adog%3A0',
    '/pets/sightings/photo?track=run%3Adog%3A1',
  ]);
  expect(screen.getAllByText('No picture kept')).toHaveLength(PAGE - 2);
  await user.click(screen.getByRole('button', { name: `Show more (${PAGE} of ${PAGE + 5} shown)` }));
  expect(screen.getAllByRole('listitem')).toHaveLength(PAGE + 5);
});

it('captions a tile in one short line, and keeps the rest for its name', () => {
  const base = { key: 't', src: null, at: null, cam: null, how: 'labeled by you', similarity: null };
  expect(captionOf({ ...base, unused: true })).toBe('not used now · date not kept');
  expect(captionOf({ ...base, similarity: -0.2 })).toBe('0% alike · date not kept');
  expect(captionOf({ ...base, similarity: 0.861, at: 1_800_000_000 })).toBe(`86% alike · ${seenAt(1_800_000_000)}`);
  expect(detailOf({ ...base, unused: true })).toBe('date not kept, labeled by you, not used by the current model');
  expect(detailOf({ ...base, similarity: 0.5 })).toBe('date not kept, labeled by you, 50% like the rest');
});

it('marks only an example well under the rest of its set as unlike the rest', () => {
  const image = (key: string, similarity: number | null, unused = false) => ({
    key,
    src: null,
    at: null,
    cam: null,
    how: 'labeled by you',
    similarity,
    unused,
  });
  const set = [image('odd', 0), image('a', 0.86), image('b', 0.84), image('c', 0.8), image('d', 0.7)];
  expect(unlikeKeys(set)).toEqual(new Set(['odd']));
  // The ordinary spread of a set is not marked, nor is anything in a set too small to have a rest.
  expect(unlikeKeys([image('a', 0.9), image('b', 0.8), image('c', 0.72)])).toEqual(new Set());
  expect(unlikeKeys([image('odd', 0), image('a', 0.9)])).toEqual(new Set());
  // An example the current model doesn't use, or one with nothing to compare, is never the odd one.
  expect(unlikeKeys([...set.slice(1), image('unused', 0, true), image('none', null)])).toEqual(new Set());
});

it('shows the odd example with an Unlike the rest badge', async () => {
  server.use(
    http.get('/people/examples', () =>
      HttpResponse.json({
        examples: [
          face('a'.repeat(24), { similarity: 0 }),
          face('b'.repeat(24), { similarity: 0.86 }),
          face('c'.repeat(24), { similarity: 0.82 }),
          face('d'.repeat(24), { similarity: 0.8 }),
        ],
      }),
    ),
  );
  const user = userEvent.setup();
  renderPage(<TrainingSet name="Dana" source={personTraining('Dana')} />);
  await user.click(screen.getByRole('button', { name: 'Show' }));
  const tiles = await screen.findAllByRole('listitem');
  expect(within(tiles[0] as HTMLElement).getByText('Unlike the rest')).toBeVisible();
  expect(within(tiles[0] as HTMLElement).getByRole('checkbox')).toHaveAccessibleName(
    expect.stringMatching(/^Select the example of Dana: unlike the rest, /),
  );
  expect(screen.getAllByText('Unlike the rest')).toHaveLength(1);
});

it('shows square placeholders while the set loads, and a retry when it fails', async () => {
  let fail = true;
  let release = () => {};
  const held = new Promise<void>(resolve => {
    release = resolve;
  });
  server.use(
    http.get('/people/examples', async () => {
      await held;
      return fail
        ? HttpResponse.json({ ok: false, error: 'the face store is busy' }, { status: 503 })
        : HttpResponse.json({ examples: [face('a'.repeat(24))] });
    }),
  );
  const user = userEvent.setup();
  renderPage(<TrainingSet name="Dana" source={personTraining('Dana')} />);
  await user.click(screen.getByRole('button', { name: 'Show' }));
  expect(screen.getByRole('list', { name: 'Loading the training images' })).toHaveAttribute('aria-busy', 'true');
  release();
  expect(await screen.findByText("Couldn't load the training images")).toBeVisible();
  fail = false;
  await user.click(screen.getByRole('button', { name: /retry/i }));
  expect(await screen.findByText(/^1 example, the least like the rest first/)).toBeInTheDocument();
});
