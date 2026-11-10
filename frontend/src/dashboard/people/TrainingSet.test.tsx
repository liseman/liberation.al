import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { captionOf, PAGE, personTraining, petTraining, TrainingSet } from './TrainingSet';

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
  await user.click(screen.getByRole('button', { name: 'View all' }));
  expect(await screen.findByText(/^3 examples, the least like the rest first/)).toBeInTheDocument();
  const tiles = screen.getAllByRole('listitem') as [HTMLElement, HTMLElement, HTMLElement];
  expect(within(tiles[0]).getByText(/you named a face group · 2% like the rest/)).toBeInTheDocument();
  expect(within(tiles[1]).getByText('No picture kept')).toBeInTheDocument();
  expect(within(tiles[1]).getByText(/^date not kept · Cam0|^date not kept · cam0/)).toBeInTheDocument();
  expect(within(tiles[2]).getByText(/learned automatically/)).toBeInTheDocument();

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
  await user.click(screen.getByRole('button', { name: 'View all' }));
  await user.click(await screen.findByRole('checkbox'));
  await user.click(screen.getByRole('button', { name: 'Remove 1 selected' }));
  await user.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Remove' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('people can only be taught from this machine');
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
  await user.click(screen.getByRole('button', { name: 'View all' }));
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

it('says in a caption when a pet label has no example in the current model', () => {
  const base = { key: 't', src: null, at: null, cam: null, how: 'labeled by you', similarity: null };
  expect(captionOf({ ...base, unused: true })).toBe('date not kept · labeled by you · not used by the current model');
  expect(captionOf({ ...base, similarity: -0.2 })).toBe('date not kept · labeled by you · 0% like the rest');
});
