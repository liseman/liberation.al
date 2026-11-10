import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { setProblem, ZoneEditor } from './ZoneEditor';

const BOX: [number, number][] = [
  [0.1, 0.6],
  [0.3, 0.6],
  [0.3, 0.9],
  [0.1, 0.9],
];
const saved = { id: 'v1', cam: 'cam0', zones: [{ id: 'zone-aaaa', name: 'Food bowl', polygon: BOX }], at: 1 };
let posts: unknown[];

beforeEach(() => {
  posts = [];
  server.use(
    http.get('/zones', () => HttpResponse.json({ cameras: { cam0: saved } })),
    http.post('/zones', async ({ request }) => {
      const body = (await request.json()) as { zones: unknown[] };
      posts.push(body);
      return HttpResponse.json({ ok: true, definition: { ...saved, id: 'v2', zones: body.zones } });
    }),
  );
});

/** Click the overlay at shares (x, y) of a 1000 x 500 picture. */
function clickAt(overlay: Element, x: number, y: number) {
  fireEvent.click(overlay, { clientX: x * 1000, clientY: y * 500 });
}

function overlay(): SVGSVGElement {
  const svg = screen.getByRole('application', { name: /Click to add a corner/ }) as unknown as SVGSVGElement;
  vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
    left: 0,
    top: 0,
    width: 1000,
    height: 500,
  } as DOMRect);
  return svg;
}

it('draws a new zone, names it, and saves the whole set from the version it opened', async () => {
  renderPage(<ZoneEditor cam="cam0" label="Kitchen" onClose={() => {}} />);
  expect(await screen.findByDisplayValue('Food bowl')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Add a zone' }));
  const svg = overlay();
  clickAt(svg, 0.5, 0.5);
  clickAt(svg, 0.7, 0.5);
  clickAt(svg, 0.7, 0.8);
  await userEvent.click(screen.getByRole('button', { name: 'Done' }));
  await userEvent.type(screen.getByLabelText('Name of zone 2'), 'Litter box');
  await userEvent.click(screen.getByRole('button', { name: 'Save zones' }));
  await waitFor(() => expect(posts).toHaveLength(1));
  expect(posts[0]).toEqual({
    cam: 'cam0',
    base: 'v1',
    zones: [
      saved.zones[0],
      {
        name: 'Litter box',
        polygon: [
          [0.5, 0.5],
          [0.7, 0.5],
          [0.7, 0.8],
        ],
      },
    ],
  });
  expect(await screen.findByText('Saved.')).toBeInTheDocument();
});

it('draws the TV as a screen zone named TV, for tagging TV by ear', async () => {
  renderPage(<ZoneEditor cam="cam0" label="Kitchen" onClose={() => {}} />);
  await screen.findByDisplayValue('Food bowl');
  await userEvent.click(screen.getByRole('button', { name: 'Draw the TV' }));
  const svg = overlay();
  clickAt(svg, 0.1, 0.1);
  clickAt(svg, 0.4, 0.1);
  clickAt(svg, 0.4, 0.4);
  await userEvent.click(screen.getByRole('button', { name: 'Done' }));
  expect(screen.getByLabelText('Name of zone 2')).toHaveValue('TV');
  expect(screen.getAllByRole('checkbox', { name: /A screen/ })[1]).toBeChecked();
  await userEvent.click(screen.getByRole('button', { name: 'Save zones' }));
  await waitFor(() => expect(posts).toHaveLength(1));
  expect((posts[0] as { zones: unknown[] }).zones[1]).toEqual({
    name: 'TV',
    kind: 'screen',
    polygon: [
      [0.1, 0.1],
      [0.4, 0.1],
      [0.4, 0.4],
    ],
  });
});

it('keeps Done off until the outline is a zone, and Undo takes the last corner back', async () => {
  renderPage(<ZoneEditor cam="cam0" label="Kitchen" onClose={() => {}} />);
  await screen.findByDisplayValue('Food bowl');
  await userEvent.click(screen.getByRole('button', { name: 'Add a zone' }));
  const svg = overlay();
  clickAt(svg, 0.5, 0.5);
  clickAt(svg, 0.7, 0.5);
  expect(screen.getByRole('button', { name: 'Done' })).toBeDisabled();
  clickAt(svg, 0.7, 0.8);
  expect(screen.getByRole('button', { name: 'Done' })).toBeEnabled();
  await userEvent.click(screen.getByRole('button', { name: 'Undo corner' }));
  expect(screen.getByRole('button', { name: 'Done' })).toBeDisabled();
});

it('adds corners from the keyboard: arrows move the crosshair and Enter puts a corner there', async () => {
  renderPage(<ZoneEditor cam="cam0" label="Kitchen" onClose={() => {}} />);
  await screen.findByDisplayValue('Food bowl');
  await userEvent.click(screen.getByRole('button', { name: 'Add a zone' }));
  overlay().focus();
  await userEvent.keyboard('{Enter}{Shift>}{ArrowRight}{/Shift}{Enter}{Shift>}{ArrowDown}{/Shift}{Enter}');
  await userEvent.click(screen.getByRole('button', { name: 'Done' }));
  await userEvent.type(screen.getByLabelText('Name of zone 2'), 'Rug');
  await userEvent.click(screen.getByRole('button', { name: 'Save zones' }));
  await waitFor(() => expect(posts).toHaveLength(1));
  expect((posts[0] as { zones: { polygon: unknown }[] }).zones[1]?.polygon).toEqual([
    [0.5, 0.5],
    [0.55, 0.5],
    [0.55, 0.55],
  ]);
});

it('removes a zone and saves the camera with none', async () => {
  renderPage(<ZoneEditor cam="cam0" label="Kitchen" onClose={() => {}} />);
  await screen.findByDisplayValue('Food bowl');
  await userEvent.click(screen.getByRole('button', { name: 'Remove' }));
  await userEvent.click(screen.getByRole('button', { name: 'Save zones' }));
  await waitFor(() => expect(posts).toEqual([{ cam: 'cam0', base: 'v1', zones: [] }]));
});

it('marks a zone as a screen and saves its kind', async () => {
  renderPage(<ZoneEditor cam="cam0" label="Kitchen" onClose={() => {}} />);
  await screen.findByDisplayValue('Food bowl');
  const box = screen.getByRole('checkbox', { name: /A screen/ });
  expect(box).not.toBeChecked();
  await userEvent.click(box);
  await userEvent.click(screen.getByRole('button', { name: 'Save zones' }));
  await waitFor(() =>
    expect(posts).toEqual([
      { cam: 'cam0', base: 'v1', zones: [{ id: 'zone-aaaa', name: 'Food bowl', polygon: BOX, kind: 'screen' }] },
    ]),
  );
});

it('shows a saved screen zone as one, and unmarking it saves a plain zone', async () => {
  const tv = { ...saved, zones: [{ id: 'zone-aaaa', name: 'TV', polygon: BOX, kind: 'screen' }] };
  server.use(http.get('/zones', () => HttpResponse.json({ cameras: { cam0: tv } })));
  renderPage(<ZoneEditor cam="cam0" label="Kitchen" onClose={() => {}} />);
  await screen.findByDisplayValue('TV');
  const box = screen.getByRole('checkbox', { name: /A screen/ });
  expect(box).toBeChecked();
  await userEvent.click(box);
  await userEvent.click(screen.getByRole('button', { name: 'Save zones' }));
  await waitFor(() =>
    expect(posts).toEqual([{ cam: 'cam0', base: 'v1', zones: [{ id: 'zone-aaaa', name: 'TV', polygon: BOX }] }]),
  );
});

it("shows the server's refusal when someone else changed the zones first", async () => {
  server.use(
    http.post('/zones', () =>
      HttpResponse.json({ ok: false, error: 'these zones changed since you opened them' }, { status: 409 }),
    ),
  );
  renderPage(<ZoneEditor cam="cam0" label="Kitchen" onClose={() => {}} />);
  await userEvent.clear(await screen.findByDisplayValue('Food bowl'));
  await userEvent.type(screen.getByLabelText('Name of zone 1'), 'Water bowl');
  await userEvent.click(screen.getByRole('button', { name: 'Save zones' }));
  expect(await screen.findByText(/changed since you opened them/)).toBeInTheDocument();
});

it('starts a camera with no zones from version null', async () => {
  server.use(http.get('/zones', () => HttpResponse.json({ cameras: {} })));
  renderPage(<ZoneEditor cam="cam1" label="Hall" onClose={() => {}} />);
  await userEvent.click(await screen.findByRole('button', { name: 'Add a zone' }));
  const svg = overlay();
  clickAt(svg, 0.1, 0.1);
  clickAt(svg, 0.3, 0.1);
  clickAt(svg, 0.3, 0.3);
  await userEvent.click(screen.getByRole('button', { name: 'Done' }));
  await userEvent.type(screen.getByLabelText('Name of zone 1'), 'Door');
  await userEvent.click(screen.getByRole('button', { name: 'Save zones' }));
  await waitFor(() => expect(posts).toMatchObject([{ cam: 'cam1', base: null }]));
});

it.each([
  ['an unnamed zone', [{ name: ' ', polygon: BOX }], 'needs a name'],
  [
    'two zones with one name',
    [
      { name: 'Bowl', polygon: BOX },
      { name: 'bowl', polygon: BOX },
    ],
    'share a name',
  ],
])('setProblem refuses %s', (_, zones, message) => {
  expect(setProblem(zones)).toContain(message);
});
