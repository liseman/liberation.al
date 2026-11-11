// What the Places page says "Learn surroundings" is doing (#283, #294): collecting pictures for the 3D map being
// rebuilt with a new method. Each state, the pictures kept, each camera's row, and the badge.
import { fireEvent, screen, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import {
  cameraRows,
  LEARNING_OFF_MS,
  LEARNING_ON_MS,
  type LearningStatus,
  learningEvery,
  learningState,
  REBUILDING,
} from './learning';
import { PlacesTab } from './PlacesTab';

const status = (more: Partial<LearningStatus> = {}): LearningStatus => ({
  on: true,
  sampling: ['front', 'hall'],
  kept: 37,
  cameras: {},
  ...more,
});

it.each([
  [
    'off',
    status({ on: false, sampling: [] }),
    'Not collecting pictures',
    'Turn on Learn surroundings on the Live tab to collect them.',
  ],
  [
    'unchosen',
    status({ sampling: [] }),
    'No camera chosen',
    'Choose the cameras to collect pictures from beside Learn surroundings on the Live tab.',
  ],
  ['collecting', status(), 'Collecting pictures from 2 cameras', REBUILDING],
])('leads with one state: %s', (kind, told, title, next) => {
  expect(learningState(told)).toMatchObject({ kind, title, next });
});

it('gives each sampled camera a row: keeping frames, or why none was kept lately', () => {
  const cameras = {
    front: { recent: { same: 5, copy: 4, person: 3 }, last_kept: 1.8e9 - 120 }, // 9 of 12 found nothing new
    hall: { recent: { person: 7, same: 5 }, last_kept: null },
    porch: { recent: { dark: 12 }, last_kept: null },
    yard: { recent: { blurred: 7, same: 5 }, last_kept: null },
    desk: { recent: { same: 11, view: 1 }, last_kept: 1.8e9 - 30 }, // kept one lately
    attic: { recent: {}, last_kept: null }, // not looked at yet
    shed: { recent: { same: 12 }, last_kept: null }, // not sampled any more: no row
  };
  const sampling = ['front', 'hall', 'porch', 'yard', 'desk', 'attic', 'unseen'];
  const rows = cameraRows(status({ sampling, cameras, frames: { front: 'a'.repeat(32) } }), 1.8e9);
  expect(rows.map(r => [r.camera, r.tone, r.label])).toEqual([
    ['front', 'none', 'Nothing new in view'],
    ['hall', 'none', 'Person or animal in view'],
    ['porch', 'none', 'Too dark'],
    ['yard', 'none', 'Picture is blurred'],
    ['desk', 'ok', 'Keeping pictures'],
    ['attic', 'none', 'Not looked at yet'],
    ['unseen', 'none', 'Not looked at yet'],
  ]);
  expect(rows[0]).toMatchObject({ detail: 'Last kept a picture 2 min ago.', frame: 'a'.repeat(32) });
  expect(rows[1]).toMatchObject({ detail: null, frame: null });
});

it('asks less often while it is off', () => {
  expect(learningEvery(status())).toBe(LEARNING_ON_MS);
  expect(learningEvery(status({ on: false }))).toBe(LEARNING_OFF_MS);
  expect(learningEvery(undefined)).toBe(LEARNING_OFF_MS);
});

// ---- on the page

const PLAN = 'plan "Home" walls 0.2/0.1\nroom kitchen "Kitchen" kitchen rect 0,0 4x3\n';
const place = (id: string, name: string) => ({
  id,
  name,
  named: true,
  plan: PLAN,
  plan_source: 'machine' as const,
  machine_newer: false,
  revision: { id: `${id}-r1`, at: 1.8e9, producer: 'places reset v1' },
  plan_revision: `${id}-r1`,
  machine_plan: null,
  last_visit: null,
});

function serve(told: LearningStatus | null) {
  server.use(
    http.get('/places', () => HttpResponse.json({ places: [place('p1', 'Home'), place('p2', 'Cabin')] })),
    http.get('/places/plan/history', () => HttpResponse.json({ place: 'p1', versions: [], previous: null })),
    http.get('/places/positions', () => HttpResponse.json({ error: 'none' }, { status: 404 })),
    ...(told ? [http.get('/surroundings/status', () => HttpResponse.json(told))] : []),
  );
}
const panel = () => screen.findByRole('region', { name: 'Learning surroundings' });
const state = async () => within(await panel()).getByRole('status');
/** The number shown over `label`. */
const stat = (label: string) => screen.getByText(label).parentElement?.querySelector('dd')?.textContent;

beforeEach(() => {
  history.replaceState(null, '', '/places');
});

it('shows no badge and no panel while it is off and nothing was ever kept', async () => {
  serve(null);
  renderPage(<PlacesTab examples={[]} />);
  await screen.findByRole('button', { name: /Home/ });
  expect(screen.queryByText('Learning')).toBeNull();
  expect(screen.queryByRole('region', { name: 'Learning surroundings' })).toBeNull();
});

it('says it is off, with the pictures kept, once some were', async () => {
  serve(status({ on: false, sampling: [], kept: 12 }));
  renderPage(<PlacesTab examples={[]} />);
  expect((await state()).getAttribute('data-state')).toBe('off');
  expect(stat('Pictures kept')).toBe('12');
  expect(within(await panel()).queryByRole('list', { name: 'Cameras learned from' })).toBeNull();
  expect(screen.queryByText('Learning')).toBeNull(); // no badge on a place
});

it('marks each place Learning while it collects, and says the 3D map is being rebuilt', async () => {
  serve(status());
  renderPage(<PlacesTab examples={[]} />);
  expect((await state()).textContent).toBe(`Collecting pictures from 2 cameras${REBUILDING}`);
  expect(stat('Pictures kept')).toBe('37');
  expect(screen.getAllByText('Learning')).toHaveLength(2); // one on each place in the list
  expect(within(await panel()).queryByRole('progressbar')).toBeNull(); // no map step to show
});

it('gives each camera a row with its name, how it is doing, and its newest kept frame when there is one', async () => {
  const frame = 'ab'.repeat(16);
  serve({
    ...status({ frames: { front: frame } }),
    cameras: { front: { recent: { person: 9, same: 3 }, last_kept: 1.8e9 } },
  });
  renderPage(<PlacesTab examples={[]} />);
  const rows = within(await within(await panel()).findByRole('list', { name: 'Cameras learned from' })).getAllByRole(
    'listitem',
  );
  expect(rows.map(r => r.textContent)).toEqual(['frontPerson or animal in view', 'hallNot looked at yet']);
  const picture = within(rows[0] as HTMLElement).getByRole('img', { name: 'The newest view kept from front' });
  expect(picture.getAttribute('src')).toBe(`/surroundings/thumb/${frame}.jpg`);
  expect(within(rows[1] as HTMLElement).queryByRole('img')).toBeNull(); // no frame of the hall: a plain tile
  // a picture that doesn't load gives way to the plain tile
  fireEvent.error(picture);
  expect(within(rows[0] as HTMLElement).queryByRole('img')).toBeNull();
});
