// What the Places page says "Learn surroundings" is doing (#283, #294): each state, the numbers, each camera's row,
// the bar's kind, and the badge.
import { fireEvent, screen, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import {
  cameraRows,
  LEARNING_OFF_MS,
  LEARNING_ON_MS,
  LEARNING_RUNNING_MS,
  type LearningStatus,
  type LearningStep,
  learningEvery,
  learningState,
  learningStats,
  mapTotals,
  measured,
  workingOn,
} from './learning';
import { PlacesTab } from './PlacesTab';

const status = (more: Partial<LearningStatus> = {}): LearningStatus => ({
  on: true,
  sampling: ['front', 'hall'],
  kept: 37,
  waiting: 0,
  running: false,
  step: null,
  last_step: null,
  cameras: {},
  ...more,
});
const step = (more: Partial<LearningStep>): LearningStep => ({
  phase: null,
  frames: null,
  place: null,
  views: null,
  step: null,
  steps: null,
  ...more,
});
const running = (more: Partial<LearningStep>) => status({ running: true, waiting: 2, step: step(more) });

const watching = 'Watching 2 cameras for new views.';
it.each([
  [
    'pairing',
    running({ phase: 'pairing', frames: 2 }),
    'placing',
    'Placing 2 new views',
    'Matching them against the map. Training comes next.',
  ],
  [
    'depth',
    running({ phase: 'depth', frames: 1 }),
    'placing',
    'Placing 1 new view',
    'Measuring their depth. Training comes next.',
  ],
  [
    'training, before the trainer says a step',
    running({ phase: 'training', views: 2, steps: 3000 }),
    'training',
    'Training the splat',
    'With 2 new views.',
  ],
  [
    'training',
    running({ phase: 'training', views: 2, step: 1200, steps: 3000 }),
    'training',
    'Training the splat',
    'Step 1,200 of 3,000, with 2 new views.',
  ],
  ['writing', running({ phase: 'writing', views: 2 }), 'writing', 'Saving the map', 'Writing the new map and splat.'],
  [
    'a step that has not said what it is doing',
    status({ running: true }),
    'working',
    'Working on the map',
    'A step is running.',
  ],
  [
    'a step still running after the switch went off',
    status({ on: false, running: true }),
    'working',
    'Working on the map',
    'A step is running.',
  ],
  ['frames waiting', status({ waiting: 3 }), 'waiting', '3 frames waiting', 'The next step starts by itself.'],
  ['one frame waiting', status({ waiting: 1 }), 'waiting', '1 frame waiting', 'The next step starts by itself.'],
  ['up to date', status({ last_step: 1.8e9 }), 'current', 'Up to date', watching],
  [
    'nothing kept yet',
    status({ kept: 0 }),
    'watching',
    'Nothing kept yet',
    'Watching 2 cameras for views worth keeping.',
  ],
  [
    'nothing kept now, with a map already',
    status({ kept: 0, places: { p1: { views: 4, improved: null } } }),
    'current',
    'Up to date',
    watching,
  ],
  [
    'no camera chosen',
    status({ sampling: [] }),
    'unchosen',
    'No camera chosen',
    'Choose the cameras to learn from beside Learn surroundings on the Live tab.',
  ],
  [
    'off',
    status({ on: false, sampling: [] }),
    'off',
    'Learning is off',
    'Turn on Learn surroundings on the Live tab to carry on.',
  ],
] as const)('leads with one state: %s', (_, told, kind, title, next) => {
  expect(learningState(told)).toMatchObject({ kind, title, next });
});

it('reads as fine except where the owner has something to do', () => {
  expect(learningState(status()).tone).toBe('ok');
  expect(learningState(status({ sampling: [] })).tone).toBe('warn');
  expect(learningState(status({ on: false })).tone).toBe('none');
});

it('measures the bar only while the training says how far it is', () => {
  expect(measured(running({ phase: 'training', step: 1200, steps: 3000 }))).toBe(40);
  expect(measured(running({ phase: 'training', steps: 3000 }))).toBeNull();
  expect(measured(running({ phase: 'pairing', frames: 2 }))).toBeNull();
  // a count left from a step that ended measures nothing
  expect(measured(status({ step: step({ phase: 'training', step: 1200, steps: 3000 }) }))).toBeNull();
});

it('gives the numbers as numbers: the views in every map, and the frames kept, waiting, and parked', () => {
  const told = status({
    waiting: 3,
    parked: 2,
    places: {
      p1: { views: 120, improved: 1.7e9 },
      p2: { views: 8, improved: 1.8e9 },
      p3: { views: 0, improved: null },
    },
  });
  expect(learningStats(told).map(s => [s.label, s.value])).toEqual([
    ['Views', 128],
    ['Kept', 37],
    ['Waiting', 3],
    ['Parked', 2],
  ]);
  expect(mapTotals(told)).toEqual({ views: 128, improved: 1.8e9 }); // the newest improvement of any place
  // an older service says nothing of parking or of the places
  expect(learningStats(status()).map(s => s.value)).toEqual([0, 37, 0, 0]);
  expect(mapTotals(status())).toEqual({ views: 0, improved: null });
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
    ['desk', 'ok', 'Keeping frames'],
    ['attic', 'none', 'Not looked at yet'],
    ['unseen', 'none', 'Not looked at yet'],
  ]);
  expect(rows[0]).toMatchObject({ detail: 'Last kept a frame 2 min ago.', frame: 'a'.repeat(32) });
  expect(rows[1]).toMatchObject({ detail: null, frame: null });
});

it('says first that a camera is parked, and why, whether or not it is sampled now', () => {
  const told = status({
    parked: 3,
    parked_cameras: { hall: 'overlap', front: 'facing', attic: 'something new' },
    cameras: { front: { recent: { view: 2 }, last_kept: 1.8e9 } },
  });
  expect(cameraRows(told).map(r => [r.camera, r.tone, r.label, r.detail])).toEqual([
    ['front', 'warn', 'Needs a facing', 'Its spot on the plan has no facing. Give it one in the plan editor.'],
    [
      'hall',
      'warn',
      'No spot on the plan',
      'Its view overlaps nothing in the map yet, and it has no spot on the plan to place it by.',
    ],
    // a reason this page doesn't know is said plainly
    ['attic', 'warn', "Can't be placed yet", "Its kept frame can't be placed in the map yet."],
  ]);
});

it('asks often while a step runs, less while learning is on, and seldom while it is off', () => {
  expect(learningEvery(running({ phase: 'depth' }))).toBe(LEARNING_RUNNING_MS);
  expect(learningEvery(status())).toBe(LEARNING_ON_MS);
  expect(learningEvery(status({ on: false }))).toBe(LEARNING_OFF_MS);
  expect(learningEvery(undefined)).toBe(LEARNING_OFF_MS);
});

it('takes a step to be working on a place until it names another', () => {
  expect(workingOn(running({ phase: 'pairing' }), 'p1')).toBe(true);
  expect(workingOn(running({ phase: 'training', place: 'p1' }), 'p1')).toBe(true);
  expect(workingOn(running({ phase: 'training', place: 'p2' }), 'p1')).toBe(false);
  expect(workingOn(status(), 'p1')).toBe(false);
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
  revision: { id: `${id}-r1`, at: 1.8e9, producer: 'surroundings v1' },
  plan_revision: `${id}-r1`,
  machine_plan: null,
  last_visit: null,
});

function serve(told: LearningStatus | null, places: LearningStatus['places'] = {}) {
  server.use(
    http.get('/places', () => HttpResponse.json({ places: [place('p1', 'Home'), place('p2', 'Cabin')] })),
    http.get('/places/plan/history', () => HttpResponse.json({ place: 'p1', versions: [], previous: null })),
    http.get('/places/positions', () => HttpResponse.json({ error: 'none' }, { status: 404 })),
    ...(told ? [http.get('/surroundings/status', () => HttpResponse.json({ places, ...told }))] : []),
  );
}
const list = () => screen.getByRole('navigation', { name: 'Places' });
const panel = () => screen.findByRole('region', { name: 'Learning surroundings' });
const state = async () => within(await panel()).getByRole('status');
/** The number shown over `label`. */
const stat = (label: string) => screen.getByText(label).parentElement?.querySelector('dd')?.textContent;

beforeEach(() => {
  history.replaceState(null, '', '/places');
});

it('shows no badge and no panel while learning is off and nothing was ever learned', async () => {
  serve(null);
  renderPage(<PlacesTab examples={[]} />);
  await screen.findByRole('button', { name: /Home/ });
  expect(screen.queryByText('Learning')).toBeNull();
  expect(screen.queryByRole('region', { name: 'Learning surroundings' })).toBeNull();
});

it('says learning is off, with what was learned, once something was', async () => {
  serve(status({ on: false, sampling: [], kept: 12 }), { p1: { views: 9, improved: 1.8e9 } });
  renderPage(<PlacesTab examples={[]} />);
  expect((await state()).textContent).toContain('Learning is off');
  expect((await state()).getAttribute('data-state')).toBe('off');
  expect(stat('Views')).toBe('9');
  expect(stat('Kept')).toBe('12');
  expect(within(await panel()).queryByRole('list', { name: 'Cameras learned from' })).toBeNull();
  expect(screen.queryByText('Learning')).toBeNull(); // no badge on a place
});

it('marks each place Learning while it is on, still while nothing runs, with the numbers and no bar', async () => {
  serve(status({ last_step: 1.8e9, waiting: 0, parked: 1 }), { p1: { views: 128, improved: 1.8e9 } });
  renderPage(<PlacesTab examples={[]} />);
  expect((await state()).textContent).toBe('Up to dateWatching 2 cameras for new views.');
  expect(['Views', 'Kept', 'Waiting', 'Parked'].map(stat)).toEqual(['128', '37', '0', '1']);
  expect((await panel()).textContent).toContain('Map last improved');
  // one on each place in the list, and one on the open place's heading
  const badges = screen.getAllByText('Learning').map(b => b.closest('[data-working]'));
  expect(badges).toHaveLength(3);
  expect(badges.map(b => b?.getAttribute('data-working'))).toEqual(['false', 'false', 'false']);
  expect(within(await panel()).queryByRole('progressbar')).toBeNull();
  expect((await panel()).querySelector('.mantine-Loader-root')).toBeNull(); // nothing moves while nothing runs
});

it('moves the badge of the place a step is training, with a measured bar and its step count', async () => {
  serve(running({ phase: 'training', place: 'p2', views: 2, step: 1200, steps: 3000 }));
  renderPage(<PlacesTab examples={[]} />);
  expect((await state()).textContent).toBe('Training the splatStep 1,200 of 3,000, with 2 new views.');
  const bar = within(await panel()).getByRole('progressbar', { name: "How far the splat's training is" });
  expect(bar.getAttribute('aria-valuenow')).toBe('40');
  expect((await panel()).textContent).toContain('40%');
  expect((await panel()).querySelector('.mantine-Loader-root')).not.toBeNull();
  const working = (name: RegExp) =>
    within(within(list()).getByRole('button', { name }))
      .getByText('Learning')
      .closest('[data-working]')
      ?.getAttribute('data-working');
  expect(working(/Cabin/)).toBe('true');
  expect(working(/Home/)).toBe('false');
});

it('shows an unmeasured bar while a step does not say how far it is', async () => {
  serve(running({ phase: 'pairing', frames: 2 }));
  renderPage(<PlacesTab examples={[]} />);
  expect((await state()).textContent).toContain('Placing 2 new views');
  expect(within(await panel()).getByRole('progressbar', { name: 'A map step is running' })).toBeTruthy();
  // before training the step hasn't said which place: every place's badge moves
  const badges = screen.getAllByText('Learning').map(b => b.closest('[data-working]')?.getAttribute('data-working'));
  expect(badges).toEqual(['true', 'true', 'true']);
});

it('says how many frames wait, and that the next step starts by itself', async () => {
  serve(status({ waiting: 3 }));
  renderPage(<PlacesTab examples={[]} />);
  expect((await state()).textContent).toBe('3 frames waitingThe next step starts by itself.');
  expect(stat('Waiting')).toBe('3');
});

it('gives each camera a row with its name, how it is doing, and its newest kept frame when there is one', async () => {
  const frame = 'ab'.repeat(16);
  serve({
    ...status({ parked: 1, parked_cameras: { hall: 'facing' }, frames: { front: frame } }),
    cameras: { front: { recent: { person: 9, same: 3 }, last_kept: 1.8e9 } },
  });
  renderPage(<PlacesTab examples={[]} />);
  const rows = within(await within(await panel()).findByRole('list', { name: 'Cameras learned from' })).getAllByRole(
    'listitem',
  );
  expect(rows.map(r => r.textContent)).toEqual(['frontPerson or animal in view', 'hallNeeds a facing']);
  const picture = within(rows[0] as HTMLElement).getByRole('img', { name: 'The newest frame kept from front' });
  expect(picture.getAttribute('src')).toBe(`/surroundings/thumb/${frame}.jpg`);
  // no frame of the hall may be shown: a plain tile, and the reason in full for a pointer
  expect(within(rows[1] as HTMLElement).queryByRole('img')).toBeNull();
  expect(rows[1]?.getAttribute('title')).toBe('Its spot on the plan has no facing. Give it one in the plan editor.');
  // a picture that doesn't load gives way to the plain tile
  fireEvent.error(picture);
  expect(within(rows[0] as HTMLElement).queryByRole('img')).toBeNull();
});
