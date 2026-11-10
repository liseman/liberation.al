// What the Places page says "Learn surroundings" is doing (#283): each state as text, the bar's kind, and the badge.
import { screen, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { hm } from '../lib/format';
import {
  activityText,
  cameraReasons,
  parkedReasons,
  LEARNING_OFF_MS,
  LEARNING_ON_MS,
  LEARNING_RUNNING_MS,
  type LearningStatus,
  type LearningStep,
  learningEvery,
  measured,
  samplingText,
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

it.each([
  ['pairing', running({ phase: 'pairing', frames: 2 }), 'Placing 2 new views: matching them against the map.'],
  ['depth', running({ phase: 'depth', frames: 1 }), 'Placing 1 new view: measuring their depth.'],
  [
    'training, before the trainer says a step',
    running({ phase: 'training', views: 2, steps: 3000 }),
    'Training the splat with 2 new views.',
  ],
  [
    'training',
    running({ phase: 'training', views: 2, step: 1200, steps: 3000 }),
    'Training the splat with 2 new views (step 1,200 of 3,000).',
  ],
  ['writing', running({ phase: 'writing', views: 2 }), 'Writing the new map and splat.'],
  ['a step that has not said what it is doing', status({ running: true }), 'Working on the map.'],
  ['frames waiting', status({ waiting: 3 }), '3 kept frames waiting to be placed. The next step starts by itself.'],
  ['up to date', status({ last_step: 1.8e9 }), `Up to date since ${hm(1.8e9)}.`],
  ['up to date, no step yet', status(), 'Up to date.'],
  ['nothing kept yet', status({ kept: 0 }), 'Watching for views worth keeping. Nothing kept yet.'],
])('says what learning is doing: %s', (_, told, text) => {
  expect(activityText(told)).toBe(text);
});

it('measures the bar only while the training says how far it is', () => {
  expect(measured(running({ phase: 'training', step: 1200, steps: 3000 }))).toBe(40);
  expect(measured(running({ phase: 'training', steps: 3000 }))).toBeNull();
  expect(measured(running({ phase: 'pairing', frames: 2 }))).toBeNull();
  // a count left from a step that ended measures nothing
  expect(measured(status({ step: step({ phase: 'training', step: 1200, steps: 3000 }) }))).toBeNull();
});

it('says how many cameras are sampled and how many frames are kept and waiting', () => {
  expect(samplingText(status({ waiting: 3 }))).toBe('Sampling 2 cameras · 37 frames kept, 3 waiting.');
  expect(samplingText(status({ sampling: ['front'], kept: 1 }))).toBe('Sampling 1 camera · 1 frame kept.');
  expect(samplingText(status({ sampling: [], kept: 0 }))).toBe('No camera is chosen to learn from · 0 frames kept.');
});

it("says why a camera's frames aren't being kept, by what most of its recent looks came to", () => {
  const cameras = {
    front: { recent: { same: 5, copy: 4, person: 3 }, last_kept: 1.8e9 }, // 9 of 12 found nothing new
    hall: { recent: { person: 7, same: 5 }, last_kept: null },
    porch: { recent: { dark: 12 }, last_kept: null },
    yard: { recent: { blurred: 7, same: 5 }, last_kept: null },
    desk: { recent: { same: 11, view: 1 }, last_kept: 1.8e9 }, // kept one lately: nothing to explain
    attic: { recent: {}, last_kept: null }, // not looked at yet
    shed: { recent: { same: 12 }, last_kept: null }, // not sampled any more
  };
  const sampling = ['front', 'hall', 'porch', 'yard', 'desk', 'attic', 'unseen'];
  expect(cameraReasons(status({ sampling, cameras }))).toEqual([
    { camera: 'front', why: 'nothing new in view' },
    { camera: 'hall', why: 'a person or an animal is in view' },
    { camera: 'porch', why: "it's too dark" },
    { camera: 'yard', why: 'the picture is blurred' },
  ]);
});

it('counts the parked frames, and says for each camera why its frame is not in the map', () => {
  const parked = status({ parked: 2, parked_cameras: { hall: 'overlap', front: 'facing' } });
  expect(samplingText(parked)).toBe('Sampling 2 cameras · 37 frames kept, 2 parked.');
  expect(parkedReasons(parked)).toEqual([
    { camera: 'front', why: 'its spot on the plan has no facing. Give it one in the plan editor' },
    {
      camera: 'hall',
      why: 'its view overlaps nothing in the map yet, and it has no spot on the plan to place it by',
    },
  ]);
  expect(parkedReasons(status())).toEqual([]); // an older service says nothing of parking
  expect(parkedReasons(status({ parked_cameras: { front: 'something new' } }))).toEqual([
    { camera: 'front', why: "it can't be placed yet" },
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

function serve(told: LearningStatus | null) {
  server.use(
    http.get('/places', () => HttpResponse.json({ places: [place('p1', 'Home'), place('p2', 'Cabin')] })),
    http.get('/places/plan/history', () => HttpResponse.json({ place: 'p1', versions: [], previous: null })),
    http.get('/places/positions', () => HttpResponse.json({ error: 'none' }, { status: 404 })),
    ...(told ? [http.get('/surroundings/status', () => HttpResponse.json({ ...told, places: {} }))] : []),
  );
}
const list = () => screen.getByRole('navigation', { name: 'Places' });
const panel = () => screen.findByRole('status', { name: 'Learning surroundings' });

beforeEach(() => {
  history.replaceState(null, '', '/places');
});

it('shows no badge and no panel while learning is off', async () => {
  serve(null);
  renderPage(<PlacesTab examples={[]} />);
  await screen.findByRole('button', { name: /Home/ });
  expect(screen.queryByText('Learning')).toBeNull();
  expect(screen.queryByRole('status', { name: 'Learning surroundings' })).toBeNull();
});

it('marks each place Learning while it is on, still while nothing runs, with since when it is up to date', async () => {
  serve(status({ last_step: 1.8e9 }));
  renderPage(<PlacesTab examples={[]} />);
  expect((await panel()).textContent).toContain(`Up to date since ${hm(1.8e9)}.`);
  expect((await panel()).textContent).toContain('Sampling 2 cameras · 37 frames kept.');
  // one on each place in the list, and one on the open place's heading
  const badges = screen.getAllByText('Learning').map(b => b.closest('[data-working]'));
  expect(badges).toHaveLength(3);
  expect(badges.map(b => b?.getAttribute('data-working'))).toEqual(['false', 'false', 'false']);
  expect(within(await panel()).queryByRole('progressbar')).toBeNull();
});

it('moves the badge of the place a step is training, with a measured bar and its step count', async () => {
  serve(running({ phase: 'training', place: 'p2', views: 2, step: 1200, steps: 3000 }));
  renderPage(<PlacesTab examples={[]} />);
  expect((await panel()).textContent).toContain('Training the splat with 2 new views (step 1,200 of 3,000).');
  const bar = within(await panel()).getByRole('progressbar', { name: "How far the splat's training is" });
  expect(bar.getAttribute('aria-valuenow')).toBe('40');
  const working = (name: RegExp) =>
    within(within(list()).getByRole('button', { name }))
      .getByText('Learning')
      .closest('[data-working]')
      ?.getAttribute('data-working');
  expect(working(/Cabin/)).toBe('true');
  expect(working(/Home/)).toBe('false');
});

it('shows an unmeasured bar while a step does not say how far it is, and why a camera keeps nothing', async () => {
  serve({
    ...running({ phase: 'pairing', frames: 2 }),
    cameras: { hall: { recent: { person: 9, same: 3 }, last_kept: null } },
  });
  renderPage(<PlacesTab examples={[]} />);
  expect((await panel()).textContent).toContain('Placing 2 new views: matching them against the map.');
  expect(within(await panel()).getByRole('progressbar', { name: 'A map step is running' })).toBeTruthy();
  expect((await panel()).textContent).toContain('hall: no frames kept lately, a person or an animal is in view.');
  // before training the step hasn't said which place: every place's badge moves
  const badges = screen.getAllByText('Learning').map(b => b.closest('[data-working]')?.getAttribute('data-working'));
  expect(badges).toEqual(['true', 'true', 'true']);
});
