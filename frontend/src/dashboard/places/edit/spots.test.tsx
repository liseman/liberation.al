import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, expect, it } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import { floorplan, parseDsl, projection } from '../../../vendor/floorplan/index.js';
import { tipOf } from './CameraSpots';
import { PlanEditor } from './PlanEditor';
import { bearing, setSpot, spotLine, spotsOf } from './spots';
import { split } from './text';

// Two made-up rooms, one camera with a facing and one without, and x- lines written unevenly and between upstream
// lines, so a rewrite that touched anything but its own line would show.
const PLAN = `plan "Home" walls 0.2/0.1

room kitchen "Kitchen" kitchen poly 0,0 4,0 4,3 0,3
x-anchor place:home yaw 0 origin 0,0,0
room den "Den" living poly 4,0 7,0 7,3 4,3

door kitchen>den at 4,1.5 w0.9
x-camera cam0 at 1.50,1 height 2.4 yaw 45 fov 100
x-camera cam1 at 5,2 height 2.2
x-object sofa "Sofa" at 6,1 size 2x0.9
`;
const cameraLines = (text: string) => text.split('\n').filter(l => l.startsWith('x-camera'));
const others = (text: string) => text.split('\n').filter(l => !l.startsWith('x-camera'));

it('reads each spot, and setting a spot to what it says leaves the text byte for byte', () => {
  expect(spotsOf(PLAN)).toEqual({
    cam0: { at: [1.5, 1], height: 2.4, yaw: 45, fov: 100 },
    cam1: { at: [5, 2], height: 2.2 },
  });
  for (const [id, s] of Object.entries(spotsOf(PLAN))) expect(setSpot(PLAN, id, s)).toBe(PLAN);
  // A nudge that rounds away (under half a centimetre) is no change either.
  expect(setSpot(PLAN, 'cam1', { at: [5.001, 2], height: 2.2 })).toBe(PLAN);
});

it('rewrites only the camera changed, where its line stands, and adds a new one at the end', () => {
  const turned = setSpot(PLAN, 'cam1', { at: [5, 2], height: 2.2, yaw: 270 });
  expect(others(turned)).toEqual(others(PLAN));
  expect(cameraLines(turned)).toEqual([
    'x-camera cam0 at 1.50,1 height 2.4 yaw 45 fov 100',
    'x-camera cam1 at 5,2 height 2.2 yaw 270',
  ]);
  const added = setSpot(PLAN, 'cam2', { at: [2, 2.5] });
  expect(added).toBe(`${PLAN}x-camera cam2 at 2,2.5\n`);
  expect(spotLine('cam3', { at: [1.234, -0.006], yaw: -90.04, fov: 359.99 })).toBe(
    'x-camera cam3 at 1.23,-0.01 yaw 270 fov 360',
  );
  // The editor's split still sets every x- line aside, the new ones included.
  expect(split(added).ext.filter(l => l.startsWith('x-camera'))).toHaveLength(3);
});

it('turns toward a point: up the page is 0, clockwise from there', () => {
  expect(bearing([0, 0], [0, -1])).toBe(0);
  expect(bearing([0, 0], [1, 0])).toBe(90);
  expect(bearing([0, 0], [0, 1])).toBe(180);
  expect(bearing([0, 0], [-1, 0])).toBe(270);
});

// jsdom has no layout and no PointerEvent: screen coordinates are the overlay's own SVG units, as in edit.test.tsx.
class Point {
  x: number;
  y: number;
  constructor(x: number, y: number) {
    this.x = x;
    this.y = y;
  }
  matrixTransform() {
    return this;
  }
}
let restore: (() => void)[] = [];
beforeEach(() => {
  restore = [
    stub(SVGSVGElement.prototype, 'getScreenCTM', () => ({ inverse: () => ({}) })),
    stub(globalThis, 'DOMPoint', Point),
    stub(globalThis, 'PointerEvent', class extends MouseEvent {}),
    stub(globalThis, 'NAMES', { cams: ['cam0', 'cam1', 'cam2'], views: { cam0: 'front camera' } }),
  ];
});
afterEach(() => {
  for (const r of restore) r();
});

const proj = projection(floorplan(parseDsl(split(PLAN).base).doc).model, {});
const screenOf = (p: [number, number]) => {
  const [clientX, clientY] = proj.toScreen(p);
  return { clientX, clientY };
};

function open(onSave?: (plan: string) => void) {
  server.use(
    http.post('/places/plan', async ({ request }) => {
      onSave?.(((await request.json()) as { plan: string }).plan);
      return HttpResponse.json({ ok: true, plan: '', plan_source: 'owner' });
    }),
  );
  return renderPage(<PlanEditor place="p1" text={PLAN} frame="r1" options={{}} onDone={() => {}} />);
}

async function saved(user: ReturnType<typeof userEvent.setup>, plans: string[]) {
  await user.click(screen.getByRole('button', { name: 'Save my plan' }));
  await waitFor(() => expect(plans).toHaveLength(1));
  return plans[0] as string;
}

it('shows each spot with a cone for its facing, and selecting one opens its form', async () => {
  const user = userEvent.setup();
  const { container } = open();
  expect(container.querySelectorAll('[data-camera] path')).toHaveLength(1); // cam1 has no facing: no cone
  expect(screen.getByRole('button', { name: 'Save my plan' })).toBeDisabled();
  await user.click(screen.getByRole('button', { name: 'cam1 (no facing)' }));
  expect(screen.getByRole('textbox', { name: 'Camera' })).toHaveValue('cam1');
  expect(screen.getByRole('textbox', { name: 'Camera' })).toHaveAttribute('readonly');
  expect(screen.getByText(/places nobody until it has a facing/)).toBeTruthy();
  await user.click(screen.getByRole('button', { name: 'Camera front camera' }));
  expect(screen.getByRole('textbox', { name: 'Camera' })).toHaveValue('front camera');
  expect(screen.getByRole('textbox', { name: 'Facing, °' })).toHaveValue('45');
});

it('saves a facing typed in the form as that camera line, with the rest of the plan untouched', async () => {
  const user = userEvent.setup();
  const plans: string[] = [];
  open(p => plans.push(p));
  await user.click(screen.getByRole('button', { name: 'cam1 (no facing)' }));
  await user.type(screen.getByRole('textbox', { name: 'Facing, °' }), '200{Enter}');
  const plan = await saved(user, plans);
  expect(plan).toBe(PLAN.replace('x-camera cam1 at 5,2 height 2.2', 'x-camera cam1 at 5,2 height 2.2 yaw 200'));
});

it('moves a spot dragged with the pointer, and the cone follows; one drag is one undo', async () => {
  const user = userEvent.setup();
  const plans: string[] = [];
  const { container } = open(p => plans.push(p));
  const dot = screen.getByRole('button', { name: 'Camera front camera' });
  const cone = () => container.querySelector('[data-camera="cam0"] path')?.getAttribute('d');
  const before = cone();
  const from = screen0(dot);
  const to = screenOf([2.5, 1.5]); // a metre east and half a metre south of where it was
  await user.pointer([
    { keys: '[MouseLeft>]', target: dot, coords: from },
    { coords: { clientX: (from.clientX + to.clientX) / 2, clientY: (from.clientY + to.clientY) / 2 } },
    { coords: to },
  ]);
  expect(cone()).not.toBe(before); // live, before the release
  await user.pointer({ keys: '[/MouseLeft]', coords: to });
  expect(spotsOf(await saved(user, plans)).cam0?.at).toEqual([2.5, 1.5]);
  expect(cameraLines(plans[0] as string)[0]).toBe('x-camera cam0 at 2.5,1.5 height 2.4 yaw 45 fov 100');
});

it('turns a selected spot with its handle', async () => {
  const user = userEvent.setup();
  const plans: string[] = [];
  open(p => plans.push(p));
  await user.click(screen.getByRole('button', { name: 'Camera front camera' }));
  const handle = screen.getByRole('slider', { name: 'Facing of front camera' });
  const at: [number, number] = [1.5, 1];
  await user.pointer([
    { keys: '[MouseLeft>]', target: handle, coords: screenOf(tipOf({ at, yaw: 45 })) },
    { coords: screenOf([at[0], at[1] + 1]) },
    { coords: screenOf([at[0], at[1] + 2]) }, // straight down the page
    { keys: '[/MouseLeft]' },
  ]);
  expect(spotsOf(await saved(user, plans)).cam0?.yaw).toBe(180);
});

it('moves and turns with the keyboard', async () => {
  const user = userEvent.setup();
  const plans: string[] = [];
  open(p => plans.push(p));
  await user.click(screen.getByRole('button', { name: 'cam1 (no facing)' }));
  screen.getByRole('button', { name: 'Camera cam1' }).focus();
  await user.keyboard('{ArrowRight}{Shift>}{ArrowUp}{/Shift}');
  screen.getByRole('slider', { name: 'Facing of cam1' }).focus();
  await user.keyboard('{ArrowRight}{ArrowRight}');
  expect(spotsOf(await saved(user, plans)).cam1).toEqual({ at: [5.05, 1.5], height: 2.2, yaw: 10 });
});

it('adds a manifest camera with no spot where the plan is clicked, or in its middle', async () => {
  const user = userEvent.setup();
  const plans: string[] = [];
  open(p => plans.push(p));
  const pick = async (name: string) => {
    await user.click(screen.getByRole('combobox', { name: 'Add a camera' }));
    await user.click(within(screen.getByRole('listbox')).getByRole('option', { name }));
  };
  await pick('cam2');
  expect(screen.getByText('Click the plan where cam2 is.')).toBeTruthy();
  const catcher = screen.getByLabelText('Put cam2 here');
  await user.pointer({ keys: '[MouseLeft]', target: catcher, coords: screenOf([2, 2.5]) });
  expect(screen.getByRole('textbox', { name: 'Camera' })).toHaveValue('cam2'); // selected, ready for a facing
  expect(screen.queryByRole('combobox', { name: 'Add a camera' })).toBeNull(); // every camera has a spot now
  const plan = await saved(user, plans);
  expect(plan).toBe(`${PLAN}x-camera cam2 at 2,2.5\n`);
});

it('puts an added camera in the middle of the plan from the keyboard', async () => {
  const user = userEvent.setup();
  open();
  await user.click(screen.getByRole('combobox', { name: 'Add a camera' }));
  await user.click(within(screen.getByRole('listbox')).getByRole('option', { name: 'cam2' }));
  await user.click(screen.getByRole('button', { name: 'Put it in the middle' }));
  expect(screen.getByRole('button', { name: 'Camera cam2' })).toBeTruthy();
  expect(screen.getByRole('textbox', { name: 'X, m' })).toHaveValue('3.5'); // the drawing's middle
});

/** Where a drawn spot is on screen. */
function screen0(el: Element) {
  return { clientX: Number(el.getAttribute('cx')), clientY: Number(el.getAttribute('cy')) };
}

function stub(obj: object, key: string, value: unknown): () => void {
  const had = Object.getOwnPropertyDescriptor(obj, key);
  Object.defineProperty(obj, key, { value, configurable: true, writable: true });
  return () => (had ? Object.defineProperty(obj, key, had) : delete (obj as Record<string, unknown>)[key]);
}
