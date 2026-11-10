import { fireEvent, screen, within } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import {
  floorplan,
  type OffsetHandle,
  parseDsl,
  projection,
  type VertexHandle,
  wallHandles,
} from '../../../vendor/floorplan/index.js';
import type { Place } from '../examples';
import { OwnerBar } from './OwnerBar';
import { dragged, findingsOf, MAX_STEPS, PlanEditor } from './PlanEditor';
import { addDoor, doorSpot, join, removeOpening, renameRoom, split } from './text';

// Two rooms side by side, the wall between them at x = 4, a door through it and one from outside.
const PLAN = `plan "Home" walls 0.2/0.1

room kitchen "Kitchen" kitchen poly 0,0 4,0 4,3 0,3
room den "Den" living poly 4,0 7,0 7,3 4,3

door kitchen>den at 4,1.5 w0.9
door exterior>kitchen at 0,1.5 w1
`;
const rooms = (text: string) =>
  Object.fromEntries(
    Object.entries(parseDsl(split(text).base).doc.rooms as Record<string, { name: string }>).map(([k, v]) => [
      k,
      v.name,
    ]),
  );
const doors = (text: string) => (parseDsl(text).doc.openings as { between: string[] }[]).map(o => o.between.join('>'));

it('sets the x- lines aside while editing and puts them back, unchanged', () => {
  const text = `${PLAN}\nx-anchor place:home yaw 0 origin 0,0,0\n`;
  const { base, ext } = split(text);
  expect(base).not.toContain('x-anchor');
  expect(ext).toEqual(['x-anchor place:home yaw 0 origin 0,0,0']);
  expect(split(join(renameRoom(base, 'den', 'Study'), ext)).ext).toEqual(ext);
});

it('renames a room in place, adds a door, and removes the one asked for', () => {
  expect(rooms(renameRoom(PLAN, 'den', 'Living "room"'))).toEqual({ kitchen: 'Kitchen', den: 'Living "room"' });
  expect(doors(addDoor(PLAN, 'den', 'exterior', 0.8))).toEqual(['kitchen>den', 'exterior>kitchen', 'den>exterior']);
  expect(doors(removeOpening(PLAN, 0))).toEqual(['exterior>kitchen']);
  expect(doors(removeOpening(PLAN, 1))).toEqual(['kitchen>den']);
});

it('moves the wall between two rooms along its normal by how far it was dragged, snapped to 5 cm', () => {
  const model = floorplan(parseDsl(PLAN).doc).model;
  const handles = [...wallHandles(PLAN, model).values()];
  const shared = handles.find(
    (h): h is OffsetHandle =>
      h.kind === 'offset' &&
      model.walls.find(w => w.id === h.wallId)?.start[0] === 4 &&
      model.walls.find(w => w.id === h.wallId)?.end[0] === 4,
  );
  expect(shared).toBeDefined();
  const moved = dragged(PLAN, shared as OffsetHandle, [4, 1], [4.53, 2]); // 0.53 m east, any amount along the wall
  expect(moved).toContain('room kitchen "Kitchen" kitchen poly 0,0 4.55,0 4.55,3 0,3');
  expect(moved).toContain('room den "Den" living poly 4.55,0 7,0 7,3 4.55,3');
  const corner = handles.find((h): h is VertexHandle => h.kind === 'vertex');
  if (corner) expect(dragged(PLAN, corner, corner.at, [corner.at[0] + 0.2, corner.at[1]])).not.toBe(PLAN);
});

it('shows the findings a hand edit should look at, and none of the design rules', () => {
  expect(findingsOf(PLAN)).toEqual([]); // habitable rooms without windows are a design call, not a plan problem
  const wide = PLAN.replace('door kitchen>den at 4,1.5 w0.9', 'door kitchen>den at 4,1.5 w5');
  expect(findingsOf(wide).map(f => f.rule)).toContain('opening.overflow');
});

it('saves the edited plan as the owner correction, and undoes back to the start', async () => {
  const saved: unknown[] = [];
  server.use(
    http.post('/places/plan', async ({ request }) => {
      saved.push(await request.json());
      return HttpResponse.json({ ok: true, plan: '', plan_source: 'owner' });
    }),
  );
  let done = 0;
  renderPage(<PlanEditor place="p1" text={PLAN} frame="r1" options={{}} onDone={() => (done += 1)} />);
  const save = screen.getByRole('button', { name: 'Save plan' });
  expect(save).toBeDisabled(); // nothing changed yet
  const den = screen.getByLabelText('Name of Den');
  fireEvent.change(den, { target: { value: 'Study' } });
  fireEvent.blur(den);
  fireEvent.click(screen.getByRole('button', { name: /Remove the door between Kitchen and Study/ }));
  expect(screen.queryByRole('button', { name: /Remove the door between Kitchen and Study/ })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  expect(screen.getByRole('button', { name: /Remove the door between Kitchen and Study/ })).toBeTruthy();
  fireEvent.click(save);
  await screen.findByRole('button', { name: 'Save plan' });
  await new Promise(r => setTimeout(r, 0));
  expect(saved).toHaveLength(1);
  const body = saved[0] as { place: string; plan: string; base: string };
  expect(body.base).toBe('r1'); // saved against the frame the text was opened in
  expect(body.place).toBe('p1');
  expect(rooms(body.plan)).toEqual({ kitchen: 'Kitchen', den: 'Study' });
  expect(doors(body.plan)).toEqual(['kitchen>den', 'exterior>kitchen']);
  expect(done).toBe(1);
});

it('adds a door from the two spaces picked', () => {
  renderPage(<PlanEditor place="p1" text={PLAN} frame="r1" options={{}} onDone={() => {}} />);
  const list = () => screen.getAllByRole('button', { name: /^Remove the/ });
  expect(list()).toHaveLength(2);
  const pick = (label: string, option: string) => {
    fireEvent.click(screen.getByRole('combobox', { name: label }));
    fireEvent.click(within(screen.getByRole('listbox')).getByRole('option', { name: option }));
  };
  pick('Door from', 'Den');
  pick('to', 'Outside');
  fireEvent.click(screen.getByRole('button', { name: 'Add door' }));
  expect(list()).toHaveLength(3);
});

it('puts a new door on the middle of the longest wall the two spaces share', () => {
  const model = floorplan(parseDsl(PLAN).doc).model;
  const walls = model.walls as Parameters<typeof doorSpot>[0];
  expect(doorSpot(walls, 'kitchen', 'den')).toEqual([4, 1.5]); // the one wall between them
  // Den's three outside walls are all 3 m: the first, its north one, takes the door.
  expect(doorSpot(walls, 'den', 'exterior')).toEqual([5.5, 0]);
  expect(doorSpot(walls, 'kitchen', 'nowhere')).toBeNull();
  const added = addDoor(PLAN, 'den', 'exterior', 0.8, [5.5, 0]);
  expect(added).toContain('door den>exterior at 5.5,0 w0.8');
  expect(floorplan(parseDsl(added).doc).model.openings).toHaveLength(3); // it resolves: not ambiguous
});

it('keeps a wall dragged with the pointer: a drag is one step, undone in one', () => {
  // jsdom has no layout and no PointerEvent: let screen coordinates be the overlay's own SVG units, and pointer events
  // carry their coordinates, so a drag runs through the real handlers.
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
  const restore = [
    stub(SVGSVGElement.prototype, 'getScreenCTM', () => ({ inverse: () => ({}) })),
    stub(globalThis, 'DOMPoint', Point),
    stub(globalThis, 'PointerEvent', class extends MouseEvent {}),
  ];
  try {
    const { container } = renderPage(<PlanEditor place="p1" text={PLAN} frame="r1" options={{}} onDone={() => {}} />);
    // The wall between the kitchen and the den, at x = 4, where the overlay draws it.
    const [x, y] = projection(floorplan(parseDsl(PLAN).doc).model, {}).toScreen([4, 1]);
    const grip = [...container.querySelectorAll('svg line')].find(
      l => Number(l.getAttribute('x1')) === x && Number(l.getAttribute('x2')) === x,
    ) as Element;
    expect(grip).toBeDefined();
    fireEvent.pointerDown(grip, { clientX: x, clientY: y, pointerId: 1 });
    fireEvent.pointerMove(grip, { clientX: x + 10, clientY: y, pointerId: 1 });
    fireEvent.pointerMove(grip, { clientX: x + 20, clientY: y, pointerId: 1 });
    fireEvent.pointerUp(grip, { clientX: x + 20, clientY: y, pointerId: 1 });
    expect(screen.getByRole('button', { name: 'Save plan' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled(); // the whole drag was one step
    expect(screen.getByRole('button', { name: 'Save plan' })).toBeDisabled();
  } finally {
    for (const r of restore) r();
  }
});

/** The plan editor's text as its rooms and doors show it: each room's name box, and each door's remove button. */
const shown = () => ({
  rooms: screen.getAllByRole('textbox', { name: /^Name of / }).map(i => (i as HTMLInputElement).value),
  doors: screen.getAllByRole('button', { name: /^Remove the/ }).length,
});
const key = (k: string, mods: { shiftKey?: boolean; metaKey?: boolean } = {}, target: Element = document.body) =>
  fireEvent.keyDown(target, { key: k, ctrlKey: !mods.metaKey, ...mods });

it('undoes and redoes each edit with Ctrl+Z, Ctrl+Shift+Z, and Ctrl+Y, and Cmd too', () => {
  renderPage(<PlanEditor place="p1" text={PLAN} frame="r1" options={{}} onDone={() => {}} />);
  const den = screen.getByLabelText('Name of Den');
  fireEvent.change(den, { target: { value: 'Study' } });
  fireEvent.blur(den);
  fireEvent.click(screen.getByRole('button', { name: /Remove the door between Kitchen and Study/ }));
  expect(shown()).toEqual({ rooms: ['Kitchen', 'Study'], doors: 1 });
  key('z');
  expect(shown()).toEqual({ rooms: ['Kitchen', 'Study'], doors: 2 }); // the door back
  key('z', { metaKey: true });
  expect(shown()).toEqual({ rooms: ['Kitchen', 'Den'], doors: 2 }); // the name back
  key('z');
  expect(shown()).toEqual({ rooms: ['Kitchen', 'Den'], doors: 2 }); // nothing before the start
  expect(screen.getByRole('button', { name: 'Save plan' })).toBeDisabled();
  key('Z', { shiftKey: true });
  expect(shown()).toEqual({ rooms: ['Kitchen', 'Study'], doors: 2 });
  key('y');
  expect(shown()).toEqual({ rooms: ['Kitchen', 'Study'], doors: 1 });
  key('y');
  expect(shown()).toEqual({ rooms: ['Kitchen', 'Study'], doors: 1 }); // nothing after the end
  key('z');
  fireEvent.click(screen.getByRole('button', { name: 'Remove the door between Outside and Kitchen' }));
  expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled(); // a new edit drops what was undone
});

it('leaves Ctrl+Z to a text field while typing in it', () => {
  renderPage(<PlanEditor place="p1" text={PLAN} frame="r1" options={{}} onDone={() => {}} />);
  fireEvent.click(screen.getByRole('button', { name: /Remove the door between Kitchen and Den/ }));
  const den = screen.getByLabelText('Name of Den');
  den.focus();
  key('z', {}, den);
  expect(shown().doors).toBe(1); // the editor's history didn't move
  key('z', {}, screen.getByRole('button', { name: 'Undo' }));
  expect(shown().doors).toBe(2); // from a button it does
});

it(`keeps the last ${MAX_STEPS} edits to undo`, () => {
  renderPage(<PlanEditor place="p1" text={PLAN} frame="r1" options={{}} onDone={() => {}} />);
  for (let i = 1; i <= MAX_STEPS + 5; i++) {
    const box = screen.getAllByRole('textbox', { name: /^Name of / })[1] as HTMLInputElement;
    fireEvent.change(box, { target: { value: `Den ${i}` } });
    fireEvent.blur(box);
  }
  const undo = screen.getByRole('button', { name: 'Undo' });
  let steps = 0;
  while (!(undo as HTMLButtonElement).disabled) {
    fireEvent.click(undo);
    steps += 1;
  }
  expect(steps).toBe(MAX_STEPS);
  expect(shown().rooms).toEqual(['Kitchen', 'Den 5']); // the oldest five edits dropped off
});

const OWNED: Place = {
  id: 'p1',
  name: 'Place 1',
  lastVisit: null,
  plan: null,
  planText: PLAN,
  planRevision: 'r1',
  machinePlanText: null,
  origin: { source: 'owner', at: 1762500000, machineNewer: false },
  example: false,
};
it('shows nothing over a plan the machine has no version of its own beside', () => {
  const { container } = renderPage(<OwnerBar place={OWNED} showing="mine" onShow={() => {}} />);
  expect(container.querySelector('button')).toBeNull();
});

it("takes the machine's plan only from the comparison, and only for an owner's plan", () => {
  const compared = { ...OWNED, machinePlanText: PLAN };
  const first = renderPage(<OwnerBar place={compared} showing="mine" onShow={() => {}} />);
  expect(screen.getByRole('radio', { name: "The machine's" })).toBeTruthy();
  expect(screen.queryByRole('button', { name: "Use the machine's plan" })).toBeNull();
  first.unmount();
  const second = renderPage(<OwnerBar place={compared} showing="machine" onShow={() => {}} />);
  expect(screen.getByRole('button', { name: "Use the machine's plan" })).toBeTruthy();
  second.unmount();
  const machines = { ...compared, origin: { source: 'machine' as const, at: 1762500000, machineNewer: false } };
  const { container } = renderPage(<OwnerBar place={machines} showing="machine" onShow={() => {}} />);
  expect(container.querySelector('button')).toBeNull();
});

/** Replace `obj[key]` for one test; returns the undo. */
function stub(obj: object, key: string, value: unknown): () => void {
  const had = Object.getOwnPropertyDescriptor(obj, key);
  Object.defineProperty(obj, key, { value, configurable: true, writable: true });
  return () => (had ? Object.defineProperty(obj, key, had) : delete (obj as Record<string, unknown>)[key]);
}
