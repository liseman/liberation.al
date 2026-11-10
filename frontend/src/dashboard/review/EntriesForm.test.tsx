import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { hms } from '../lib/format';
import { EngineContext } from './context';
import { EntriesForm } from './EntriesForm';
import type { ReviewEngine } from './engine';

const D0 = new Date(2026, 9, 27, 18).getTime() / 1000;
const SPAN = { cam: 'cam1', start: D0, end: D0 + 30 };

let marked: Record<string, unknown>[];
let overlapping: boolean;
beforeEach(() => {
  marked = [];
  overlapping = false;
  server.use(
    http.post('/measurement/mark', async ({ request }) => {
      const body = (await request.json()) as Record<string, unknown>;
      marked.push(body);
      if (overlapping && !body.replace)
        return HttpResponse.json({ ok: false, error: 'this window overlaps one already labeled' }, { status: 409 });
      return HttpResponse.json({ ok: true, replaced: body.replace ? 1 : 0 });
    }),
  );
});

function open() {
  const engine = { cams: ['cam0', 'cam1'], t: D0 + 4 };
  renderPage(
    <EngineContext value={engine as unknown as ReviewEngine}>
      <EntriesForm span={SPAN} onClose={() => {}} />
    </EngineContext>,
  );
  return { engine, form: screen.getByRole('form', { name: 'Entries and exits' }) };
}

it('adds entries and exits at the playhead and an unsure stretch, and saves the window', async () => {
  const { engine, form } = open();
  await userEvent.click(within(form).getByRole('button', { name: 'Entry at playhead' }));
  engine.t = D0 + 28;
  await userEvent.click(within(form).getByRole('button', { name: 'Exit at playhead' }));
  engine.t = D0 + 10;
  await userEvent.click(within(form).getByRole('button', { name: 'Add an unsure stretch' }));
  await userEvent.clear(within(form).getByLabelText('Unsure to'));
  await userEvent.type(within(form).getByLabelText('Unsure to'), hms(D0 + 15));
  await userEvent.type(within(form).getByLabelText('Why'), 'door hides the edge');
  await userEvent.click(within(form).getByLabelText('Something in the way'));
  await userEvent.click(within(form).getByRole('button', { name: 'Save' }));
  expect(await within(form).findByRole('status')).toHaveTextContent('Saved: 2 entries and exits.');
  expect(marked).toEqual([
    {
      kind: 'entries',
      camera: 'cam1',
      start: D0,
      end: D0 + 30,
      events: [
        { kind: 'entry', at: D0 + 4 },
        { kind: 'exit', at: D0 + 28 },
      ],
      unsure: [{ start: D0 + 10, end: D0 + 15, reason: 'door hides the edge' }],
      conditions: ['occlusion'],
      none_seen: false,
    },
  ]);
});

it('saves an empty window only once the owner says nobody came or went', async () => {
  const { form } = open();
  await userEvent.click(within(form).getByRole('button', { name: 'Save' }));
  expect(within(form).getByRole('alert')).toHaveTextContent('tick that nobody came or went');
  expect(marked).toEqual([]);
  await userEvent.click(within(form).getByLabelText(/nobody came or went/));
  await userEvent.click(within(form).getByRole('button', { name: 'Save' }));
  expect(await within(form).findByRole('status')).toHaveTextContent('Saved: nobody came or went.');
  expect(marked[0]).toMatchObject({ events: [], unsure: [], none_seen: true });
});

it('offers to replace a window it overlaps, and replaces only when asked', async () => {
  overlapping = true;
  const { form } = open();
  await userEvent.click(within(form).getByRole('button', { name: 'Entry at playhead' }));
  await userEvent.click(within(form).getByRole('button', { name: 'Save' }));
  expect(await within(form).findByRole('alert')).toHaveTextContent('overlaps one already labeled');
  await userEvent.click(within(form).getByRole('button', { name: 'Replace the overlapping window' }));
  expect(await within(form).findByRole('status')).toHaveTextContent('in place of 1 overlapping window');
  expect(marked.map(m => m.replace)).toEqual([undefined, true]);
});
