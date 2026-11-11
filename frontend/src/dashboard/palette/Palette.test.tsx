import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { isoDay } from '../lib/format';
import { filterGroups, Palette, paletteActions, SHOWN_UNTYPED } from './Palette';
import { PaletteButton } from './PaletteButton';
import { palette } from './store';

beforeEach(() => {
  palette.close(); // the store is the module's: a test that stops early leaves it open
  server.use(
    http.get('/people', () => HttpResponse.json({ people: [{ name: 'Dana', voices: [] }] })),
    http.get('/pets', () => HttpResponse.json({ pets: [{ id: 'p1', name: 'Mia', species: 'cat' }], species: [] })),
  );
});

function open() {
  renderPage(
    <>
      <PaletteButton />
      <Palette />
    </>,
  );
}

it('jumps to a person by name from the top bar button', async () => {
  open();
  await userEvent.click(screen.getByRole('button', { name: /^Go to/ }));
  await userEvent.type(await screen.findByRole('combobox', { name: 'Go to' }), 'dan');
  await userEvent.click(await screen.findByRole('option', { name: /Dana/ }));
  expect(location.pathname).toBe('/people');
  expect(location.hash).toBe('#card=Dana');
});

it('opens with Ctrl+K and goes to a day in Review', async () => {
  open();
  await userEvent.keyboard('{Control>}k{/Control}');
  await userEvent.type(await screen.findByRole('combobox', { name: 'Go to' }), 'yesterday');
  await userEvent.click(await screen.findByRole('option', { name: /^Review / }));
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  expect(location.pathname).toBe('/review');
  expect(location.hash).toBe(`#day=${isoDay(yesterday)}`);
});

it('lists pages, Settings sections, people, and pets, and a day only when one is typed', () => {
  const groups = (q: string) => paletteActions(q, ['Dana'], ['Mia'], new Date(2026, 10, 6, 19));
  const ids = (q: string) => groups(q).flatMap(g => g.actions.map(a => a.id));
  expect(groups('').map(g => g.group)).toEqual(['Next steps', 'Pages', 'People', 'Pets', 'Settings']);
  expect(ids('')).toEqual(
    expect.arrayContaining(['page:live', 'page:people', 'settings:services', 'person:Dana', 'pet:Mia']),
  );
  expect(ids('')).toContain('page:known');
  const pages = groups('').find(g => g.group === 'Pages')?.actions ?? [];
  expect(pages.slice(0, 7).map(a => a.label)).toEqual([
    'Live',
    'Review',
    'Known',
    'Calls',
    'Inbox',
    'Journal',
    'Statistics',
  ]);
  expect(pages.some(a => a.label === 'Reports')).toBe(false);
  const ingest = pages.find(a => a.id === 'page:ingest');
  expect(ingest?.description).toBe('Known');
  ingest?.go();
  expect(location.pathname).toBe('/ingest');
  expect(ids('nov 3')).toContain('day:2026-11-03');
});

it("doesn't read people or pets until it's opened", async () => {
  let asked = 0;
  server.use(
    http.get('/people', () => {
      asked += 1;
      return HttpResponse.json({ people: [] });
    }),
  );
  open();
  await new Promise(r => setTimeout(r, 50));
  expect(asked).toBe(0);
  await userEvent.click(screen.getByRole('button', { name: /^Go to/ }));
  await waitFor(() => expect(asked).toBe(1));
});

it('goes to today in Review, named in the address, so Review leaves the day it showed (the review)', async () => {
  open();
  await userEvent.click(screen.getByRole('button', { name: /^Go to/ }));
  await userEvent.type(await screen.findByRole('combobox', { name: 'Go to' }), 'review today');
  await userEvent.click(await screen.findByRole('option', { name: /^Review today/ }));
  expect(location.pathname).toBe('/review');
  expect(location.hash).toBe(`#day=${isoDay(new Date())}`);
});

it('shows a few of each group until something is typed, then every match', () => {
  const people = Array.from({ length: 20 }, (_, n) => `Person ${n}`);
  const groups = paletteActions('', people, [], new Date(2026, 10, 6, 19));
  const shown = (q: string) => filterGroups(groups, q).find(g => g.group === 'People')?.actions.length;
  expect(shown('')).toBe(SHOWN_UNTYPED);
  expect(filterGroups(groups, '').map(g => g.group)).toContain('Settings'); // not pushed off by people
  expect(shown('person')).toBe(20);
  expect(shown('person 1')).toBe(11); // 1 and 10 to 19
});

it('tells a screen reader how many match and which is selected', async () => {
  open();
  await userEvent.click(screen.getByRole('button', { name: /^Go to/ }));
  expect(await screen.findByRole('dialog', { name: 'Go to' })).toBeInTheDocument();
  const box = await screen.findByRole('combobox', { name: 'Go to' });
  await userEvent.type(box, 'dana');
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/^1 match/));
  await userEvent.keyboard('{ArrowDown}');
  const option = await screen.findByRole('option', { name: /Dana/ });
  await waitFor(() => expect(box).toHaveAttribute('aria-activedescendant', option.id));
});

it('opens with Ctrl+K from a text field, but not over another dialog', async () => {
  open();
  const field = document.createElement('input');
  document.body.append(field);
  field.focus();
  await userEvent.keyboard('{Control>}k{/Control}');
  expect(await screen.findByRole('combobox', { name: 'Go to' })).toBeInTheDocument();
  await userEvent.keyboard('{Escape}');
  await waitFor(() => expect(screen.queryByRole('combobox', { name: 'Go to' })).not.toBeInTheDocument());
  const confirm = document.createElement('div');
  confirm.setAttribute('aria-modal', 'true');
  document.body.append(confirm);
  await userEvent.keyboard('{Control>}k{/Control}');
  await new Promise(r => setTimeout(r, 100));
  expect(screen.queryByRole('combobox', { name: 'Go to' })).not.toBeInTheDocument();
  confirm.remove();
  field.remove();
});
