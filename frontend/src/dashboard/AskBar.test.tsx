import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect, useState } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../test/render';
import { AskBar } from './AskBar';
import { openAsk, openAskHere, useAsk, useAskPage } from './ask/openAsk';
import { paletteActions } from './palette/Palette';
import type { TabId } from './router';

const chunk = vi.hoisted(() => {
  let finish!: () => void;
  const ready = new Promise<void>(resolve => {
    finish = resolve;
  });
  return { ready, finish };
});
/** How many times the Ask box has mounted, and the requests it was handed. */
const box = vi.hoisted(() => ({ mounts: 0 }));
vi.mock('./ask/PackAsk', async () => {
  await chunk.ready;
  return {
    PackAsk: () => {
      useEffect(() => {
        box.mounts += 1;
      }, []);
      return <textarea aria-label="Question" />;
    },
  };
});

beforeEach(() => {
  useAsk.setState({ open: false, request: null });
  useAskPage.setState({ day: null, seek: null, mark: null });
});

it('keeps the first-load spinner inside the same floating shell while the Ask body downloads', async () => {
  renderPage(
    <>
      <header data-testid="app-header">
        <AskBar tab="live" setup={false} />
      </header>
      <main>The page stays available.</main>
    </>,
  );
  await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
  const dialog = screen.getByRole('dialog', { name: 'Ask' });
  const spinner = await screen.findByLabelText('Loading Ask');
  expect(dialog).toContainElement(spinner);
  expect(screen.getByTestId('app-header')).not.toContainElement(spinner);
  expect(screen.getByText('The page stays available.')).toBeVisible();
  await act(async () => chunk.finish());
  expect(await screen.findByLabelText('Question')).toBeVisible();
  expect(screen.getByRole('dialog', { name: 'Ask' })).toBe(dialog);
  expect(screen.queryByLabelText('Loading Ask')).not.toBeInTheDocument();
});

function Harness() {
  const [settings, setSettings] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setSettings(value => !value)}>
        Toggle settings
      </button>
      <AskBar tab="live" setup={false} settings={settings} />
    </>
  );
}

it('keeps the Ask trigger mounted but inaccessible while Settings covers the page', async () => {
  renderPage(<Harness />);
  const trigger = screen.getByRole('button', { name: 'Ask' });
  await userEvent.click(screen.getByRole('button', { name: 'Toggle settings' }));
  expect(trigger).toBeInTheDocument();
  expect(trigger).not.toBeVisible();
  expect(screen.queryByRole('button', { name: 'Ask' })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Toggle settings' }));
  expect(screen.getByRole('button', { name: 'Ask' })).toBe(trigger);
});

it('opens the Ask panel when another page calls openAsk', async () => {
  renderPage(<AskBar tab="inbox" setup={false} />);
  expect(screen.queryByRole('dialog', { name: 'Ask' })).not.toBeInTheDocument();
  act(() => openAsk({ hint: { kind: 'report', id: 'r1', question: 'Who came by?' } }));
  expect(await screen.findByRole('dialog', { name: 'Ask' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Ask' })).toHaveAttribute('aria-expanded', 'true');
});

/** The shell's top bar, on a tab a test can change. */
const shown: { set?: (tab: TabId) => void } = {};
function Shell() {
  const [tab, setTab] = useState<TabId>('live');
  shown.set = setTab;
  return <AskBar tab={tab} setup={false} />;
}

it('is the same one Ask from every way in: the top bar, "/", the palette, Review, and the Inbox', async () => {
  vi.stubGlobal('NAMES', { assistant: 'Bob', owner: 'Alice', cams: ['cam0'], hotkeys: true });
  const before = box.mounts;
  renderPage(<Shell />);
  useAskPage.setState({ day: { day: '2026-10-20', label: 'Tue 20 Oct', today: false } });
  const ways: [string, () => void][] = [
    ['the top bar', () => screen.getByRole('button', { name: 'Ask' }).click()],
    ['the / key', () => document.body.dispatchEvent(new KeyboardEvent('keydown', { key: '/', bubbles: true }))],
    [
      'the palette',
      () =>
        paletteActions('', [], [])[0]
          ?.actions.find(a => a.id === 'do:ask')
          ?.go(),
    ],
    ['a report follow-up', () => openAsk({ hint: { kind: 'report', id: 'r1', question: 'Who?' } })],
  ];
  await act(async () => chunk.finish());
  for (const [, open] of ways) {
    act(() => useAsk.setState({ open: false }));
    act(open);
    expect(await screen.findAllByRole('dialog', { name: 'Ask' })).toHaveLength(1);
    expect(await screen.findByLabelText('Question')).toBeVisible();
  }
  // Review's top bar opens the same Ask, with its day as a hint
  act(() => shown.set?.('review'));
  act(() => useAsk.setState({ open: false, request: null }));
  act(() => screen.getByRole('button', { name: 'Ask' }).click());
  expect(useAsk.getState().request?.hint).toEqual({
    kind: 'day',
    day: '2026-10-20',
    label: 'Tue 20 Oct',
    today: false,
  });
  expect(screen.getAllByRole('dialog', { name: 'Ask' })).toHaveLength(1);
  // One box the whole time: opening it another way never mounts a second or another kind
  expect(box.mounts - before).toBe(1);
  vi.unstubAllGlobals();
});

it('opens with no hint off Review, so a day from before is not carried along', () => {
  useAskPage.setState({ day: { day: '2026-10-20', label: 'Tue 20 Oct', today: false } });
  openAskHere('inbox');
  expect(useAsk.getState().request?.hint).toBeNull();
});
