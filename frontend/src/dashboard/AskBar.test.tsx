import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { expect, it, vi } from 'vitest';
import { renderPage } from '../test/render';
import { AskBar } from './AskBar';
import { openAsk } from './ask/openAsk';

const chunk = vi.hoisted(() => {
  let finish!: () => void;
  const ready = new Promise<void>(resolve => {
    finish = resolve;
  });
  return { ready, finish };
});
vi.mock('./ask/PackAsk', async () => {
  await chunk.ready;
  return { PackAsk: () => <textarea aria-label="Question" /> };
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
  act(() => openAsk({ context: { kind: 'report', id: 'r1', question: 'Who came by?' } }));
  expect(await screen.findByRole('dialog', { name: 'Ask' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Ask' })).toHaveAttribute('aria-expanded', 'true');
});
