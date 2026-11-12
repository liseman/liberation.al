import { act, fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { failed, SAVE_IDLE, SAVE_MARK_MS, SaveState, type SaveStatus, saved, saving } from './SaveState';

afterEach(() => {
  vi.useRealTimers();
});

/** A group whose title and mark a caller changes state on, as the panels do. */
function Group({ start = SAVE_IDLE }: { start?: SaveStatus }) {
  const [status, setStatus] = useState<SaveStatus>(start);
  return (
    <div>
      <p>Camera controls</p>
      <button type="button" onClick={() => setStatus(saving('Saving the camera controls.'))}>
        change
      </button>
      <button type="button" onClick={() => setStatus(saved())}>
        saved
      </button>
      <button type="button" onClick={() => setStatus(failed('The camera did not answer.'))}>
        fail
      </button>
      <SaveState status={status} />
    </div>
  );
}

const slot = () => screen.getByText('Camera controls').parentElement?.querySelector('[aria-live="polite"]');

it('keeps its slot in the page, at every state, so nothing beside it can move', () => {
  renderPage(<Group />);
  expect(slot()).toHaveAttribute('data-phase', 'idle');
  fireEvent.click(screen.getByRole('button', { name: 'change' }));
  expect(slot()).toHaveAttribute('data-phase', 'saving');
  expect(screen.getByText('Camera controls')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'saved' }));
  expect(slot()).toHaveAttribute('data-phase', 'saved');
  fireEvent.click(screen.getByRole('button', { name: 'fail' }));
  expect(slot()).toHaveAttribute('data-phase', 'error');
});

it('spins while the change is saved, then checks it and says both to a screen reader', async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  renderPage(<Group />);
  expect(screen.queryByTestId('save-spinner')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'change' }));
  expect(screen.getByTestId('save-spinner')).toBeInTheDocument();
  expect(slot()).toHaveTextContent('Saving the camera controls.');
  fireEvent.click(screen.getByRole('button', { name: 'saved' }));
  expect(screen.queryByTestId('save-spinner')).not.toBeInTheDocument();
  expect(slot()).toHaveTextContent('Saved and applied.');
  expect(slot()).toHaveAttribute('data-phase', 'saved');
  await act(async () => {
    await vi.advanceTimersByTimeAsync(SAVE_MARK_MS + 100);
  });
  // The check goes by itself; the slot stays, and what was said stays said.
  expect(slot()).toHaveAttribute('data-phase', 'idle');
  expect(slot()).toHaveTextContent('Saved and applied.');
});

it('says why it failed, in a tooltip and to a screen reader', async () => {
  renderPage(<Group start={failed('The camera did not answer.')} />);
  expect(slot()).toHaveTextContent('The camera did not answer.');
  await userEvent.hover(screen.getByTestId('save-error'));
  // The mark itself is hidden from the accessibility tree: the live region beside it has already said the same words.
  expect(await screen.findByRole('tooltip', { hidden: true })).toHaveTextContent('The camera did not answer.');
});
