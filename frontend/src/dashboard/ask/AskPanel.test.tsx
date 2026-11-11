import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { AskPanel, fitWindow, savedSpot } from './AskPanel';

const KEY = 'liberational.ask-window';
const screenSize = (width: number, height: number) => {
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: width });
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: height });
};

beforeEach(() => {
  localStorage.clear();
  screenSize(1440, 900);
});
afterEach(() => vi.restoreAllMocks());

describe('fitWindow', () => {
  const view = { width: 1440, height: 900 };

  it('opens at the top right under the header the first time', () => {
    expect(fitWindow(null, view, 56)).toEqual({ left: 984, top: 64, width: 440, height: 620 });
  });

  it('keeps a window dragged past an edge on screen', () => {
    expect(fitWindow({ left: 2000, top: -50 }, view, 56)).toEqual({ left: 992, top: 64, width: 440, height: 620 });
    expect(fitWindow({ left: -300, top: 5000 }, view, 56)).toMatchObject({ left: 8, top: 844, height: 48 });
  });

  it('shrinks to fit a small viewport', () => {
    expect(fitWindow({ left: 0, top: 0 }, { width: 400, height: 500 }, 56)).toEqual({
      left: 8,
      top: 64,
      width: 384,
      height: 428,
    });
  });
});

function Window({ onClose = () => {} }: { onClose?: () => void }) {
  return (
    <>
      <button type="button" data-ask-trigger>
        Ask
      </button>
      <AskPanel opened onClose={onClose} title="Ask">
        <textarea aria-label="Question" />
      </AskPanel>
    </>
  );
}

it('floats over the page, outside the tab, and puts focus in its question box', async () => {
  const { container } = renderPage(<Window />);
  const dialog = screen.getByRole('dialog', { name: 'Ask' });
  expect(container.contains(dialog)).toBe(false);
  expect(dialog.parentElement).toBe(document.body);
  expect(dialog).toHaveStyle({ left: '984px', top: '8px', width: '440px' });
  await waitFor(() => expect(screen.getByLabelText('Question')).toHaveFocus());
});

it('drags by its header, stays on screen, and is where it was left next time', () => {
  const { unmount } = renderPage(<Window />);
  const header = within(screen.getByRole('dialog')).getByText('Ask').closest('header') as HTMLElement;
  fireEvent.pointerDown(header, { button: 0, clientX: 1000, clientY: 20, pointerId: 1 });
  fireEvent.pointerMove(header, { clientX: 700, clientY: 220, pointerId: 1 });
  fireEvent.pointerUp(header, { clientX: 400, clientY: 320, pointerId: 1 });
  expect(screen.getByRole('dialog')).toHaveStyle({ left: '384px', top: '308px' });
  expect(savedSpot()).toEqual({ left: 384, top: 308 });
  unmount();
  renderPage(<Window />);
  expect(screen.getByRole('dialog')).toHaveStyle({ left: '384px', top: '308px' });
});

it('moves with the arrow keys on its move handle', async () => {
  localStorage.setItem(KEY, JSON.stringify({ left: 300, top: 200 }));
  renderPage(<Window />);
  await waitFor(() => expect(screen.getByLabelText('Question')).toHaveFocus()); // where opening puts focus
  const handle = screen.getByRole('button', { name: 'Move Ask with the arrow keys' });
  handle.focus();
  await userEvent.keyboard('{ArrowLeft}{Shift>}{ArrowDown}{/Shift}');
  expect(screen.getByRole('dialog')).toHaveStyle({ left: '284px', top: '264px' });
});

it('comes back on screen after the window shrinks', () => {
  localStorage.setItem(KEY, JSON.stringify({ left: 1000, top: 600 }));
  renderPage(<Window />);
  screenSize(900, 700);
  fireEvent(window, new Event('resize'));
  expect(screen.getByRole('dialog')).toHaveStyle({ left: '452px', top: '600px', maxHeight: '92px' });
});

it('ignores a saved spot it can not read', () => {
  localStorage.setItem(KEY, '{not json');
  renderPage(<Window />);
  expect(screen.getByRole('dialog')).toHaveStyle({ left: '984px' });
});

it.each([
  ['Escape', () => userEvent.keyboard('{Escape}')],
  ['the close button', () => userEvent.click(screen.getByRole('button', { name: 'Close Ask' }))],
])('closes on %s and gives focus back to the Ask button', async (_, act) => {
  const onClose = vi.fn();
  renderPage(<Window onClose={onClose} />);
  await waitFor(() => expect(screen.getByLabelText('Question')).toHaveFocus());
  await act();
  expect(onClose).toHaveBeenCalledOnce();
  expect(screen.getByRole('button', { name: 'Ask' })).toHaveFocus();
});

it('is a bottom sheet on a phone, with nothing to drag', () => {
  screenSize(390, 844);
  renderPage(<Window />);
  const dialog = screen.getByRole('dialog', { name: 'Ask' });
  expect(dialog).toHaveAttribute('data-ask-sheet', 'true');
  expect(screen.queryByRole('button', { name: /Move Ask/ })).not.toBeInTheDocument();
});
