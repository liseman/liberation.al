import { act, fireEvent, screen } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { TOAST_ACTION_MS, TOAST_MS, Toast } from './Toast';

beforeEach(() => vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] }));
afterEach(() => vi.useRealTimers());

function Said({ text, error = false, undo = false }: { text: string; error?: boolean; undo?: boolean }) {
  const [shown, setShown] = useState<string | null>(text);
  return (
    <Toast
      text={shown}
      error={error}
      action={undo ? <button type="button">Undo</button> : undefined}
      onClose={() => setShown(null)}
    />
  );
}

const wait = (ms: number) => act(async () => void (await vi.advanceTimersByTimeAsync(ms)));

it('goes by itself after a while', async () => {
  renderPage(<Said text="Pet added." />);
  expect(screen.getByRole('status')).toHaveTextContent('Pet added.');
  await wait(TOAST_MS - 100);
  expect(screen.getByRole('status')).toBeInTheDocument();
  await wait(200);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});

it('stays longer with an Undo, and while the pointer is on it', async () => {
  renderPage(<Said text="Face 5 tagged as Kara." undo />);
  await wait(TOAST_MS + 100);
  expect(screen.getByRole('status')).toBeInTheDocument();
  fireEvent.mouseEnter(screen.getByRole('status'));
  await wait(TOAST_ACTION_MS * 2);
  expect(screen.getByRole('status')).toBeInTheDocument();
  fireEvent.mouseLeave(screen.getByRole('status'));
  await wait(TOAST_ACTION_MS + 100);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});

it('keeps an error until it is dismissed', async () => {
  renderPage(<Said text="Tagging failed" error />);
  await wait(TOAST_ACTION_MS * 2);
  expect(screen.getByRole('alert')).toHaveTextContent('Tagging failed');
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
  await wait(1); // it slides out
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

// Codex on #1990: a toast dismissed while hovered or focused leaves no hold behind for the next one.
function Twice() {
  const [shown, setShown] = useState<string | null>('Pet added.');
  return (
    <>
      <Toast text={shown} onClose={() => setShown(null)} />
      <button type="button" onClick={() => setShown('Pet name updated.')}>
        Save again
      </button>
    </>
  );
}

it('times out the next confirmation after one was dismissed while hovered and focused', async () => {
  renderPage(<Twice />);
  const status = screen.getByRole('status');
  fireEvent.mouseEnter(status);
  fireEvent.focus(screen.getByRole('button', { name: 'Dismiss' }));
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss' })); // gone without a leave or a blur
  await wait(1);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Save again' }));
  expect(screen.getByRole('status')).toHaveTextContent('Pet name updated.');
  await wait(TOAST_MS + 100);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});

it('says a Message, with Undo calling back with its token, and an error as an alert', () => {
  const onUndo = vi.fn();
  const first = renderPage(
    <Toast message={{ text: 'Named Dana.', undo: 'tok-1' }} onUndo={onUndo} onClose={() => {}} />,
  );
  expect(screen.getByRole('status')).toHaveTextContent('Named Dana.');
  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  expect(onUndo).toHaveBeenCalledWith('tok-1');
  first.unmount();
  renderPage(<Toast message={{ err: 'That did not save.' }} onUndo={onUndo} onClose={() => {}} />);
  expect(screen.getByRole('alert')).toHaveTextContent('That did not save.');
  expect(screen.queryByRole('button', { name: 'Undo' })).not.toBeInTheDocument();
});

it('keeps a sticky Message up until it is replaced', () => {
  const onClose = vi.fn();
  renderPage(<Toast message={{ text: 'Saving', sticky: true }} onClose={onClose} />);
  act(() => {
    vi.advanceTimersByTime(TOAST_ACTION_MS * 2);
  });
  expect(onClose).not.toHaveBeenCalled();
});

it('offers no Undo for an empty undo token', () => {
  renderPage(<Toast message={{ text: 'Nothing to undo.', undo: '' }} onUndo={() => {}} onClose={() => {}} />);
  expect(screen.getByRole('status')).toHaveTextContent('Nothing to undo.');
  expect(screen.queryByRole('button', { name: 'Undo' })).not.toBeInTheDocument();
});

// #2596: one queue for every page, so two things said at once both show, and closing one leaves the other.
it('stacks toasts from two places instead of replacing one with the other', async () => {
  const closed = vi.fn();
  renderPage(
    <>
      <Toast text="Pet added." onClose={closed} />
      <Toast text="Couldn't save the name" error onClose={() => {}} />
    </>,
  );
  expect(screen.getByRole('status')).toHaveTextContent('Pet added.');
  expect(screen.getByRole('alert')).toHaveTextContent("Couldn't save the name");
  await wait(TOAST_MS + 100);
  expect(closed).toHaveBeenCalledOnce();
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  expect(screen.getByRole('alert')).toBeInTheDocument();
});

function SavingToast({ onClose }: { onClose: () => void }) {
  const [saved, setSaved] = useState(false);
  return (
    <>
      <Toast text={saved ? 'Saved.' : 'Saving'} sticky={!saved} onClose={onClose} />
      <button type="button" onClick={() => setSaved(true)}>
        Finish
      </button>
    </>
  );
}

it('replaces what one Toast said without calling onClose, and restarts the clock', async () => {
  const closed = vi.fn();
  renderPage(<SavingToast onClose={closed} />);
  fireEvent.click(screen.getByRole('button', { name: 'Finish' }));
  await wait(1);
  expect(screen.getByRole('status')).toHaveTextContent('Saved.');
  expect(screen.getAllByRole('status')).toHaveLength(1);
  expect(closed).not.toHaveBeenCalled();
  await wait(TOAST_MS + 100);
  expect(closed).toHaveBeenCalledOnce();
});
