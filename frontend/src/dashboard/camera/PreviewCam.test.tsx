import { act, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { PreviewCam } from './PreviewCam';
import type { PreviewState } from './preview';

const fake = vi.hoisted(() => ({ listeners: [] as ((state: PreviewState) => void)[], off: vi.fn() }));
vi.mock('./preview', () => ({
  attachPreview: vi.fn((_cam: string, _canvas: HTMLCanvasElement, changed: (state: PreviewState) => void) => {
    fake.listeners.push(changed);
    changed({ status: 'loading', capturedAt: null, error: null });
    return fake.off;
  }),
}));
beforeEach(() => {
  fake.listeners.length = 0;
  fake.off.mockClear();
});

it('uses a separate labelled preview canvas and releases its watcher when switched or removed', async () => {
  const { rerender, unmount } = render(<PreviewCam cam="camA" alt="The entrance" />);
  await waitFor(() => expect(fake.listeners).toHaveLength(1));
  const canvas = screen.getByRole('img', { name: 'The entrance' });
  expect(canvas).toHaveClass('preview-canvas', 'off');
  expect(canvas).not.toHaveClass('live-canvas');
  expect(canvas).toHaveAttribute('data-cam', 'camA');
  expect(screen.getByText('Loading preview…')).toBeInTheDocument();
  rerender(<PreviewCam cam="camB" alt="The hallway" />);
  expect(fake.off).toHaveBeenCalledOnce();
  expect(screen.getByRole('img', { name: 'The hallway' })).not.toBe(canvas);
  await waitFor(() => expect(fake.listeners).toHaveLength(2));
  unmount();
  expect(fake.off).toHaveBeenCalledTimes(2);
});

it('dims retained pictures and names unavailable evidence instead of showing them as fresh', async () => {
  render(<PreviewCam cam="camA" alt="The entrance" />);
  await waitFor(() => expect(fake.listeners).toHaveLength(1));
  const listener = fake.listeners[0];
  if (!listener) throw Error('No preview watcher');
  const at = Date.now() / 1000 - 5;
  act(() => listener({ status: 'ready', capturedAt: at, error: null }));
  expect(screen.getByText('Preview · 5 s ago')).toBeInTheDocument();
  act(() => listener({ status: 'stale', capturedAt: at, error: 'Preview connection unavailable.' }));
  expect(screen.getByRole('img', { name: 'The entrance' })).toHaveClass('stale');
  expect(screen.getByText('Older preview · 5 s ago')).toBeInTheDocument();
  expect(screen.getByText('Open camera for live video')).toBeInTheDocument();
  expect(screen.queryByText('Loading preview…')).not.toBeInTheDocument();
});
