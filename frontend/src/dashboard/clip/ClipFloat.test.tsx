import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { MakeVideo } from '../review/Player';
import { ClipFloat, floatingRect } from './ClipFloat';

it.each([
  [1440, 900, 960],
  [390, 844, 358],
])('fits the uncropped player inside %s by %s', (width, height, expectedWidth) => {
  const box = floatingRect(width, height);
  expect(box.width).toBe(expectedWidth);
  expect(box.left).toBeGreaterThanOrEqual(16);
  expect(box.left + box.width).toBeLessThanOrEqual(width - 16);
  expect(box.top).toBeGreaterThanOrEqual(16);
  expect(box.top + box.height).toBeLessThanOrEqual(height - 16);
});

it('gives a portrait camera a tall window that still fits', () => {
  const box = floatingRect(1440, 900, 9 / 16);
  expect(box.width).toBeLessThan(box.height);
  expect(box.top + box.height).toBeLessThanOrEqual(900 - 16);
});

/** A stand-in for the gateway's decoder: real DOM media events, nothing played. */
const makeVideo: MakeVideo = () => {
  const element = document.createElement('video');
  Object.defineProperty(element, 'readyState', { get: () => 4 });
  element.play = vi.fn(async () => {});
  element.pause = vi.fn();
  element.load = vi.fn();
  return Object.assign(element, { close: vi.fn() });
};

it('keeps what the full player opens (its help popover) above the window, not behind it', async () => {
  server.use(
    http.get('/ambient', () => HttpResponse.json({ devices: {} })),
    http.get('/review/boxes', () => HttpResponse.json({ boxes: [] })),
  );
  const origin = document.createElement('button');
  document.body.append(origin);
  const segments = [{ cam: 'cam0', start: 100, end: 160, url: 'gateway:1:0:60000000' }];
  renderPage(
    <ClipFloat
      cam="cam0"
      start={100}
      end={160}
      label="squats"
      origin={origin}
      onClose={() => {}}
      clip={{ at: 100, segments }}
      full
      makeVideo={makeVideo}
    />,
  );
  const window = screen.getByRole('dialog', { name: 'squats' });
  await userEvent.click(screen.getByRole('button', { name: 'Shortcuts and timeline help' }));
  const help = (await screen.findByRole('table', { name: 'Keyboard shortcuts' })).closest<HTMLElement>(
    '[style*="z-index"]',
  );
  expect(Number(help?.style.zIndex)).toBeGreaterThan(Number(window.style.zIndex));
  origin.remove();
});
