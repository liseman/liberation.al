// Vitest setup: jest-dom matchers, the browser APIs jsdom lacks that Mantine and the pages use, and one MSW server
// that every test's HTTP goes through. Unhandled frames fail teardown even when a page catches the network error.
import '@testing-library/jest-dom/vitest';
import { setImmediate as nextTurn } from 'node:timers/promises';
import { notifications } from '@mantine/notifications';
import { cleanup, configure } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, expect, vi } from 'vitest';
import { cancelQueuedRefreshes } from '../dashboard/livetab/api';
import { releaseClients } from './render';
import { server } from './server';

// findBy and waitFor resolve on the first matching DOM change, so their timeout only bounds how long a failing check
// takes to report. Testing Library's 1 s default assumed a CPU of our own: on a quiet machine a Settings page took
// 960 ms of it to render, and under the shared machine's load whole files failed. 10 s, with vite.config.ts's
// 30 s per test, leaves room for that load.
configure({ asyncUtilTimeout: 10_000 });

window.matchMedia ??= (query: string) =>
  ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }) as MediaQueryList;

globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

Element.prototype.scrollIntoView ??= () => {};
Element.prototype.setPointerCapture ??= () => {};
window.scrollTo = () => {};

// jsdom has no canvas: pages get a context that records nothing.
HTMLCanvasElement.prototype.getContext = vi.fn(
  () =>
    ({
      clearRect() {},
      drawImage() {},
      fillRect() {},
      strokeRect() {},
      beginPath() {},
      moveTo() {},
      lineTo() {},
      closePath() {},
      stroke() {},
    }) as unknown as CanvasRenderingContext2D,
) as unknown as HTMLCanvasElement['getContext'];

// Capture before awaiting diagnostics, so a caller catching fetch's rejection
// cannot hide a missing handler. The custom policy must throw too: defaults.error
// only prints in MSW 3 and does not reject the network frame by itself.
const unhandledFrames: Promise<string>[] = [];
beforeAll(() =>
  server.listen({
    onUnhandledFrame: async ({ frame, defaults }) => {
      const message = frame.getUnhandledMessage();
      unhandledFrames.push(message);
      const text = await message;
      await defaults.error();
      throw new Error(text);
    },
  }),
);

async function assertHandledFrames(): Promise<void> {
  const messages = await Promise.all(unhandledFrames.splice(0));
  expect(messages, `Unhandled MSW requests:\n${messages.join('\n')}`).toEqual([]);
}
afterEach(async () => {
  cleanup();
  // Toasts live in Mantine's notifications store, one per module: a sticky one (a Retry) would outlast its test and
  // take the focus or the text a later test looks for.
  notifications.clean();
  // A correction's refresh reads the transcript from a timer of its own (livetab/api.ts): drop any still queued or
  // reading, so it can't read after this test's mocks are reset or hold a request slot into the next test.
  cancelQueuedRefreshes();
  // Reads a page started without watching them (fetchQuery) outlive its unmount: cancel them, freeing their slots.
  await releaseClients();
  // Let fire-and-forget handler lookups finish before resetting their mocks or
  // checking the failures. A native turn also works while a test uses fake timers.
  await nextTurn();
  server.resetHandlers();
  localStorage.clear();
  sessionStorage.clear(); // Ask keeps its job here
  await assertHandledFrames();
});
afterAll(async () => {
  server.close();
  await nextTurn();
  await assertHandledFrames();
});

/** jsdom never loads images. This one "loads" any src as a 640x360 picture, or fails when the src says "missing". */
class FakeImage {
  naturalWidth = 640;
  naturalHeight = 360;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  #src = '';
  get src() {
    return this.#src;
  }
  set src(value: string) {
    this.#src = value;
    queueMicrotask(() => (value.includes('missing') ? this.onerror?.() : this.onload?.()));
  }
}
globalThis.Image = FakeImage as unknown as typeof Image;
