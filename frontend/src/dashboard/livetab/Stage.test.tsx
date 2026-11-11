import { render } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { beforeEach, expect, it, vi } from 'vitest';
import { newQueryClient, Providers } from '../../shared/Providers';
import { server } from '../../test/server';
import { setActiveCameras } from '../shell';
import { Stage } from './Stage';

/** An EventSource that never sends: the camera boxes subscribe to one. */
class QuietEventSource {
  static readonly CLOSED = 2;
  readyState = 0;
  onmessage = null;
  onerror = null;
  addEventListener() {}
  close() {}
}

beforeEach(() => {
  localStorage.clear();
  setActiveCameras([]);
  vi.stubGlobal('EventSource', QuietEventSource);
  server.use(
    http.get('/stats', () => HttpResponse.json({})),
    http.get('/devices', () => HttpResponse.json({ control: true, mics: [], cameras: [], notes: {} })),
    http.get('/config', () => HttpResponse.json({ ok: true, control: true, needs_setup: false, sections: [] })),
    http.get('/people', () => HttpResponse.json({ people: [], pets: [] })),
  );
});

/** The cameras' place before anything is known: the stage's own height, held, so nothing under it jumps when the
 *  pictures arrive. */
it('holds the stage place at the height remembered for this window width', () => {
  vi.stubGlobal('innerWidth', 1024);
  localStorage.setItem('liberational.height.live.stage.wide', '333');
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false } });
  const { container } = render(
    <Providers client={client} test>
      <Stage last={{}} live={[]} />
    </Providers>,
  );
  expect((container.querySelector('.cam-wait') as HTMLElement).style.height).toBe('333px');
});

it("doesn't take a phone's stage height on a wide window", () => {
  vi.stubGlobal('innerWidth', 1024);
  localStorage.setItem('liberational.height.live.stage.narrow', '555');
  const client = newQueryClient();
  client.setDefaultOptions({ queries: { retry: false } });
  const { container } = render(
    <Providers client={client} test>
      <Stage last={{}} live={[]} />
    </Providers>,
  );
  expect((container.querySelector('.cam-wait') as HTMLElement).style.height).toBe('');
});
