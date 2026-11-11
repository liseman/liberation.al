import { fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { setPref, usePrefs } from '../lib/prefs';
import type { UsageHistory, UsageSample } from '../sources/api';
import { DEFAULT_SIZE, placed, UsagePanel } from './UsagePanel';

const GB = 1073741824;
const EMPTY: UsageHistory = { step: 5, window: 1800, cpu_count: 4, mem_total: 32 * GB, gpus: [], samples: [] };
const panel = () => screen.queryByRole('region', { name: 'Resource monitor' });
const opened = (pref = {}) => setPref('usagePanel', { open: true, ...pref });

beforeEach(() => {
  server.use(http.get('/stats/history', () => HttpResponse.json(EMPTY)));
  opened();
});
afterEach(() => {
  setPref('usagePanel', undefined);
  vi.useRealTimers();
});

describe('the task manager panel', () => {
  it('asks for the history only while it is open', async () => {
    const asked: string[] = [];
    server.use(
      http.get('/stats/history', ({ request }) => {
        asked.push(request.url);
        return HttpResponse.json(EMPTY);
      }),
    );
    setPref('usagePanel', { open: false });
    const { unmount } = renderPage(<UsagePanel />);
    await new Promise(resolve => setTimeout(resolve, 50));
    expect(asked).toEqual([]);
    unmount();
    opened();
    renderPage(<UsagePanel />);
    await vi.waitFor(() => expect(asked).toHaveLength(1));
  });

  it('floats where it was left, at the size it was given', () => {
    opened({ x: 200, y: 120, w: 600, h: 500 });
    renderPage(<UsagePanel />);
    const style = (panel() as HTMLElement).style;
    expect([style.left, style.top, style.width, style.height]).toEqual(['200px', '120px', '600px', '500px']);
  });

  it('is dragged by its title bar, and saves where it was let go', () => {
    opened({ x: 200, y: 120 });
    renderPage(<UsagePanel />);
    const bar = screen.getByRole('heading', { name: 'Resource monitor' }).parentElement as HTMLElement;
    bar.setPointerCapture = () => {};
    fireEvent.pointerDown(bar, { button: 0, clientX: 210, clientY: 130, pointerId: 1 });
    fireEvent.pointerMove(bar, { clientX: 310, clientY: 180, pointerId: 1 });
    fireEvent.pointerUp(bar, { clientX: 310, clientY: 180, pointerId: 1 });
    expect(usePrefs.getState().usagePanel).toMatchObject({ open: true, x: 300, y: 170 });
  });

  it('docks as a bottom sheet on a phone, with no floating place', () => {
    vi.spyOn(window, 'matchMedia').mockImplementation(
      query =>
        ({
          matches: query === '(max-width: 700px)',
          media: query,
          addEventListener() {},
          removeEventListener() {},
        }) as unknown as MediaQueryList,
    );
    opened({ x: 200, y: 120 });
    renderPage(<UsagePanel />);
    expect((panel() as HTMLElement).style.left).toBe('');
    expect(screen.getByRole('heading', { name: 'Resource monitor' }).parentElement).not.toHaveAttribute(
      'data-draggable',
    );
  });
});

describe('placed', () => {
  const view = { width: 1440, height: 900 };
  it('opens new at the top right, under the header', () => {
    expect(placed({}, DEFAULT_SIZE.w, view)).toEqual({ x: 1440 - DEFAULT_SIZE.w - 16, y: 64 });
  });
  it('keeps the title bar on screen when the window shrank or it was dragged off', () => {
    expect(placed({ x: 1400, y: 880 }, 560, { width: 800, height: 600 })).toEqual({ x: 800 - 96, y: 600 - 40 });
    expect(placed({ x: -1000, y: -50 }, 560, view)).toEqual({ x: 96 - 560, y: 0 });
  });
});

describe("what's using it", () => {
  const T = 1_800_000_000;
  const at = (k: number, gpu: UsageSample['gpu'] = []): UsageSample => ({
    ts: T + 5 * k,
    cpu: { listen: 10 + k, 'see-actions': 30, other: 5 },
    mem: { listen: 2 * GB, 'see-actions': 4 * GB, other: 6 * GB },
    gpu,
  });
  const gpus = (k: number): UsageSample['gpu'] => [
    { index: 0, used: 10 * GB, util: 60 + k, apps: { 'see-actions': 6 * GB, 'ask-ollama': 3 * GB, other: GB } },
    { index: 1, used: 5 * GB, util: 4, apps: { ollama: 4 * GB, other: GB } },
  ];
  const history = (samples: UsageSample[], withGpus: boolean): UsageHistory => ({
    ...EMPTY,
    gpus: withGpus
      ? [
          { index: 0, name: 'RTX 3090', total: 24 * GB },
          { index: 1, name: 'RTX 3090', total: 24 * GB },
        ]
      : [],
    samples,
  });
  const section = async () => screen.findByRole('region', { name: 'Resource monitor' });

  it('stacks each service over time, with a legend, the newest values, and a toggle per GPU', async () => {
    server.use(
      http.get('/stats/history', () =>
        HttpResponse.json(
          history(
            [0, 1, 2].map(k => at(k, gpus(k))),
            true,
          ),
        ),
      ),
    );
    renderPage(<UsagePanel />);
    const region = await section();
    const legend = await within(region).findByRole('list', { name: 'Services' });
    // Every service busy on any resource gets a band, in name order, so it keeps its colour on every toggle.
    expect(
      within(legend)
        .getAllByRole('listitem')
        .map(li => li.textContent),
    ).toEqual(['ask-ollama', 'listen', 'Ollama', 'see-actions', 'Everything else']);
    const now = within(region).getByRole('table', { name: 'Processor, % of the machine, now' });
    expect(
      within(now)
        .getAllByRole('row')
        .map(r => r.textContent),
    ).toEqual(['see-actions30%', 'listen12%', 'Everything else5.0%']);
    await userEvent.click(within(region).getByRole('radio', { name: 'GPU 0' }));
    const gpu = within(region).getByRole('table', { name: 'GPU 0 memory (RTX 3090), now' });
    expect(
      within(gpu)
        .getAllByRole('row')
        .map(r => r.textContent),
    ).toEqual(['see-actions6.0 GB', 'ask-ollama3.0 GB', 'Everything else1.0 GB']);
    expect(within(region).getByRole('radio', { name: 'GPU 1' })).toBeInTheDocument();
  });

  it('reads each moment from the keyboard', async () => {
    server.use(
      http.get('/stats/history', () =>
        HttpResponse.json(
          history(
            [0, 1, 2].map(k => at(k)),
            false,
          ),
        ),
      ),
    );
    renderPage(<UsagePanel />);
    const chart = await within(await section()).findByRole('slider');
    chart.focus();
    await userEvent.keyboard('{ArrowLeft}');
    const tip = await screen.findByRole('tooltip');
    expect(
      within(tip)
        .getAllByRole('listitem')
        .map(li => li.textContent),
    ).toEqual(['see-actions30%', 'listen11%', 'Everything else5.0%']);
    expect(chart).toHaveAttribute('aria-valuetext', expect.stringMatching(/see-actions 30%, listen 11%/));
  });

  it('offers no GPU without one, and says it is gathering before there is enough to draw', async () => {
    server.use(http.get('/stats/history', () => HttpResponse.json(history([at(0)], false))));
    renderPage(<UsagePanel />);
    const region = await section();
    expect(await within(region).findByText(/Gathering a few samples/)).toBeInTheDocument();
    expect(within(region).queryByRole('radio', { name: /GPU/ })).toBeNull();
    expect(within(region).getByRole('radio', { name: 'Memory' })).toBeInTheDocument();
  });

  it('asks only for samples newer than those it has', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const asked: string[] = [];
    server.use(
      http.get('/stats/history', ({ request }) => {
        const since = new URL(request.url).searchParams.get('since') ?? '';
        asked.push(since);
        return HttpResponse.json(history(since === '0' ? [at(0), at(1)] : [at(2)], false));
      }),
    );
    renderPage(<UsagePanel />);
    await within(await section()).findByRole('slider');
    await vi.advanceTimersByTimeAsync(5000);
    await vi.waitFor(() => expect(asked).toEqual(['0', String(T + 5)]));
  });
});
