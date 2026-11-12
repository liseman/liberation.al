import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../../test/render';
import { server } from '../../../test/server';
import type { Stats, UsageHistory } from '../../sources/api';
import { stats } from '../../sources/fixtures';
import { Machine } from './Machine';

/** Every process's models open: each folds to one line; these tests read the models under it. */
async function openModels() {
  await screen.findByRole('heading', { name: 'Models in memory' });
  for (const t of screen.queryAllByRole('button', { expanded: false })) await userEvent.click(t);
}

const GB = 1073741824;
const serve = (s: Stats) => server.use(http.get('/stats', () => HttpResponse.json(s)));
const health = () => screen.getByRole('region', { name: 'Health' });
const tile = (name: string) => screen.getByRole('region', { name });

const EMPTY: UsageHistory = { step: 5, window: 1800, cpu_count: 4, mem_total: 32 * GB, gpus: [], samples: [] };

beforeEach(() => {
  server.use(
    http.get('/services', () => HttpResponse.json({ ok: true, control: true, platform: 'linux-x86_64', services: [] })),
    http.get('/stats/history', () => HttpResponse.json(EMPTY)),
  );
});
afterEach(() => {
  vi.useRealTimers();
});

describe('this machine', () => {
  it('polls every 2 seconds', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let calls = 0;
    server.use(http.get('/stats', () => HttpResponse.json(stats({ cpu: ++calls * 10 }))));
    renderPage(<Machine />);
    expect(await screen.findByText('10%')).toBeInTheDocument();
    await vi.advanceTimersByTimeAsync(1900);
    expect(calls).toBe(1);
    await vi.advanceTimersByTimeAsync(100);
    expect(await screen.findByText('20%')).toBeInTheDocument();
  });

  it("leaves What's using it to the header's task manager", async () => {
    serve(stats());
    renderPage(<Machine />);
    expect(await screen.findByRole('region', { name: 'Health' })).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: "What's using it" })).toBeNull();
  });

  it('says what the machine is, and how it is doing', async () => {
    serve(stats());
    renderPage(<Machine />);
    expect(
      await screen.findByText('Linux on x86-64, with 8 processor cores and 16.0 GB of memory.'),
    ).toBeInTheDocument();
    expect(health()).toHaveTextContent('Running smoothly');
    expect(health()).toHaveTextContent('The processor is at 13%, and 31% of memory is in use.');
  });

  it('says it is busy when the processor is or memory has gone to swap', async () => {
    serve(stats({ cpu: 86, swap_used: 8 * GB }));
    renderPage(<Machine />);
    expect(await screen.findByText('Busy')).toBeInTheDocument();
    expect(health()).toHaveTextContent(
      'The processor is at 86%, and 8.0 GB of memory has moved to swap. Things may respond slowly.',
    );
  });

  it('shows the processor, memory, and models in tiles', async () => {
    serve(
      stats({
        ollama: {
          models: [
            { name: 'qwen3:8b', size: 6 * GB, size_vram: 6 * GB, params: '8.2B', ctx: 40960 },
            { name: 'emb:4b', size: 2 * GB, size_vram: 2 * GB, params: '4.0B', ctx: 4096 },
          ],
          up: true,
          missing: [],
        },
      }),
    );
    renderPage(<Machine />);
    await screen.findByText('Running smoothly');
    expect(tile('Processor')).toHaveTextContent('13%');
    expect(tile('Processor')).toHaveTextContent('8 cores. Load 1.2 now, 0.3 over the last 15 minutes.');
    expect(within(tile('Processor')).getByRole('img')).toHaveAccessibleName('13% of the processor in use');
    expect(tile('Memory')).toHaveTextContent('5.0 GB of 16.0 GB');
    expect(tile('Memory')).toHaveTextContent('31% in use. 0.5 GB in swap.');
    expect(tile('Models')).toHaveTextContent('2 loaded');
    expect(tile('Models')).toHaveTextContent('8.0 GB, all in GPU memory. Nothing missing.');

    await openModels();
    const ollama = screen.getByRole('region', { name: 'Ollama models' });
    const models = within(ollama).getAllByRole('listitem');
    expect(models).toHaveLength(2);
    expect(models[0]).toHaveTextContent('qwen3:8b');
    expect(models[0]).toHaveTextContent('8.2B parameters, context 40,960');
    expect(models[0]).toHaveTextContent('6.0 GB model allocation');
    expect(models[1]).toHaveTextContent('emb:4b');
    expect(models[1]).toHaveTextContent('4.0B parameters, context 4,096');
    expect(models[1]).toHaveTextContent('2.0 GB model allocation');
    expect(ollama).toHaveTextContent('8.0 GB GPU model allocations');
    expect(ollama).toHaveTextContent('Process memory unavailable');
    expect(ollama).toHaveTextContent('Load time unavailable');
    expect(screen.getByText('Application model reports are unavailable on this server.')).toBeInTheDocument();
    expect(screen.getByText('Ollama running')).toBeInTheDocument();
  });

  it('leaves out swap when none is used, and says when no models are loaded', async () => {
    serve(stats({ swap_used: 0, ollama: { models: [], error: null, up: true } }));
    renderPage(<Machine />);
    await screen.findByText('Running smoothly');
    expect(tile('Memory')).not.toHaveTextContent('swap');
    expect(tile('Models')).toHaveTextContent('0 loaded');
    expect(tile('Models')).toHaveTextContent('None loaded right now. Nothing missing.');
    expect(screen.queryByRole('region', { name: 'Ollama models' })).not.toBeInTheDocument();
  });

  it("shows Ollama's error in place of the models", async () => {
    serve(stats({ ollama: { error: 'connection refused' } }));
    renderPage(<Machine />);
    await screen.findByText('Running smoothly');
    expect(tile('Models')).toHaveTextContent('Ollama: connection refused');
    expect(screen.getByText('Ollama not running')).toBeInTheDocument();
  });

  it('says how to start Ollama when it is down', async () => {
    serve(
      stats({
        ollama: { models: [], error: 'refused', up: false, missing: [], message: 'Ollama is not running: start it' },
      }),
    );
    renderPage(<Machine />);
    await screen.findByText('Running smoothly');
    expect(tile('Models')).toHaveTextContent('Off');
    expect(tile('Models')).toHaveTextContent('Ollama is not running: start it');
  });

  it("shows the server's error when it can't measure", async () => {
    serve({ error: 'psutil not installed', procs: [] });
    renderPage(<Machine />);
    expect(await screen.findByText('psutil not installed')).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Health' })).not.toBeInTheDocument();
  });
});

describe('all model sources', () => {
  const loaded = (key: string, name: string, purpose: string, device: string) => ({
    key,
    name,
    purpose,
    device,
    since: Date.parse('2026-10-31T09:00:00Z') / 1000,
  });

  it("folds each process's models to one line: its name, how many, and its memory", async () => {
    serve(
      stats({
        model_runtime: {
          processes: [
            {
              pid: 501,
              service: 'body-worker',
              since: 1,
              rss_bytes: 2 * GB,
              gpu_bytes: GB,
              models: [
                loaded('models.detector', 'Person detector', 'people', 'cuda:0'),
                loaded('models.pose', 'Pose model', 'pose', 'cuda:0'),
              ],
            },
          ],
          configured: [],
          errors: [],
        },
      }),
    );
    renderPage(<Machine />);
    const group = await screen.findByRole('region', { name: 'body-worker models' });
    const line = within(group).getByRole('button', { expanded: false });
    expect(line).toHaveTextContent('body-worker2 models1.0 GB VRAM, 2.0 GB RSS');
    expect(within(group).queryAllByRole('listitem')).toHaveLength(0);
    await userEvent.click(line);
    expect(within(group).getAllByRole('listitem')).toHaveLength(2);
  });

  it('groups multiple models under one process and counts its memory only once alongside Ollama', async () => {
    serve(
      stats({
        model_runtime: {
          processes: [
            {
              pid: 501,
              service: 'body-worker',
              since: 1,
              rss_bytes: 2 * GB,
              gpu_bytes: GB,
              models: [
                loaded('models.detector', 'Person detector', 'people', 'cuda:0'),
                loaded('models.pose', 'Pose model', 'pose', 'cuda:0'),
              ],
            },
          ],
          configured: [],
          errors: [],
        },
      }),
    );
    renderPage(<Machine />);
    await openModels();
    const group = await screen.findByRole('region', { name: 'body-worker models' });
    expect(group).toHaveTextContent('PID 501');
    expect(within(group).getAllByRole('listitem')).toHaveLength(2);
    expect(within(group).getAllByText('1.0 GB VRAM, 2.0 GB RSS')).toHaveLength(1); // once, on its line
    const detector = within(group).getAllByRole('listitem')[0] as HTMLElement;
    expect(detector).toHaveTextContent('Person detector');
    expect(detector).toHaveTextContent('people');
    expect(detector).toHaveTextContent('GPU');
    expect(detector).toHaveTextContent('Loaded since');
    expect(detector.querySelector('time')).toHaveAttribute('datetime', '2026-10-31T09:00:00.000Z');
    expect(tile('Models')).toHaveTextContent('3 loaded');
    expect(tile('Models')).toHaveTextContent('2.0 GB process RSS');
    expect(tile('Models')).toHaveTextContent('1.0 GB process VRAM');
    expect(tile('Models')).toHaveTextContent('6.0 GB Ollama model allocations');
    expect(tile('Models')).not.toHaveTextContent('4.0 GB process RSS');
  });

  it('keeps unknown process measurements distinct from measured zero and preserves loaded models when Ollama is down', async () => {
    serve(
      stats({
        ollama: { up: false, models: [], message: 'Ollama is not running: start it' },
        model_runtime: {
          processes: [
            {
              pid: 601,
              service: 'faces',
              since: 1,
              rss_bytes: null,
              gpu_bytes: null,
              models: [{ ...loaded('models.faces', 'Face embeddings', 'faces', 'mps'), since: null }],
            },
            {
              pid: 602,
              service: 'vad',
              since: 1,
              rss_bytes: 0,
              gpu_bytes: 0,
              models: [loaded('models.vad', 'Voice activity', 'speech detection', 'cpu')],
            },
          ],
          configured: [],
          errors: [],
        },
      }),
    );
    renderPage(<Machine />);
    await openModels();
    const faces = await screen.findByRole('region', { name: 'faces models' });
    expect(faces).toHaveTextContent('RSS unavailable');
    expect(faces).toHaveTextContent('VRAM unavailable');
    expect(faces).toHaveTextContent('Apple GPU');
    expect(faces).toHaveTextContent('Load time unavailable');
    expect(faces).not.toHaveTextContent('0 MB');
    const vad = screen.getByRole('region', { name: 'vad models' });
    expect(vad).toHaveTextContent('0 MB RSS');
    expect(vad).toHaveTextContent('0 MB VRAM');
    expect(vad).toHaveTextContent('CPU');
    expect(tile('Models')).toHaveTextContent('2 loaded');
    expect(tile('Models')).not.toHaveTextContent('Off');
    expect(tile('Models')).toHaveTextContent('Process RSS unavailable');
    expect(screen.getByText('Ollama not running')).toBeInTheDocument();
  });

  it('shows configured unloaded models without claiming that unreadable reports mean unloaded', async () => {
    serve(
      stats({
        ollama: { up: true, models: [] },
        model_runtime: {
          processes: [],
          configured: [
            {
              key: 'models.vad',
              name: 'Voice activity',
              purpose: 'speech detection',
              device: 'cpu',
              state: 'unloaded',
            },
            { key: 'models.asr', name: 'Speech model', purpose: 'speech to text', device: 'auto', state: 'unknown' },
          ],
          errors: ['listen model report is unreadable'],
        },
      }),
    );
    renderPage(<Machine />);
    await openModels();
    const list = await screen.findByRole('list', { name: 'Configured models' });
    const rows = within(list).getAllByRole('listitem');
    expect(rows[0]).toHaveTextContent('Voice activity');
    expect(rows[0]).toHaveTextContent('speech detection');
    expect(rows[0]).toHaveTextContent('Loads when needed');
    expect(rows[1]).toHaveTextContent('Speech model');
    expect(rows[1]).toHaveTextContent('speech to text');
    expect(rows[1]).toHaveTextContent('Status unavailable');
    expect(rows[1]).not.toHaveTextContent('Loads when needed');
    expect(rows[1]).toHaveTextContent('Device not reported');
    expect(screen.getByRole('status')).toHaveTextContent('listen model report is unreadable');
    expect(tile('Models')).toHaveTextContent('0 loaded');
  });

  it("puts a configured model's reason for being used on a second line under its purpose", async () => {
    serve(
      stats({
        ollama: { up: true, models: [] },
        model_runtime: {
          processes: [],
          configured: [
            {
              key: 'objects.mask_backend',
              name: 'EdgeTAM',
              purpose: 'find objects',
              why: "SAM 3, which finds objects by name, isn't installed. EdgeTAM finds everything in view instead.",
              device: 'unknown',
              state: 'unloaded',
            },
          ],
          errors: [],
        },
      }),
    );
    renderPage(<Machine />);
    await openModels();
    const list = await screen.findByRole('list', { name: 'Configured models' });
    // The purpose reads like every other row's; the reason it is this model sits dimmed under it.
    expect(within(list).getByText('find objects')).toBeInTheDocument();
    expect(
      within(list).getByText(
        "SAM 3, which finds objects by name, isn't installed. EdgeTAM finds everything in view instead.",
      ),
    ).toBeInTheDocument();
  });

  it('keeps a configured replacement visible while the old model is still loaded under the same key', async () => {
    serve(
      stats({
        ollama: { up: true, models: [] },
        model_runtime: {
          processes: [
            {
              pid: 701,
              service: 'faces',
              since: 1,
              rss_bytes: GB,
              gpu_bytes: 0,
              models: [loaded('models.faces', 'Face model v1', 'faces', 'cpu')],
            },
          ],
          configured: [
            { key: 'models.faces', name: 'Face model v1', purpose: 'faces', device: 'cpu', state: 'unloaded' },
            { key: 'models.faces', name: 'Face model v2', purpose: 'faces', device: 'cuda:0', state: 'unloaded' },
          ],
          errors: [],
        },
      }),
    );
    renderPage(<Machine />);
    await openModels();
    const configured = await screen.findByRole('list', { name: 'Configured models' });
    expect(within(configured).getAllByRole('listitem')).toHaveLength(1);
    expect(configured).toHaveTextContent('Face model v2');
    expect(configured).toHaveTextContent('Loads when needed');
    expect(configured).not.toHaveTextContent('Face model v1');
    expect(screen.getByRole('region', { name: 'faces models' })).toHaveTextContent('Face model v1');
    expect(tile('Models')).toHaveTextContent('1 loaded');
  });
});

describe('needs installing', () => {
  it('lists each missing model and package with the command that adds it, and copies it', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    serve(
      stats({
        ollama: { models: [], error: null, up: true, missing: ['emb:4b'] },
        system: {
          missing: [
            {
              name: 'ffmpeg',
              why: 'Used to save clips.',
              optional: false,
              services: ['listen'],
              command: 'sudo apt-get update && sudo apt-get install -y ffmpeg',
            },
          ],
          commands: ['sudo apt-get update', 'sudo apt-get install -y ffmpeg'],
        },
      }),
    );
    renderPage(<Machine />);
    const card = await screen.findByRole('region', { name: 'Needs installing' });
    const model = within(card).getByRole('listitem', { name: 'emb:4b' });
    expect(model).toHaveTextContent('emb:4b model');
    expect(within(model).getByText('ollama pull emb:4b')).toBeInTheDocument();
    const ffmpeg = within(card).getByRole('listitem', { name: 'ffmpeg' });
    expect(ffmpeg).toHaveTextContent('package. Used to save clips.');
    expect(within(ffmpeg).getByText('sudo apt-get update && sudo apt-get install -y ffmpeg')).toBeInTheDocument();
    await userEvent.click(within(model).getByRole('button', { name: 'Copy command' }));
    expect(writeText).toHaveBeenCalledWith('ollama pull emb:4b');
    expect(await within(model).findByRole('button', { name: 'Copied' })).toBeInTheDocument();
    expect(tile('Models')).toHaveTextContent('1 missing.');
  });

  it('gives one command for every package when the server gives none of its own', async () => {
    serve(
      stats({
        system: {
          missing: [
            { name: 'sox', why: 'Voice cloning runs the sox program.', optional: false, services: ['voice'] },
            { name: 'libusb', why: 'XVF3800 controls.', optional: true, services: ['ambient'] },
          ],
          commands: ['sudo apt-get install -y sox libusb-1.0-0'],
        },
      }),
    );
    renderPage(<Machine />);
    const card = await screen.findByRole('region', { name: 'Needs installing' });
    expect(within(card).getByRole('listitem', { name: 'libusb' })).toHaveTextContent(
      'libusb package, optional. XVF3800 controls.',
    );
    expect(within(card).getAllByRole('button', { name: 'Copy command' })).toHaveLength(1);
    expect(within(card).getByText('sudo apt-get install -y sox libusb-1.0-0')).toBeInTheDocument();
  });

  it('is left out when nothing is missing, and says when the packages could not be checked', async () => {
    serve(stats({ system: { missing: [], commands: [] } }));
    const { unmount } = renderPage(<Machine />);
    await screen.findByText('Running smoothly');
    expect(screen.queryByRole('region', { name: 'Needs installing' })).not.toBeInTheDocument();
    unmount();
    serve(stats({ system: { missing: [], commands: [], error: 'sox: say why it is needed' } }));
    renderPage(<Machine />);
    expect(await screen.findByRole('region', { name: 'Needs installing' })).toHaveTextContent(
      "Couldn't check the packages: sox: say why it is needed",
    );
  });
});
