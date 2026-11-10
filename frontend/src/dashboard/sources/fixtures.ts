// A /stats snapshot in the server's full shape, for the tests.
import type { Stats } from './api';

export const stats = (over: Partial<Stats> = {}): Stats => ({
  cpu: 12.6,
  cpu_count: 8,
  load: [1.234, 0.5, 0.25],
  mem_total: 17179869184, // 16 GiB
  mem_used: 5368709120, // 5 GiB
  swap_used: 536870912, // 0.5 GiB
  ollama: { models: [{ name: 'qwen3:8b', size: 6442450944 }], error: null },
  procs: [
    { pid: 101, role: 'ambient', cpu: 23.4, rss: 314572800 }, // 300 MiB
    { pid: 102, role: 'vision', cpu: 150, rss: 1073741824 },
  ],
  ...over,
});
