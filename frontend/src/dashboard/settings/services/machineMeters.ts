// This machine's numbers in words: the platform line, the verdict at the top, the models' size and where they
// sit.
import type { LoadedModel, Stats } from '../../sources/api';
import { gb } from '../../sources/format';
import type { Tone } from './model';

const GB = 1073741824;

/** Memory in a few characters: MB under 1 GB, else GB. */
export function size(bytes: number): string {
  return bytes < GB ? `${Math.round(bytes / 1048576)} MB` : gb(bytes);
}

const SYSTEMS: Record<string, string> = { linux: 'Linux', darwin: 'macOS' };
const CHIPS: Record<string, string> = {
  x86_64: 'x86-64',
  arm64: 'Apple Silicon',
  aarch64: 'ARM',
};

/** "Linux on x86-64, with 20 processor cores and 123.2 GB of memory." from the platform key and /stats. */
export function platformLine(platform: string | undefined, s: Stats): string {
  const [os = '', chip = ''] = (platform ?? '').split('-');
  const system = SYSTEMS[os] ?? os;
  const where = !platform ? 'This machine' : chip ? `${system} on ${CHIPS[chip] ?? chip}` : system;
  return `${where}, with ${s.cpu_count ?? '?'} processor cores and ${gb(s.mem_total ?? 0)} of memory.`;
}

export const BUSY_CPU = 80; // percent of the machine: amber from here
export const BUSY_SWAP = GB; // amber with more than this in swap
export const FULL_MEMORY = 90; // percent of memory in use: red from here

export interface Verdict {
  tone: Tone;
  title: string;
  text: string;
}

const pct = (n: number) => `${Math.round(n)}%`;
const memoryPct = (s: Stats) => (s.mem_total ? (100 * (s.mem_used ?? 0)) / s.mem_total : 0);

/** One sentence on how the machine is doing: red when memory is nearly full, amber when the processor is busy or
 *  memory has moved to swap, else green. */
export function verdict(s: Stats): Verdict {
  const cpu = s.cpu ?? 0;
  const mem = memoryPct(s);
  const swap = s.swap_used ?? 0;
  if (mem >= FULL_MEMORY)
    return {
      tone: 'bad',
      title: 'Low on memory',
      text: `Memory is ${pct(mem)} full. Programs may slow down, or be stopped to make room.`,
    };
  const busy = [
    cpu >= BUSY_CPU && `the processor is at ${pct(cpu)}`,
    swap > BUSY_SWAP && `${gb(swap)} of memory has moved to swap`,
  ].filter((x): x is string => !!x);
  if (busy.length) {
    const said = busy.join(', and ');
    return {
      tone: 'warn',
      title: 'Busy',
      text: `${said[0]?.toUpperCase()}${said.slice(1)}. Things may respond slowly.`,
    };
  }
  return {
    tone: 'ok',
    title: 'Running smoothly',
    text: `The processor is at ${pct(cpu)}, and ${pct(mem)} of memory is in use.`,
  };
}

/** Where the loaded models sit: all in GPU memory, all in main memory, or split. */
export function modelsWhere(models: readonly LoadedModel[]): string {
  const total = models.reduce((n, m) => n + m.size, 0);
  const vram = models.reduce((n, m) => n + (m.size_vram ?? 0), 0);
  if (!vram) return 'in main memory';
  if (vram >= total) return 'all in GPU memory';
  return `${gb(vram)} of it in GPU memory`;
}
