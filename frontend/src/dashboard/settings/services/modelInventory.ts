// Machine's model inventory joins application loader reports and Ollama without assigning model bytes to a process.
import type { ConfiguredModel, RuntimeModel, Stats } from '../../sources/api';
import { gb } from '../../sources/format';
import { size } from './machineMeters';

export interface InventoryModel extends RuntimeModel {
  params?: string | null;
  ctx?: number | null;
  allocation?: number;
}

export interface InventoryGroup {
  key: string;
  name: string;
  pid: number | null;
  rss: number | null;
  gpu: number | null;
  allocation: number | null;
  ollama: boolean;
  models: InventoryModel[];
}

export interface ModelInventory {
  reported: boolean;
  groups: InventoryGroup[];
  configured: ConfiguredModel[];
  errors: string[];
  loaded: number;
}

const measured = (n: number | null | undefined): number | null =>
  n != null && Number.isFinite(n) && n >= 0 ? n : null;

/** Unknown allocation is not zero; totals are partial when any contributor has no measurement. */
function total(values: (number | null | undefined)[]): number | null {
  return values.length && values.every(n => measured(n) != null)
    ? values.reduce<number>((sum, n) => sum + (n ?? 0), 0)
    : null;
}

export function deviceLabel(device: string | null | undefined): string {
  const value = device?.trim();
  if (!value || /^(unknown|auto)$/i.test(value)) return 'Device not reported';
  if (/^(mps|mlx|apple gpu|metal)$/i.test(value)) return 'Apple GPU';
  if (/^(cuda(?::\d+)?|gpu)$/i.test(value)) return 'GPU';
  if (/^cpu$/i.test(value)) return 'CPU';
  return value;
}

export function modelInventory(stats: Stats): ModelInventory {
  const runtime = stats.model_runtime;
  const groups: InventoryGroup[] = (runtime?.processes ?? [])
    .filter(p => p.models.length > 0)
    .map(p => ({
      key: `process-${p.pid}`,
      name: p.service,
      pid: p.pid,
      rss: measured(p.rss_bytes),
      gpu: measured(p.gpu_bytes),
      allocation: null,
      ollama: false,
      models: p.models.map(m => ({ ...m, purpose: m.purpose || 'Purpose not reported' })),
    }));
  const ollama = stats.ollama?.models ?? [];
  if (ollama.length) {
    groups.push({
      key: 'ollama',
      name: 'Ollama',
      pid: null,
      rss: null,
      gpu: total(ollama.map(m => m.size_vram)),
      allocation: total(ollama.map(m => m.size)),
      ollama: true,
      models: ollama.map(m => {
        const vram = measured(m.size_vram);
        return {
          key: `ollama:${m.name}`,
          name: m.name,
          purpose: m.purpose || 'Purpose not reported',
          device: m.device || (vram == null ? 'unknown' : vram === 0 ? 'CPU' : vram >= m.size ? 'GPU' : 'GPU and CPU'),
          since: m.since ?? null,
          params: m.params,
          ctx: m.ctx,
          allocation: m.size,
        };
      }),
    });
  }
  const loadedKeys = new Set(groups.flatMap(g => g.models.map(m => `${m.key}\n${m.name}`)));
  const ollamaNames = new Set(ollama.map(m => m.name));
  return {
    reported: runtime != null,
    groups,
    configured: (runtime?.configured ?? []).filter(
      m => !loadedKeys.has(`${m.key}\n${m.name}`) && !ollamaNames.has(m.name),
    ),
    errors: runtime?.errors ?? [],
    loaded: groups.reduce((n, g) => n + g.models.length, 0),
  };
}

/** Sum each application's process counters once, keeping RSS, VRAM, and Ollama allocations separate. */
export function inventoryMemory(inventory: ModelInventory): string {
  const processes = inventory.groups.filter(g => !g.ollama);
  const parts: string[] = [];
  if (processes.length) {
    const rss = total(processes.map(g => g.rss));
    const gpu = total(processes.map(g => g.gpu));
    parts.push(rss == null ? 'Process RSS unavailable' : `${size(rss)} process RSS`);
    parts.push(gpu == null ? 'Process VRAM unavailable' : `${size(gpu)} process VRAM`);
  }
  const ollama = inventory.groups.find(g => g.ollama);
  if (ollama)
    parts.push(
      ollama.allocation == null
        ? 'Ollama allocations unavailable'
        : `${gb(ollama.allocation)} Ollama model allocations`,
    );
  return parts.join('. ');
}
