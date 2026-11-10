import { describe, expect, it } from 'vitest';
import { stats } from '../../sources/fixtures';
import { modelsWhere, platformLine, size, verdict } from './machineMeters';

const GB = 1073741824;

describe('platformLine', () => {
  it('says the system, the chip, the cores, and the memory', () => {
    expect(platformLine('linux-x86_64', stats({ cpu_count: 20, mem_total: 123.2 * GB }))).toBe(
      'Linux on x86-64, with 20 processor cores and 123.2 GB of memory.',
    );
    expect(platformLine('darwin-arm64', stats())).toBe(
      'macOS on Apple Silicon, with 8 processor cores and 16.0 GB of memory.',
    );
  });

  it('keeps a platform it has no words for, and does without one', () => {
    expect(platformLine('plan9-mips', stats())).toBe('plan9 on mips, with 8 processor cores and 16.0 GB of memory.');
    expect(platformLine(undefined, stats())).toBe('This machine, with 8 processor cores and 16.0 GB of memory.');
  });
});

describe('verdict', () => {
  const at = (cpu: number, memPct: number, swap = 0) =>
    verdict(stats({ cpu, mem_total: 100 * GB, mem_used: memPct * GB, swap_used: swap }));

  it('is green with room to spare', () => {
    expect(at(12, 58)).toEqual({
      tone: 'ok',
      title: 'Running smoothly',
      text: 'The processor is at 12%, and 58% of memory is in use.',
    });
  });

  it('is amber from 80% of the processor, or more than 1 GB in swap', () => {
    expect(at(79.4, 50)).toMatchObject({ tone: 'ok' });
    expect(at(80, 50)).toEqual({
      tone: 'warn',
      title: 'Busy',
      text: 'The processor is at 80%. Things may respond slowly.',
    });
    expect(at(10, 50, GB)).toMatchObject({ tone: 'ok' }); // 1 GB exactly is not over
    expect(at(86, 50, 8 * GB).text).toBe(
      'The processor is at 86%, and 8.0 GB of memory has moved to swap. Things may respond slowly.',
    );
    expect(at(10, 50, 2 * GB).text).toBe('2.0 GB of memory has moved to swap. Things may respond slowly.');
  });

  it('is red from 90% of memory, whatever else', () => {
    expect(at(95, 90, 8 * GB)).toEqual({
      tone: 'bad',
      title: 'Low on memory',
      text: 'Memory is 90% full. Programs may slow down, or be stopped to make room.',
    });
  });
});

describe('modelsWhere', () => {
  const m = (size: number, size_vram: number | null) => ({ name: 'm', size, size_vram });
  it('says where the loaded models sit', () => {
    expect(modelsWhere([m(3 * GB, 3 * GB), m(GB, GB)])).toBe('all in GPU memory');
    expect(modelsWhere([m(3 * GB, 0), m(GB, null)])).toBe('in main memory');
    expect(modelsWhere([m(3 * GB, GB), m(GB, GB)])).toBe('2.0 GB of it in GPU memory');
  });
});

describe('size', () => {
  it('gives memory in MB under 1 GB, else GB', () => {
    expect(size(150 * 1048576)).toBe('150 MB');
    expect(size(GB)).toBe('1.0 GB');
  });
});
