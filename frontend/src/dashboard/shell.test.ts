import { afterEach, expect, it, vi } from 'vitest';
import { setActiveCameras, shell } from './shell';
import { camStatus, camText } from './status';

afterEach(() => vi.unstubAllGlobals());

it('keeps exact native camera names in the shell and Live status', () => {
  vi.stubGlobal('NAMES', { cams: ['front-door', 'CAM0', '0'] });
  const cams = shell().cams;
  expect(cams).toEqual(['front-door', 'CAM0', '0']);
  const status = camStatus([{ cam: 'front-door', ts: 100 }], cams, 100);
  expect(camText(status, cams)).toBe('1 of 3 cameras on (CAM0, 0 quiet)');
});

it('keeps an explicitly camera-free manifest off without inventing cam0', () => {
  vi.stubGlobal('NAMES', { cams: [] });
  expect(shell().cams).toEqual([]);
  const status = camStatus([], shell().cams, 100);
  expect(camText(status, shell().cams)).toBe('Cameras off');
});

it('uses the legacy shell default only when camera configuration is omitted', () => {
  vi.stubGlobal('NAMES', {});
  expect(shell().cams).toEqual(['cam0']);
});

it('keeps disabled camera recordings available while a successful Use save updates Live only', () => {
  vi.stubGlobal('NAMES', { cams: ['entryway', 'desk'], activeCams: ['desk'] });
  expect(shell().cams).toEqual(['entryway', 'desk']);
  expect(shell().activeCams).toEqual(['desk']);
  setActiveCameras(['entryway', 'desk']);
  expect(shell().activeCams).toEqual(['entryway', 'desk']);
  expect(shell().cams).toEqual(['entryway', 'desk']);
});

it('falls back to the full roster when an older server omits active cameras', () => {
  vi.stubGlobal('NAMES', { cams: ['entryway'] });
  expect(shell().activeCams).toEqual(['entryway']);
});
