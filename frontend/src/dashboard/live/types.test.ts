import { expect, it } from 'vitest';
import { oneLine, someLines } from './types';

it('names lines by their keys, a time standing for a line with no device (#247)', () => {
  expect(someLines(['mic-a@5.000', 6])).toEqual({ lines: ['mic-a@5.000', '6.000'] });
  expect([oneLine('mic-a@5.000'), oneLine(5)]).toEqual([{ line: 'mic-a@5.000' }, { ts_start: 5 }]);
});
