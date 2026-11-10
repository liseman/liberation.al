import { expect, it } from 'vitest';
import { lineClip } from './api';

it('plays a line by its key when it has one, else by its time', () => {
  expect([lineClip('mic-a@1793000000.123'), lineClip(1793000000.123)]).toEqual([
    '/clip?line=mic-a%401793000000.123',
    '/clip?ts_start=1793000000.123',
  ]);
});
