import { expect, it } from 'vitest';
import { reportTitle } from './title';

it('puts a saved title in sentence case and names the walk-by report plainly', () => {
  expect(reportTitle({ arg: 'event:crossing', title: 'crossing over time' })).toBe('Walked past, day by day');
  expect(reportTitle({ arg: 'event:crossing', title: 'crossing, day by day' })).toBe('Walked past, day by day');
  expect(reportTitle({ arg: 'event:door', title: 'door over time' })).toBe('Door over time');
  expect(reportTitle({ arg: null, title: "Who's around" })).toBe("Who's around");
});
