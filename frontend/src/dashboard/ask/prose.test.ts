import { expect, it } from 'vitest';
import { blocks } from './prose';

it('reads a pipe table with its alignment, and goes back to prose after it', () => {
  const md = [
    '## Counts',
    '| What | Today | Usual |',
    '| --- | ---: | :---: |',
    '| Barks | 14 | 9 to 20 |',
    '| Door | 11 | 8 to 14 |',
    '',
    'After.',
  ].join('\n');
  expect(blocks(md)).toEqual([
    { kind: 'h5', text: 'Counts' },
    {
      kind: 'table',
      head: ['What', 'Today', 'Usual'],
      align: [undefined, 'right', 'center'],
      rows: [
        ['Barks', '14', '9 to 20'],
        ['Door', '11', '8 to 14'],
      ],
    },
    { kind: 'p', text: 'After.' },
  ]);
});

it('leaves a line with a pipe but no delimiter row under it as prose', () => {
  expect(blocks('a | b\nc')).toEqual([{ kind: 'p', text: 'a | b c' }]);
});
