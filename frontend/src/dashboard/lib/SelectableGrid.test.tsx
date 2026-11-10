import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { arrowStep, SelectableGrid } from './SelectableGrid';

function Grid() {
  const [on, setOn] = useState<ReadonlySet<string>>(() => new Set(['b']));
  return (
    <SelectableGrid
      label="Pictures"
      tiles={['a', 'b', 'c'].map(k => ({ key: k, label: `Picture ${k}`, content: k }))}
      isOn={k => on.has(k)}
      onToggle={k =>
        setOn(have => {
          const next = new Set(have);
          if (!next.delete(k)) next.add(k);
          return next;
        })
      }
    />
  );
}

const tile = (k: string) => screen.getByRole('button', { name: `Picture ${k}` });

it('toggles a tile by click or Space and moves between tiles with the arrows', async () => {
  const user = userEvent.setup();
  renderPage(<Grid />);
  expect(screen.getByRole('group', { name: 'Pictures' })).toBeInTheDocument();
  expect(tile('b')).toHaveAttribute('aria-pressed', 'true');
  await user.click(tile('a'));
  expect(tile('a')).toHaveAttribute('aria-pressed', 'true');
  await user.keyboard('{ArrowRight}');
  expect(tile('b')).toHaveFocus();
  await user.keyboard(' ');
  expect(tile('b')).toHaveAttribute('aria-pressed', 'false');
  await user.keyboard('{ArrowRight}{ArrowRight}'); // stops at the last tile
  expect(tile('c')).toHaveFocus();
  await user.keyboard('{ArrowLeft}');
  expect(tile('b')).toHaveFocus();
});

it.each([
  ['ArrowRight', 4, 1],
  ['ArrowLeft', 4, -1],
  ['ArrowDown', 4, 4],
  ['ArrowUp', 4, -4],
  ['ArrowDown', 0, 1],
  ['Enter', 4, 0],
])('steps %s with %i columns by %i', (key, columns, step) => {
  expect(arrowStep(key, columns)).toBe(step);
});
