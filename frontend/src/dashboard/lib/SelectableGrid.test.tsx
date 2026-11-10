import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { arrowStep, SelectableGrid } from './SelectableGrid';

function Grid({ mark }: { mark?: 'badge' | 'corner' }) {
  const [on, setOn] = useState<ReadonlySet<string>>(() => new Set(['b']));
  return (
    <SelectableGrid
      label="Pictures"
      mark={mark}
      tone={mark === 'corner' ? 'keep' : undefined}
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

it('marks only the chosen tiles with the small corner mark, and a click on the mark toggles its tile', async () => {
  const user = userEvent.setup();
  renderPage(<Grid mark="corner" />);
  expect(screen.getByRole('group', { name: 'Pictures' })).toHaveAttribute('data-mark', 'corner');
  const marked = () => ['a', 'b', 'c'].filter(k => tile(k).querySelector('[data-mark]'));
  expect(marked()).toEqual(['b']); // a tile that's off has no mark at all
  expect(tile('b').querySelector('[role="checkbox"], input')).toBeNull(); // not the full check box
  // The mark is inside its tile's button and takes no clicks of its own: the whole picture is the target.
  await user.click(tile('b').querySelector('[data-mark]') as HTMLElement);
  expect(tile('b')).toHaveAttribute('aria-pressed', 'false');
  expect(marked()).toEqual([]);
  await user.click(tile('a'));
  expect(marked()).toEqual(['a']);
});
