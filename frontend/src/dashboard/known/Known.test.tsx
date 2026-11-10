import { act, render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { useRowEdges, Waiting } from './Known';

it('says what a tab badge counts, and shows nothing when none waits', () => {
  renderPage(
    <>
      <Waiting n={201} what="to name" />
      <Waiting n={0} what="to check" />
    </>,
  );
  expect(screen.getByText('201 to name')).toBeInTheDocument();
  expect(screen.queryByText(/to check/)).not.toBeInTheDocument();
});

it.each([
  ['still loading', undefined],
  ['with none waiting', 0],
  ['with a count', 7],
])('keeps a tab its count line %s, so the tab row never moves when the count arrives', (_, n) => {
  renderPage(
    <div data-testid="tab">
      <Waiting n={n} what="to check" />
    </div>,
  );
  // jsdom has no layout: the line's fixed height comes from CSS, so here it's enough that the line is always there.
  const line = screen.getByTestId('tab').children;
  expect(line).toHaveLength(1);
  expect(line[0]?.tagName).toBe('SPAN');
  expect(line[0]).toHaveTextContent(n ? /^7 to check$/ : /^$/);
});

/** A row of tabs 500px wide in 300px, as on a phone; jsdom has no layout, so the widths are set on the element. */
function Row() {
  const { list, edges } = useRowEdges('people');
  return (
    <div
      ref={el => {
        if (el && !('scrollWidth' in Object.getOwnPropertyDescriptors(el))) {
          let left = 0;
          Object.defineProperties(el, {
            clientWidth: { value: 300 },
            scrollWidth: { value: 500 },
            scrollLeft: { get: () => left, set: (v: number) => (left = v) },
          });
        }
        list.current = el;
      }}
      data-testid="row"
      data-start={edges.start || undefined}
      data-end={edges.end || undefined}
    />
  );
}

it('marks the ends of the tab row that have more tabs past them, as it scrolls', () => {
  render(<Row />);
  const row = screen.getByTestId('row');
  expect(row).toHaveAttribute('data-end');
  expect(row).not.toHaveAttribute('data-start');
  act(() => {
    row.scrollLeft = 100;
    row.dispatchEvent(new Event('scroll'));
  });
  expect(row).toHaveAttribute('data-start');
  expect(row).toHaveAttribute('data-end');
  act(() => {
    row.scrollLeft = 200;
    row.dispatchEvent(new Event('scroll'));
  });
  expect(row).toHaveAttribute('data-start');
  expect(row).not.toHaveAttribute('data-end');
});
