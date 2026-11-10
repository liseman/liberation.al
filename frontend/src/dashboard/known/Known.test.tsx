import { act, render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { countLineChars, setSessionCount } from '../lib/loading';
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

it('sizes a first-read skeleton to one digit and the count words', () => {
  expect(countLineChars('to name')).toBe(9);
  renderPage(<Waiting n={undefined} what="to name" loading />);
  expect(screen.getByTestId('count-skeleton')).toBeInTheDocument();
});

it('shows the last count while a tab reloads after a first read', () => {
  setSessionCount('test:tab', 0);
  renderPage(
    <div data-testid="tab">
      <Waiting cacheKey="test:tab" n={undefined} what="to name" loading />
    </div>,
  );
  expect(screen.queryByTestId('count-skeleton')).not.toBeInTheDocument();
  expect(screen.getByTestId('tab')).toHaveTextContent(/^$/);
});

it('marks a held count as an earlier read while the tab reloads, then shows the new one as current', () => {
  setSessionCount('test:held', 7);
  const { rerender } = renderPage(<Waiting cacheKey="test:held" n={undefined} what="to name" loading />);
  const held = screen.getByText('7 to name');
  expect(held).toHaveAttribute('data-stale');
  expect(held).toHaveAttribute('aria-busy', 'true');
  rerender(<Waiting cacheKey="test:held" n={9} what="to name" />);
  const fresh = screen.getByText('9 to name');
  expect(fresh).not.toHaveAttribute('data-stale');
  expect(fresh).not.toHaveAttribute('aria-busy');
});

it.each([
  ['loading', { n: undefined, loading: true }, null],
  ['loaded', { n: 12, loading: false }, '12 to name'],
  ['zero', { n: 0, loading: false }, ''],
] as const)('a tab count while %s: a skeleton until its read answers, then its words', (_, props, text) => {
  renderPage(
    <div data-testid="tab">
      <Waiting {...props} what="to name" />
    </div>,
  );
  const skeleton = screen.queryByTestId('count-skeleton');
  if (text === null) expect(skeleton).toBeInTheDocument();
  else {
    expect(skeleton).not.toBeInTheDocument();
    expect(screen.getByTestId('tab')).toHaveTextContent(text === '' ? /^$/ : text);
  }
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
