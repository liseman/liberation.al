import { act, fireEvent, renderHook, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { Bars, Heatmap, placeTip, useNewestInView } from './charts';

/** A chart's scrolling frame: jsdom has no layout, so its widths are given. */
function frame(scrollWidth: number, clientWidth: number): { current: HTMLElement } {
  const el = document.createElement('figure');
  Object.defineProperty(el, 'scrollWidth', { value: scrollWidth });
  Object.defineProperty(el, 'clientWidth', { value: clientWidth });
  return { current: el };
}

it('opens a chart wider than its frame at the newest end', () => {
  const ref = frame(1060, 316);
  renderHook(() => useNewestInView(ref, 1060));
  expect(ref.current.scrollLeft).toBe(744);
});

it('leaves a chart that fits where it is', () => {
  const ref = frame(316, 316);
  renderHook(() => useNewestInView(ref, 316));
  expect(ref.current.scrollLeft).toBe(0);
});

it("keeps the reader's scrolling until the drawing's width changes", () => {
  const ref = frame(1060, 316);
  const { rerender } = renderHook(({ drawn }) => useNewestInView(ref, drawn), { initialProps: { drawn: 1060 } });
  ref.current.scrollLeft = 100; // the reader scrolled back
  rerender({ drawn: 1060 });
  expect(ref.current.scrollLeft).toBe(100);
  rerender({ drawn: 1100 });
  expect(ref.current.scrollLeft).toBe(744);
});

// Chart tooltips float over the page: placed in the window, not in the chart's clipping card.
it.each([
  ['below and to the right of the pointer', 100, 100, { left: 112, top: 112 }],
  ['to its left near the right edge', 950, 100, { left: 738, top: 112 }],
  ['above it near the bottom edge', 100, 700, { left: 112, top: 628 }],
])('places a tooltip %s', (_where, x, y, at) => {
  expect(placeTip(x, y, 200, 60, 1000, 768)).toEqual(at);
});

it('keeps a tooltip bigger than the room on either side inside the window', () => {
  expect(placeTip(500, 400, 980, 60, 1000, 768)).toEqual({ left: 12, top: 412 }); // 1000 - 8 margin - 980
});

it("shows a bar's tooltip on the page's body, outside the chart's card, and takes it down on a scroll", () => {
  const points = [{ key: 'a', value: 3, label: 'Monday: 3 GB' }];
  const { container } = renderPage(
    <Bars label="Storage" points={points} ticks={new Set()} tickLabel={() => ''} xTitle="day" yTitle="GB" />,
  );
  const bar = [...container.querySelectorAll('svg g')].find(g => g.querySelector('rect'));
  expect(bar).toBeDefined();
  fireEvent.mouseMove(bar as Element, { clientX: 40, clientY: 50 });
  const tip = screen.getByText('Monday: 3 GB', { selector: 'div' });
  expect(tip.parentElement).toBe(document.body);
  expect(container.contains(tip)).toBe(false);
  act(() => {
    window.dispatchEvent(new Event('scroll'));
  });
  expect(screen.queryByText('Monday: 3 GB', { selector: 'div' })).toBeNull();
});

it('keeps the x-axis caption out of the scrolling plot, so the sticky labels never cover its start', () => {
  renderPage(
    <Heatmap
      label="Lines by person and hour"
      rows={['Alice', 'Dana']}
      cols={['12 am', '1 am']}
      colTicks={new Set([0])}
      cell={() => ({ value: 1, label: 'one line' })}
      xTitle="Hour of day (links open the latest day)"
      yTitle=""
    />,
  );
  const caption = screen.getByText('Hour of day (links open the latest day)');
  expect(caption.tagName).toBe('FIGCAPTION');
  expect(caption.closest('svg')).toBeNull();
});

// A pointer move changes only where the tooltip sits: the grid under it is the same cells, scales, and tab stops.
it("doesn't rebuild the heatmap's grid for a pointer move, only its tooltip", () => {
  const cell = vi.fn((_row: number, _col: number) => ({ value: 1, label: 'one line' }));
  const { container } = renderPage(
    <Heatmap
      label="Lines by person and hour"
      rows={['Alice']}
      cols={['12 am', '1 am']}
      colTicks={new Set([0])}
      cell={cell}
      xTitle="Hour"
      yTitle=""
    />,
  );
  const cellsBuilt = cell.mock.calls.length;
  expect(cellsBuilt).toBeGreaterThan(0);
  const marks = [...container.querySelectorAll('svg g')].filter(g => g.querySelector('rect'));
  expect(marks).toHaveLength(2);
  fireEvent.mouseMove(marks[0] as Element, { clientX: 10, clientY: 12 });
  fireEvent.mouseMove(marks[1] as Element, { clientX: 14, clientY: 16 });
  expect(screen.getAllByText('one line', { selector: 'div' })).toHaveLength(1);
  expect(cell.mock.calls.length).toBe(cellsBuilt);
});

it("doesn't rebuild the bars for a pointer move, only their tooltip", () => {
  const tickLabel = vi.fn(() => 'Mon');
  const points = [
    { key: 'a', value: 3, label: 'Monday: 3 GB' },
    { key: 'b', value: 5, label: 'Tuesday: 5 GB' },
  ];
  const { container } = renderPage(
    <Bars label="Storage" points={points} ticks={new Set([0, 1])} tickLabel={tickLabel} xTitle="day" yTitle="GB" />,
  );
  const labels = tickLabel.mock.calls.length;
  expect(labels).toBeGreaterThan(0);
  const marks = [...container.querySelectorAll('svg g')].filter(g => g.querySelector('rect'));
  fireEvent.mouseMove(marks[0] as Element, { clientX: 10, clientY: 12 });
  expect(screen.getAllByText('Monday: 3 GB', { selector: 'div' })).toHaveLength(1);
  expect(tickLabel.mock.calls.length).toBe(labels);
});
