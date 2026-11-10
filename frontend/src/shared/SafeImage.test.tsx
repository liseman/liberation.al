import { act, fireEvent, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { renderPage } from '../test/render';
import { Providers } from './Providers';
import { RETRY_MS, SafeImage } from './SafeImage';

it('replaces a failed image with an accessible placeholder and retries only a new URL', () => {
  const view = renderPage(<SafeImage src="/missing.jpg" alt="Alice" width={160} height={160} />);
  fireEvent.error(screen.getByRole('img', { name: 'Alice' }));
  expect(screen.getByRole('img', { name: 'Alice: image unavailable' })).not.toHaveAttribute('src');
  expect(document.querySelector('img')).toBeNull();
  view.rerender(
    <Providers client={view.client} test>
      <SafeImage src="/new.jpg" alt="Alice" />
    </Providers>,
  );
  expect(screen.getByRole('img', { name: 'Alice' })).toHaveAttribute('src', '/new.jpg');
});

it('keeps decorative missing images hidden from assistive technology', () => {
  renderPage(<SafeImage src="/missing.jpg" alt="" />);
  fireEvent.error(screen.getByAltText(''));
  expect(screen.queryByRole('img')).toBeNull();
  expect(document.querySelector('img')).toBeNull();
});

it('omits an unavailable image when the caller explicitly requests a null fallback', () => {
  renderPage(<SafeImage src="/missing.jpg" alt="Alice" fallback={null} />);
  fireEvent.error(screen.getByRole('img', { name: 'Alice' }));
  expect(document.querySelector('[data-safe-image]')).toBeNull();
  expect(screen.queryByRole('img')).toBeNull();
});

it('tries a failed image again after a wait, as many times as asked, then shows the placeholder (#2018)', () => {
  vi.useFakeTimers();
  try {
    renderPage(<SafeImage src="/faces/shot/a.jpg" alt="Alice" retries={2} />);
    fireEvent.error(screen.getByRole('img', { name: 'Alice' }));
    expect(screen.getByRole('img', { name: 'Alice' })).toHaveAttribute('src', '/faces/shot/a.jpg'); // waiting
    act(() => void vi.advanceTimersByTime(RETRY_MS[0] ?? 0));
    expect(screen.getByRole('img', { name: 'Alice' })).toHaveAttribute('src', '/faces/shot/a.jpg?retry=1');
    fireEvent.error(screen.getByRole('img', { name: 'Alice' }));
    act(() => void vi.advanceTimersByTime(RETRY_MS[1] ?? 0));
    expect(screen.getByRole('img', { name: 'Alice' })).toHaveAttribute('src', '/faces/shot/a.jpg?retry=2');
    fireEvent.error(screen.getByRole('img', { name: 'Alice' }));
    expect(screen.getByRole('img', { name: 'Alice: image unavailable' })).toBeInTheDocument();
  } finally {
    vi.useRealTimers();
  }
});

// #2049: while a picture loads, and through its retries, a skeleton holds its place at its final size.
const skeleton = () => document.querySelector('[data-safe-image]');

it("shows a skeleton in the image's own box until the picture arrives", () => {
  renderPage(<SafeImage src="/faces/shot/b.jpg" alt="Bea" className="photo" width={152} height={152} />);
  const box = skeleton();
  expect(box).toHaveAttribute('data-visible', 'true');
  expect(box).toHaveClass('photo');
  expect(box).toHaveStyle({ width: '152px', height: '152px' });
  fireEvent.load(screen.getByRole('img', { name: 'Bea' }));
  expect(skeleton()).not.toHaveAttribute('data-visible', 'true');
});

it('keeps the skeleton up while a failed picture waits for its retry', () => {
  vi.useFakeTimers();
  try {
    renderPage(<SafeImage src="/faces/shot/c.jpg" alt="Cal" retries={1} />);
    fireEvent.error(screen.getByRole('img', { name: 'Cal' }));
    expect(skeleton()).toHaveAttribute('data-visible', 'true');
    act(() => void vi.advanceTimersByTime(RETRY_MS[0] ?? 0));
    expect(skeleton()).toHaveAttribute('data-visible', 'true');
    fireEvent.load(screen.getByRole('img', { name: 'Cal' }));
    expect(skeleton()).not.toHaveAttribute('data-visible', 'true');
  } finally {
    vi.useRealTimers();
  }
});
