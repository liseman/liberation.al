import { render } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.doUnmock('./preview');
  vi.doUnmock('../lib/staleChunk');
  vi.resetModules();
});

it('reloads for the new build when the preview chunk is gone, instead of saying unavailable', async () => {
  const reload = vi.fn(() => true);
  vi.doMock('../lib/staleChunk', () => ({ reloadForNewBuild: reload }));
  vi.doMock('./preview', () => {
    throw new TypeError('Failed to fetch dynamically imported module: /static/app/assets/preview-old.js');
  });
  const { PreviewCam } = await import('./PreviewCam');
  const view = render(<PreviewCam cam="cam0" alt="cam0" />);
  await vi.waitFor(() => expect(reload).toHaveBeenCalled());
  expect(view.container.textContent).not.toContain('Preview unavailable');
});
