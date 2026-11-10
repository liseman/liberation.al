import { act, renderHook, waitFor } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import type { ReactNode } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import { newQueryClient, Providers } from '../../../shared/Providers';
import { server } from '../../../test/server';
import { CONFIG_KEY, type ConfigView } from '../model';
import { useDeviceIssues } from './issues';

const config = (places: string[]): ConfigView =>
  ({
    sections: [
      {
        name: 'vision',
        description: '',
        settings: [
          {
            key: 'vision.camera_names',
            value: places,
            default: [],
            type: 'list',
            secret: false,
            description: '',
            section: 'devices',
          },
        ],
      },
    ],
  }) as unknown as ConfigView;

beforeEach(() => {
  vi.stubGlobal('NAMES', { cams: ['cam0', 'cam1', 'cam2'], activeCams: ['cam0', 'cam1'], views: {} });
  // Vision and the microphones have nothing to report, so only the names are left to badge
  server.use(
    http.get('/vision', () => HttpResponse.json({})),
    http.get('/ambient', () => HttpResponse.json({})),
  );
});

function render() {
  const client = newQueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => (
    <Providers client={client} test>
      {children}
    </Providers>
  );
  return { client, ...renderHook(() => useDeviceIssues(), { wrapper }) };
}

it('asks to name the cameras in use that have no name (#2585)', () => {
  const { result } = render();
  expect(result.current).toEqual({ text: 'Name 2 cameras', tone: 'dim' });
});

it('counts a name saved in Settings at once, without waiting for a page load', async () => {
  const { client, result } = render();
  act(() => client.setQueryData(CONFIG_KEY, config(['cam0: front'])));
  await waitFor(() => expect(result.current).toEqual({ text: 'Name 1 camera', tone: 'dim' }));
  act(() => client.setQueryData(CONFIG_KEY, config(['cam0: front', 'cam1: desk'])));
  await waitFor(() => expect(result.current).toBeNull());
});
