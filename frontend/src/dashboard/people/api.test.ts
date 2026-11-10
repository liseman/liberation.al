import { QueryClient } from '@tanstack/react-query';
import { expect, it } from 'vitest';
import { cardKey, PEOPLE_KEY, profileKey, tookView } from './api';

it("takes the reply's list and marks the profiles and cards out of date, not the list", () => {
  const client = new QueryClient();
  client.setQueryData(profileKey('Dana'), {});
  client.setQueryData(cardKey('Dana'), {});
  tookView(client, { people: [] });
  expect(client.getQueryData(PEOPLE_KEY)).toEqual(expect.objectContaining({ people: [] }));
  const stale = (k: readonly unknown[]) => client.getQueryState(k)?.isInvalidated;
  expect([stale(profileKey('Dana')), stale(cardKey('Dana')), stale(PEOPLE_KEY)]).toEqual([true, true, false]);
});
