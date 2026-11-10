import { queryOptions, useQuery } from '@tanstack/react-query';
import { getPeople, PEOPLE_KEY } from '../people/api';

/** Naming suggestions share the roster's request and cache instead of loading the full view a second time. */
export const knownPeopleQuery = () =>
  queryOptions({
    queryKey: PEOPLE_KEY,
    queryFn: getPeople,
    select: view => view.people.map(p => p.name).filter(Boolean),
    staleTime: 30_000,
  });

/** Everyone known by name (empty until it loads, or if it can't). */
export function useKnownPeople(): string[] {
  return useQuery(knownPeopleQuery()).data ?? [];
}
