import { screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { renderPage } from '../../../test/render';
import { PetRoster } from './PetRoster';

const roster = (waiting: number | null, waitingFailed = false) =>
  renderPage(
    <PetRoster
      pets={[]}
      seen={{}}
      waiting={waiting}
      waitingFailed={waitingFailed}
      loaded
      error={null}
      retrying={false}
      onRetry={() => {}}
      onShowAll={() => {}}
      sel={null}
      onOpen={() => {}}
      onNew={() => {}}
    />,
  );

it('holds the waiting line as a skeleton while the sightings count loads', () => {
  roster(null);
  expect(screen.getByTestId('count-skeleton')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Sightings to name · Loading the count' })).toBeInTheDocument();
});

it.each([
  [3, '3 animals need a name'],
  [1, '1 animal needs a name'],
  [0, 'Sightings to name'],
])('shows the count once read: %i waiting reads "%s"', (n, words) => {
  roster(n);
  expect(screen.queryByTestId('count-skeleton')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: words })).toBeInTheDocument();
});

it('says the count is unavailable when its read failed, never a skeleton for good', () => {
  roster(null, true);
  expect(screen.queryByTestId('count-skeleton')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Sightings to name · Unavailable' })).toBeInTheDocument();
});
