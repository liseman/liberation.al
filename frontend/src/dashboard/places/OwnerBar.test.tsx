import { screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import type { Place } from './examples';
import { OwnerBar } from './OwnerBar';
import { EXAMPLE_SCENE } from './scene/example';

const OWNED: Place = {
  id: 'p1',
  name: 'Place 1',
  lastVisit: null,
  scene: EXAMPLE_SCENE,
  planRevision: 'r1',
  machineScene: null,
  origin: { source: 'owner', at: 1762500000, machineNewer: false },
  example: false,
};

it('shows nothing over a plan the machine has no version of its own beside', () => {
  const { container } = renderPage(<OwnerBar place={OWNED} showing="mine" onShow={() => {}} />);
  expect(container.querySelector('button')).toBeNull();
});

it("takes the machine's plan only from the comparison, and only for an owner's plan", () => {
  const compared = { ...OWNED, machineScene: EXAMPLE_SCENE };
  const first = renderPage(<OwnerBar place={compared} showing="mine" onShow={() => {}} />);
  expect(screen.getByRole('radio', { name: "The machine's" })).toBeTruthy();
  expect(screen.queryByRole('button', { name: "Use the machine's plan" })).toBeNull();
  first.unmount();
  const second = renderPage(<OwnerBar place={compared} showing="machine" onShow={() => {}} />);
  expect(screen.getByRole('button', { name: "Use the machine's plan" })).toBeTruthy();
  second.unmount();
  const machines = { ...compared, origin: { source: 'machine' as const, at: 1762500000, machineNewer: false } };
  const { container } = renderPage(<OwnerBar place={machines} showing="machine" onShow={() => {}} />);
  expect(container.querySelector('button')).toBeNull();
});
