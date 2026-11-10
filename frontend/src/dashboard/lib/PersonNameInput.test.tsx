import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { nameOptions, PersonNameInput } from './PersonNameInput';

const labels = (...args: Parameters<typeof nameOptions>) => nameOptions(...args).map(o => o.label);

describe('nameOptions', () => {
  const known = ['Dan', 'Dana', 'Eli'];

  it('puts guesses first at rest', () => {
    expect(labels(known, '', [{ name: 'Eli', label: 'Eli (guess, 80%)' }])).toEqual([
      'Eli (guess, 80%)',
      'Dan',
      'Dana',
    ]);
  });

  it('puts an exact match first, any case, then the other names that contain what is typed', () => {
    expect(labels(known, 'dan')).toEqual(['Dan', 'Dana']);
  });

  it('offers to add a name nobody has, first, and only where new names are allowed', () => {
    expect(labels(known, 'Da ')).toEqual(['Add “Da”', 'Dan', 'Dana']);
    expect(labels(known, 'Da', [], false)).toEqual(['Dan', 'Dana']);
    expect(labels(known, 'Zed', [], false)).toEqual([]);
  });
});

function Box({ onPick }: { onPick: (name: string) => void }) {
  const [typed, setTyped] = useState('');
  return (
    <PersonNameInput
      known={Array.from({ length: 300 }, (_, i) => `Person ${i}`)}
      suggestions={[{ name: 'Person 7', label: 'Person 7 (guess)' }]}
      value={typed}
      onChange={setTyped}
      onPick={onPick}
      ariaLabel="Name"
    />
  );
}

describe('PersonNameInput', () => {
  it('keeps no options in the page while closed, so a list of name boxes costs one box each', async () => {
    const onPick = vi.fn();
    renderPage(<Box onPick={onPick} />);
    expect(screen.queryAllByRole('option', { hidden: true })).toEqual([]);
    await userEvent.click(screen.getByRole('combobox', { name: 'Name' }));
    expect(await screen.findAllByRole('option')).toHaveLength(300);
    await userEvent.keyboard('{Escape}');
    expect(screen.queryAllByRole('option', { hidden: true })).toEqual([]);
  });

  it('opens with the top guess highlighted, so Enter takes it', async () => {
    const onPick = vi.fn();
    renderPage(<Box onPick={onPick} />);
    await userEvent.click(screen.getByRole('combobox', { name: 'Name' }));
    await screen.findByRole('option', { name: 'Person 7 (guess)' });
    await userEvent.keyboard('{Enter}');
    expect(onPick).toHaveBeenCalledExactlyOnceWith('Person 7');
  });
});
