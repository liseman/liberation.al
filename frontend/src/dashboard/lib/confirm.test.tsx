import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { confirm } from './confirm';

const open = (options: Parameters<typeof confirm>[0]) => {
  let answer!: Promise<boolean>;
  act(() => {
    answer = confirm(options);
  });
  return answer;
};

describe('confirm (#2538)', () => {
  it('focuses Cancel on a destructive prompt and answers no on Cancel', async () => {
    renderPage(<div />);
    const answer = open({ title: 'Delete Dana', body: 'Recordings stay.', confirm: 'Delete', danger: true });
    const dialog = await screen.findByRole('dialog', { name: 'Delete Dana' });
    expect(within(dialog).getByText('Recordings stay.')).toBeInTheDocument();
    await waitFor(() => expect(within(dialog).getByRole('button', { name: 'Cancel' })).toHaveFocus());
    await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    await expect(answer).resolves.toBe(false);
  });

  it('focuses Cancel on an ordinary prompt too, so Enter right away answers no (#2541)', async () => {
    renderPage(<div />);
    const answer = open({ title: 'Place a call', body: 'Have Bob call Dana now?', confirm: 'Call' });
    const dialog = await screen.findByRole('dialog', { name: 'Place a call' });
    await waitFor(() => expect(within(dialog).getByRole('button', { name: 'Cancel' })).toHaveFocus());
    await userEvent.keyboard('{Enter}');
    await expect(answer).resolves.toBe(false);
  });

  it('focuses the confirm button only when asked to, and answers yes on it', async () => {
    renderPage(<div />);
    const answer = open({
      title: 'New person',
      body: 'Add Robin as a new person?',
      confirm: 'Add',
      focusConfirm: true,
    });
    const dialog = await screen.findByRole('dialog', { name: 'New person' });
    await waitFor(() => expect(within(dialog).getByRole('button', { name: 'Add' })).toHaveFocus());
    await userEvent.click(within(dialog).getByRole('button', { name: 'Add' }));
    await expect(answer).resolves.toBe(true);
  });

  it('opens one prompt per id: asking again while it is up answers no at once', async () => {
    renderPage(<div />);
    const first = open({ id: 'remove-pet-mia', title: 'Remove Mia', confirm: 'Remove', danger: true });
    await screen.findByRole('dialog', { name: 'Remove Mia' });
    await expect(open({ id: 'remove-pet-mia', title: 'Remove Mia', confirm: 'Remove', danger: true })).resolves.toBe(
      false,
    );
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
    await userEvent.click(screen.getByRole('button', { name: 'Remove' }));
    await expect(first).resolves.toBe(true);
    // Answered, the id is free again.
    const again = open({ id: 'remove-pet-mia', title: 'Remove Mia', confirm: 'Remove', danger: true });
    await userEvent.click(await screen.findByRole('button', { name: 'Cancel' }));
    await expect(again).resolves.toBe(false);
  });
});
