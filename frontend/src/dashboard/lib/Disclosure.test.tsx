import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { Disclosure } from './Disclosure';

function Controlled({ change }: { change: (opened: boolean) => void }) {
  const [opened, setOpened] = useState(false);
  return (
    <Disclosure
      summary="Saved"
      opened={opened}
      onChange={next => {
        change(next);
        setOpened(next);
      }}
    >
      Content
    </Disclosure>
  );
}

describe('Disclosure', () => {
  it('builds lazy content on first open, retaining edits when closed and reopened with the keyboard', async () => {
    const build = vi.fn(() => <input aria-label="Draft" defaultValue="First" />);
    renderPage(
      <Disclosure summary="More" lazy>
        {build}
      </Disclosure>,
    );
    const button = screen.getByRole('button', { name: 'More' });
    expect(build).not.toHaveBeenCalled();
    expect(button).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByLabelText('Draft')).not.toBeInTheDocument();
    await userEvent.tab();
    await userEvent.keyboard('{Enter}');
    expect(button).toHaveAttribute('aria-expanded', 'true');
    const input = screen.getByRole('textbox', { name: 'Draft' });
    expect(document.getElementById(button.getAttribute('aria-controls') ?? '')).toContainElement(input);
    await userEvent.type(input, ' edit');
    await userEvent.click(button);
    expect(button).toHaveAttribute('aria-expanded', 'false');
    await userEvent.keyboard(' ');
    expect(screen.getByRole('textbox', { name: 'Draft' })).toBe(input);
    expect(input).toHaveValue('First edit');
  });

  it('notifies a controlled consumer on both open and close', async () => {
    const change = vi.fn();
    renderPage(<Controlled change={change} />);
    const button = screen.getByRole('button', { name: 'Saved' });
    await userEvent.click(button);
    await userEvent.click(button);
    expect(change.mock.calls).toEqual([[true], [false]]);
    expect(button).toHaveAttribute('aria-expanded', 'false');
  });
});
