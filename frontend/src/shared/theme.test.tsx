import { Drawer, Modal } from '@mantine/core';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { Providers } from './Providers';

// A dialog's content is a Paper, which the theme gives the card surface; its header has to match, or the title reads
// as a strip on a block.
describe('dialogs', () => {
  it.each([
    ['Modal', Modal],
    ['Drawer', Drawer],
  ] as const)('a %s header sits on the same surface as its body', (name, Dialog) => {
    render(
      <Providers test>
        <Dialog opened onClose={() => {}} title="Ask LLMs">
          <p>Body</p>
        </Dialog>
      </Providers>,
    );
    const header = screen.getByText('Ask LLMs').closest(`.mantine-${name}-header`) as HTMLElement;
    const content = header.closest(`.mantine-${name}-content`) as HTMLElement;
    expect(header.style.backgroundColor).toBe('var(--lib-card)');
    expect(content.style.backgroundColor).toBe('var(--lib-card)');
  });
});

// Their close (x) button has only an icon: it needs a name a screen reader can say.
describe('dialog close buttons', () => {
  it.each([
    ['Modal', Modal],
    ['Drawer', Drawer],
  ] as const)("a %s's close button is called Close", (_name, Dialog) => {
    render(
      <Providers test>
        <Dialog opened onClose={() => {}} title="Ask LLMs">
          <p>Body</p>
        </Dialog>
      </Providers>,
    );
    expect(screen.getByRole('button', { name: 'Close' })).toBeInTheDocument();
  });
});
