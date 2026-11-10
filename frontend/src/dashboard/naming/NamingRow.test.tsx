// The naming inbox's card keys: the arrows move the selection to the card beside, above, or below on screen,
// whatever the list order, and Space runs the card's own Space answer. And the card itself:
// Yes beside the guess, and the answers as captioned buttons with their keys and a popover each.
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Providers } from '../../shared/Providers';
import { type InboxItem, NamingInbox } from './NamingInbox';
import { type CardAnswer, NamingRow } from './NamingRow';

const CARD_W = 400;
const CARD_H = 200;
const GAP = 12;

/** Lays the cards out as a grid of `columns`, in list order, by mocking each card's rectangle. */
function layOut(columns: number) {
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    const at = [...(this.parentElement?.children ?? [])].indexOf(this);
    if (!this.matches('li[data-row]') || at < 0) return new DOMRect(0, 0, 0, 0);
    const x = (at % columns) * (CARD_W + GAP);
    const y = Math.floor(at / columns) * (CARD_H + GAP);
    return new DOMRect(x, y, CARD_W, CARD_H);
  });
}

function show(count: number, onSpace?: (key: string) => void) {
  const items: InboxItem[] = Array.from({ length: count }, (_, i) => ({
    key: `c${i}`,
    guess: null,
    render: () => (
      <NamingRow
        rowKey={`c${i}`}
        label={`Card ${i}`}
        lead={null}
        meta={null}
        guess={null}
        picker={<input aria-label={`Name for card ${i}`} />}
        onIgnore={() => {}}
        ignoreHint="Set it aside."
        onSpace={onSpace ? () => onSpace(`c${i}`) : undefined}
      />
    ),
  }));
  render(
    <Providers>
      <NamingInbox prefs="test" title="Cards" headingId="cards-h" items={items} />
    </Providers>,
  );
  return screen.getAllByRole('listitem');
}

const selected = () => screen.getAllByRole('listitem').find(li => li.getAttribute('aria-current') === 'true');

afterEach(() => vi.restoreAllMocks());

describe('arrow keys on the naming cards', () => {
  it('moves by screen position in a three-column grid', async () => {
    layOut(3);
    const cards = show(7);
    const user = userEvent.setup();
    // Laid out:  c0 c1 c2
    //            c3 c4 c5
    //            c6
    await user.click(cards[0] as HTMLElement);
    await user.keyboard('{ArrowRight}');
    expect(selected()).toBe(cards[1]);
    expect(cards[1]).toHaveFocus();
    await user.keyboard('{ArrowDown}');
    expect(selected()).toBe(cards[4]);
    await user.keyboard('{ArrowRight}');
    expect(selected()).toBe(cards[5]);
    await user.keyboard('{ArrowRight}'); // the row's end: it stays
    expect(selected()).toBe(cards[5]);
    await user.keyboard('{ArrowDown}'); // nothing under c5: c6 is the nearest card in the row below
    expect(selected()).toBe(cards[6]);
    await user.keyboard('{ArrowDown}');
    expect(selected()).toBe(cards[6]);
    await user.keyboard('{ArrowUp}');
    expect(selected()).toBe(cards[3]);
    await user.keyboard('{ArrowLeft}');
    expect(selected()).toBe(cards[3]);
    await user.keyboard('{ArrowUp}');
    expect(selected()).toBe(cards[0]);
  });

  it('moves straight down a two-column grid, not to the next card in the list', async () => {
    layOut(2);
    const cards = show(4);
    const user = userEvent.setup();
    await user.click(cards[1] as HTMLElement);
    await user.keyboard('{ArrowDown}');
    expect(selected()).toBe(cards[3]);
    await user.keyboard('{ArrowLeft}');
    expect(selected()).toBe(cards[2]);
  });

  it('keeps Left and Right still in one column, and Up and Down move', async () => {
    layOut(1);
    const cards = show(3);
    const user = userEvent.setup();
    await user.click(cards[1] as HTMLElement);
    await user.keyboard('{ArrowRight}');
    expect(selected()).toBe(cards[1]);
    await user.keyboard('{ArrowLeft}');
    expect(selected()).toBe(cards[1]);
    await user.keyboard('{ArrowDown}');
    expect(selected()).toBe(cards[2]);
  });

  it('scrolls the card it moves to into view', async () => {
    layOut(2);
    const cards = show(4);
    const scrolled = vi.spyOn(HTMLElement.prototype, 'scrollIntoView');
    const user = userEvent.setup();
    await user.click(cards[0] as HTMLElement);
    await user.keyboard('{ArrowDown}');
    expect(scrolled.mock.contexts).toEqual([cards[2]]);
  });

  it('leaves the arrows to a name box while typing', async () => {
    layOut(2);
    const cards = show(4);
    const user = userEvent.setup();
    await user.click(cards[0] as HTMLElement);
    await user.click(within(cards[0] as HTMLElement).getByRole('textbox', { name: 'Name for card 0' }));
    await user.keyboard('{ArrowRight}{ArrowDown}');
    expect(selected()).toBe(cards[0]);
  });
});

describe('Space on the naming cards', () => {
  it("runs the selected card's Space answer", async () => {
    layOut(2);
    const spaced = vi.fn();
    const cards = show(2, spaced);
    const user = userEvent.setup();
    await user.click(cards[1] as HTMLElement);
    await user.keyboard(' ');
    expect(spaced.mock.calls).toEqual([['c1']]);
  });

  it('types a space in the name box instead', async () => {
    layOut(2);
    const spaced = vi.fn();
    const cards = show(2, spaced);
    const user = userEvent.setup();
    const box = within(cards[0] as HTMLElement).getByRole('textbox', { name: 'Name for card 0' });
    await user.click(box);
    await user.keyboard('a b');
    expect(box).toHaveValue('a b');
    expect(spaced).not.toHaveBeenCalled();
  });
});

describe("a naming card's answers (#3263)", () => {
  const MIA = { id: 'mia', name: 'Mia', score: 67, band: 'maybe' as const };
  function card(answers: CardAnswer[] = [], hotkeys?: Record<string, () => void>) {
    const items: InboxItem[] = [
      {
        key: 'c0',
        guess: MIA,
        render: () => (
          <NamingRow
            rowKey="c0"
            label="Card 0"
            lead={null}
            meta={null}
            guess={MIA}
            picker={<input aria-label="Name for card 0" />}
            onYes={() => {}}
            onNot={() => {}}
            notHint={who => `This isn't ${who}.`}
            onIgnore={() => {}}
            ignoreHint="Set it aside for good."
            answers={answers}
            hotkeys={hotkeys}
          />
        ),
      },
    ];
    render(
      <Providers>
        <NamingInbox prefs="test" title="Cards" headingId="cards-h" items={items} />
      </Providers>,
    );
    return screen.getByRole('listitem');
  }
  const multi = (onClick = () => {}): CardAnswer => ({
    label: 'Multiple speakers',
    caption: 'Multi',
    hint: 'Several people talk over each other.',
    icon: null,
    shortcut: 'M',
    onClick,
  });

  it("puts Yes on the guess's line", () => {
    const row = card();
    const yes = within(row).getByRole('button', { name: 'Yes, Mia' });
    expect(yes.parentElement).toHaveTextContent(/^Maybe MiaYes, Mia$/);
  });

  it("shows each answer as its caption over its key, in order: Not, the kind's own, then Ignore", () => {
    const row = card([multi()]);
    const name = within(row).getByRole('textbox', { name: 'Name for card 0' });
    const buttons = ['Not Mia', 'Multiple speakers', 'Ignore'].map(n => within(row).getByRole('button', { name: n }));
    expect(buttons.map(b => b.textContent)).toEqual(['NotN', 'MultiM', 'IgnoreI']);
    expect(buttons.map(b => b.getAttribute('aria-keyshortcuts'))).toEqual(['N', 'M', 'I']);
    // One row of them, under the name box
    const answers = buttons[0]?.parentElement;
    expect(buttons.every(b => b.parentElement === answers)).toBe(true);
    expect(answers?.previousElementSibling).toContainElement(name);
  });

  it('says what an answer does in a popover, on hover and on keyboard focus', async () => {
    const row = card([multi()]);
    const user = userEvent.setup();
    await user.hover(within(row).getByRole('button', { name: 'Multiple speakers' }));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Multi (M): Several people talk over each other.');
    await user.unhover(within(row).getByRole('button', { name: 'Multiple speakers' }));
    within(row).getByRole('button', { name: 'Ignore' }).focus();
    expect(await screen.findByRole('tooltip', { name: /^Ignore/ })).toHaveTextContent(
      'Ignore (I): Set it aside for good.',
    );
  });

  it("runs an answer's key, and a key with no button of its own, on the focused card", async () => {
    const said: string[] = [];
    const row = card([multi(() => said.push('multi'))], { a: () => said.push('all') });
    const user = userEvent.setup();
    row.focus();
    await user.keyboard('ma');
    expect(said).toEqual(['multi', 'all']);
  });
});
