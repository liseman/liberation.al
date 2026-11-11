import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { Prose } from './prose';
import { citeLinks, Result } from './Result';

const show = (result: unknown, history = false) => renderPage(<Result result={result} history={history} />);
const cards = () => screen.queryAllByRole('article');

describe('the answer', () => {
  it('reads as prose, with markup kept inert and warnings as a list', () => {
    show({
      status: 'partial',
      answer:
        'First **bold** paragraph.\n\nSecond *gentle* paragraph.\n\n- Detail one\n- Detail two\n\n<script>alert(1)</script> [unsafe](javascript:alert(1))',
      warnings: ['Camera two unavailable', 'Transcript has gaps'],
    });
    const answer = screen.getByRole('heading', { name: 'Answer' }).closest('section') as HTMLElement;
    expect(within(answer).getByText('bold').tagName).toBe('STRONG');
    expect(within(answer).getByText('gentle').tagName).toBe('EM');
    expect(within(answer).getByText(/<script>alert\(1\)<\/script>/)).toBeInTheDocument();
    expect(within(answer).getByText('unsafe').tagName).toBe('SPAN');
    expect(screen.getByText('Camera two unavailable').tagName).toBe('LI');
    expect(document.querySelector('script')).toBeNull();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('bounds a long answer and expands it once, in place', async () => {
    const long = `${'A long answer paragraph. '.repeat(1800)}END_SENTINEL`;
    show({ answer: long });
    const answer = screen.getByRole('heading', { name: 'Answer' }).closest('section') as HTMLElement;
    expect(answer).not.toHaveTextContent('END_SENTINEL');
    expect((answer.textContent ?? '').length).toBeLessThan(20000);
    const more = within(answer).getByText('Read full answer (45,012 characters)');
    await userEvent.click(more);
    expect(answer).toHaveTextContent('END_SENTINEL');
  });

  it.each([
    ['nothing optional', {}],
    ['nulls', { warnings: null, coverage: null, evidence: null }],
    ['prose in place of lists', { warnings: { message: 'w' }, coverage: 'c' }],
    ['wrong types', { warnings: 42, coverage: 42 }],
    ['junk entries', { evidence: [null, 42], warnings: [null, { message: 'Structured warning' }] }],
  ])('stays readable with %s', (_, optional) => {
    show({ answer: 'Still readable', ...optional }, true);
    expect(screen.getByText('Still readable')).toBeInTheDocument();
    expect(document.body).not.toHaveTextContent(/\[object Object\]|"message"/);
  });

  it('renders structured warnings as sentences', () => {
    show({ answer: 'x', warnings: [null, { message: 'Structured warning' }] });
    expect(screen.getByText('Structured warning').tagName).toBe('LI');
    expect(cards()).toHaveLength(0);
  });
});

describe('history panels', () => {
  it('appear only for a history answer', () => {
    const result = { answer: 'A', coverage: { partial: true }, evidence: [{ id: 1, text: 'frame' }] };
    const view = show({ ...result, is_history: false }, true);
    expect(screen.queryByText(/Evidence \(|History coverage/)).not.toBeInTheDocument();
    view.unmount();
    show({ ...result, scope: 'general' }, true);
    expect(screen.queryByText(/Evidence \(|History coverage/)).not.toBeInTheDocument();
  });

  it('show coverage as labelled facts, with first and last as times', async () => {
    show(
      {
        answer: 'A',
        coverage: { truncated: true, first: 1700000000, cameras: ['cam0'], detail: { a: { b: { c: { d: 1 } } } } },
      },
      true,
    );
    expect(screen.getByText('This answer is based on partial recorded evidence.')).toBeInTheDocument();
    await userEvent.click(screen.getByText('History coverage'));
    expect(screen.getByText('Truncated').nextElementSibling).toHaveTextContent(/^Yes$/);
    expect(screen.getByText('First').nextElementSibling?.querySelector('time')).toHaveAttribute(
      'datetime',
      '2023-11-14T22:13:20.000Z',
    );
    expect(screen.getByText('Cameras').nextElementSibling).toHaveTextContent(/^cam0$/);
    expect(screen.getByText('Additional details omitted')).toBeInTheDocument();
  });

  it('show evidence meta and its timestamp', () => {
    show({ answer: 'A', evidence: [{ id: 'C1', speaker: 'Dana', camera: 'cam0', timestamp: 'not a date' }] }, true);
    const card = cards()[0] as HTMLElement;
    expect(card).toHaveTextContent(/^not a date · Speaker: Dana · Camera: cam0 · ID: C1$/);
  });
});

describe('Prose', () => {
  it('renders the Markdown subset and nothing else', () => {
    renderPage(
      <Prose
        text={[
          '## Heading',
          'one line',
          'continued',
          '1. first',
          '2) second',
          '> quoted',
          '***',
          '```',
          '<b>raw</b>',
          '  kept',
          '```',
          'use `code`, __strong__, [docs](https://example.org/d), and [review](/review#t=1).',
        ].join('\n')}
      />,
    );
    expect(screen.getByRole('heading', { name: 'Heading' }).tagName).toBe('H5');
    expect(screen.getByText('one line continued').tagName).toBe('P');
    expect(screen.getAllByRole('listitem').map(li => li.textContent)).toEqual(['first', 'second']);
    expect(screen.getByRole('list').tagName).toBe('OL');
    expect(screen.getByText('quoted').tagName).toBe('BLOCKQUOTE');
    expect(screen.getByRole('separator')).toBeInTheDocument();
    expect(document.querySelector('pre code')?.textContent).toBe('<b>raw</b>\n  kept');
    expect(document.querySelector('b')).toBeNull();
    expect(screen.getByText('code').tagName).toBe('CODE');
    expect(screen.getByText('strong').tagName).toBe('STRONG');
    const docs = screen.getByRole('link', { name: 'docs' });
    expect(docs).toHaveAttribute('href', 'https://example.org/d');
    expect(docs).toHaveAttribute('target', '_blank');
    expect(docs).toHaveAttribute('rel', 'noopener noreferrer');
    expect(screen.getByRole('link', { name: 'review' })).toHaveAttribute('href', '/review#t=1');
  });

  it('keeps an unclosed code fence to the end', () => {
    renderPage(<Prose text={'~~~\nline one\n```\nline two'} />);
    expect(document.querySelector('pre code')?.textContent).toBe('line one\n```\nline two');
  });
});

describe('citeLinks', () => {
  const evidence = [
    { id: 'C6', href: '/review#day=2026-11-04&t=1&cam=living-room' },
    { label: 'J3', link: '/review#day=2026-11-04&t=2' },
    { id: 'C7', href: '/review#bad (paren)' },
  ];
  it('links each code an evidence item has, bare or in brackets, and leaves the rest as written', () => {
    expect(citeLinks('C6 (13:10) shows two; J3 notes sleep [J3]; C99 and C7 stay; WC6 and C6x too.', evidence)).toBe(
      '[C6](/review#day=2026-11-04&t=1&cam=living-room) (13:10) shows two; [J3](/review#day=2026-11-04&t=2) notes ' +
        'sleep [J3](/review#day=2026-11-04&t=2); C99 and C7 stay; WC6 and C6x too.',
    );
  });
  it('leaves code spans and links already in the answer alone', () => {
    expect(citeLinks('`C6` and [C6](/review#x) and C6', evidence)).toBe(
      '`C6` and [C6](/review#x) and [C6](/review#day=2026-11-04&t=1&cam=living-room)',
    );
  });
});
