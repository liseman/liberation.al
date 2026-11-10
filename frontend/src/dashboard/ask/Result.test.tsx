import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { Prose } from './prose';
import { citeLinks, Result } from './Result';

const show = (result: unknown, history = false) => renderPage(<Result result={result} history={history} />);
const cards = () => screen.queryAllByRole('article');

describe('model answers', () => {
  it('gives each model a visible card with its status, timing, and readable prose', () => {
    show({
      status: 'partial',
      answer: 'Combined independent conclusion.',
      warnings: ['Camera two unavailable', 'Transcript has gaps'],
      coverage: { partial: true, available: 8, included: 5 },
      panel: {
        participants: [
          {
            model: 'provider/Alpha',
            status: 'ok',
            backend: 'ollama',
            elapsed_seconds: 1.25,
            answer:
              'First **bold** paragraph.\n\nSecond *gentle* paragraph.\n\n- Detail one\n- Detail two\n\n<script>alert(1)</script> [unsafe](javascript:alert(1))',
          },
          {
            model: 'provider/Beta',
            status: 'error',
            backend: 'cloud',
            elapsed_seconds: 12.34,
            error: { message: 'Provider timed out' },
          },
        ],
        unavailable: [{ model: 'provider/Gamma', reason: 'Credentials unavailable' }],
        discovery_errors: [{ provider: 'Delta', error: 'Discovery endpoint offline' }],
      },
    });
    const [alpha, beta] = cards() as [HTMLElement, HTMLElement];
    expect(cards()).toHaveLength(2);
    expect(within(alpha).getByRole('heading', { name: 'provider/Alpha' })).toBeInTheDocument();
    expect(alpha).toHaveTextContent('Answered');
    expect(alpha).toHaveTextContent('Local model · 1.25s');
    expect(within(alpha).getByText('bold').tagName).toBe('STRONG');
    expect(within(alpha).getByText('gentle').tagName).toBe('EM');
    expect(
      within(alpha)
        .getAllByRole('listitem')
        .map(li => li.textContent),
    ).toEqual(['Detail one', 'Detail two']);
    expect(within(alpha).getByText(/<script>alert\(1\)<\/script>/)).toBeInTheDocument();
    expect(within(alpha).getByText('unsafe').tagName).toBe('SPAN');
    expect(alpha).not.toHaveTextContent('Combined independent conclusion');
    expect(beta).toHaveTextContent('Failed');
    expect(beta).toHaveTextContent('Cloud model · 12.3s');
    expect(beta).toHaveTextContent('Provider timed out');
    expect(screen.getByText('1 of 2 models returned an answer.')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Combined answer' })).toBeInTheDocument();
    expect(screen.getByText('provider/Gamma - Credentials unavailable')).toBeInTheDocument();
    expect(screen.getByText('Discovery endpoint offline')).toBeInTheDocument();
    expect(screen.getByText('Camera two unavailable').tagName).toBe('LI');
    expect(screen.queryByText(/coverage/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(document.querySelector('script')).toBeNull();
  });

  it('still gets one card per model from a legacy bare list', () => {
    show({
      answer: 'Legacy combined',
      panel: [{ model: 'Legacy friend', status: 'ok', answer: 'Legacy individual', elapsed_seconds: 0 }],
    });
    expect(cards()).toHaveLength(1);
    expect(cards()[0]).toHaveTextContent('Legacy individual');
    expect(cards()[0]).toHaveTextContent('0.00s');
  });

  it('keeps the individual answers when the combining step failed', () => {
    show({
      status: 'synthesis_failed',
      answer: 'Synthesis failed; individual answers remain available.',
      panel: {
        participants: [{ model: 'Survivor', status: 'ok', answer: 'Preserved individual answer' }],
        referee: { model: 'Referee', status: 'error', error: { message: 'Synthesis timed out' } },
      },
    });
    expect(cards()[0]).toHaveTextContent('Preserved individual answer');
    expect(screen.getByText('Summary unavailable')).toBeInTheDocument();
    expect(screen.getByText('Combined answer could not be completed: Synthesis timed out')).toBeInTheDocument();
  });

  it('names the referee that combined the answers', () => {
    show({ answer: 'x', panel: { participants: [], referee: { name: 'Judge', status: 'ok', elapsed_seconds: 3 } } });
    expect(screen.getByText('Combined by Judge · 3.00s')).toBeInTheDocument();
    expect(screen.getByText('No individual model answers were returned.')).toBeInTheDocument();
  });

  it('shows a model without an answer or a reason as returning nothing', () => {
    show({ panel: { participants: [{ model: 'Quiet', finish_reason: 'length' }] } });
    expect(cards()[0]).toHaveTextContent('Unavailable');
    expect(cards()[0]).toHaveTextContent('No answer was returned.');
    expect(cards()[0]).toHaveTextContent('This response reached its length limit.');
  });

  it('bounds a long answer and expands it once, in place', async () => {
    const long = `${'A long answer paragraph. '.repeat(1800)}END_SENTINEL`;
    show({ answer: 'Brief', panel: { participants: [{ model: 'Verbose', status: 'ok', answer: long }] } });
    const card = cards()[0] as HTMLElement;
    expect(card).not.toHaveTextContent('END_SENTINEL');
    expect((card.textContent ?? '').length).toBeLessThan(20000);
    const more = within(card).getByText('Read full answer (45,012 characters)');
    await userEvent.click(more);
    expect(card).toHaveTextContent('END_SENTINEL');
    const expanded = card.textContent;
    await userEvent.click(more);
    expect(card.textContent).toBe(expanded);
  });

  it.each([
    ['nothing optional', {}],
    ['nulls', { panel: null, warnings: null, coverage: null, evidence: null }],
    ['prose in place of lists', { panel: { participants: null }, warnings: { message: 'w' }, coverage: 'c' }],
    [
      'wrong types',
      { panel: { participants: 'malformed', unavailable: 42, discovery_errors: {} }, warnings: 42, coverage: 42 },
    ],
    [
      'junk entries',
      {
        panel: {
          participants: [null, 42, {}, { model: 'Odd', answer: null, error: { message: 'No answer available' } }],
        },
        evidence: [null, 42],
        warnings: [null, { message: 'Structured warning' }],
      },
    ],
  ])('stays readable with %s', (_, optional) => {
    show({ answer: 'Still readable', ...optional }, true);
    expect(screen.getByText('Still readable')).toBeInTheDocument();
    expect(document.body).not.toHaveTextContent(/\[object Object\]|"participants"|"message"/);
  });

  it('renders structured warnings and odd panel entries as sentences', () => {
    show({
      answer: 'x',
      panel: {
        participants: [null, 42, {}, { model: 'Odd', answer: null, error: { message: 'No answer available' } }],
      },
      warnings: [null, { message: 'Structured warning' }],
    });
    expect(cards()).toHaveLength(2);
    expect(cards()[1]).toHaveTextContent('No answer available');
    expect(screen.getByText('Structured warning').tagName).toBe('LI');
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
