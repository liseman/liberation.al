// Model text is never HTML. A small Markdown subset (paragraphs, headings, lists, quotes, rules, code, pipe tables,
// bold, italics, and safe links) gives readable prose; everything else stays literal text.
import type { ReactNode } from 'react';
import classes from './ask.module.css';
import { AppLink } from './bits';
import { plain } from './format';
import { answerHref } from './links';
import { Lazy } from './Paged';

/** Characters of an answer shown before "Read full answer". */
export const LIMIT = 16000;

const INLINE = /(`[^`\n]+`|\*\*[^\n]+?\*\*|__[^\n]+?__|\*[^*\n]+\*|\[[^\]\n]+\]\([^\s)]+\))/g;

/** One line of model text with its inline marks as elements. */
export function inline(value: unknown): ReactNode[] {
  const text = plain(value);
  const out: ReactNode[] = [];
  let end = 0;
  for (const m of text.matchAll(INLINE)) {
    const t = m[0];
    const at = m.index;
    out.push(text.slice(end, at));
    const key = at;
    if (t[0] === '`') out.push(<code key={key}>{t.slice(1, -1)}</code>);
    else if (t.startsWith('**') || t.startsWith('__')) out.push(<strong key={key}>{t.slice(2, -2)}</strong>);
    else if (t[0] === '*') out.push(<em key={key}>{t.slice(1, -1)}</em>);
    else {
      const link = t.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      const target = link?.[2] ? answerHref(link[2]) : null;
      if (!link || !target) out.push(<span key={key}>{link ? link[1] : t}</span>);
      else if (/^https?:/i.test(target))
        out.push(
          <a key={key} href={target} target="_blank" rel="noopener noreferrer">
            {link[1]}
          </a>,
        );
      else
        out.push(
          <AppLink key={key} to={target}>
            {link[1]}
          </AppLink>,
        );
    }
    end = at + t.length;
  }
  out.push(text.slice(end));
  return out;
}

type Block =
  | { kind: 'p' | 'h5' | 'quote'; text: string }
  | { kind: 'ul' | 'ol'; items: string[] }
  | { kind: 'code'; text: string }
  | { kind: 'table'; head: string[]; rows: string[][]; align: ('left' | 'right' | 'center' | undefined)[] }
  | { kind: 'hr' };

/** A pipe-table row's cells: `| a | b |` gives ["a", "b"]. */
const cells = (line: string): string[] =>
  line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map(c => c.trim());

/** A table's delimiter row (`| --- | ---: |`), with its columns' alignment, or null for any other line. */
function delimiter(line: string): ('left' | 'right' | 'center' | undefined)[] | null {
  if (!/^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line)) return null;
  return cells(line).map(c =>
    c.startsWith(':') && c.endsWith(':')
      ? 'center'
      : c.endsWith(':')
        ? 'right'
        : c.startsWith(':')
          ? 'left'
          : undefined,
  );
}

/** Model text split into Markdown blocks. */
export function blocks(text: string): Block[] {
  const out: Block[] = [];
  let paragraph: string[] = [];
  let open: { kind: 'ul' | 'ol'; items: string[] } | null = null;
  let fence: string | null = null;
  let code: string[] = [];
  const flush = () => {
    if (paragraph.length) out.push({ kind: 'p', text: paragraph.join(' ') });
    paragraph = [];
  };
  const flushCode = () => {
    out.push({ kind: 'code', text: code.join('\n') });
    code = [];
  };
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] as string;
    const marker = line.match(/^\s*(`{3,}|~{3,})/)?.[1];
    if (fence) {
      if (marker && marker[0] === fence[0] && marker.length >= fence.length) {
        flushCode();
        fence = null;
      } else code.push(line);
      continue;
    }
    if (marker) {
      flush();
      open = null;
      fence = marker;
      continue;
    }
    if (!line.trim()) {
      flush();
      open = null;
      continue;
    }
    // A pipe table: a header row, then its delimiter row, then body rows until a line without a pipe.
    const align = !fence && line.includes('|') ? delimiter(lines[i + 1] ?? '') : null;
    if (align) {
      flush();
      open = null;
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && (lines[i] as string).includes('|') && (lines[i] as string).trim())
        rows.push(cells(lines[i++] as string));
      i--;
      out.push({ kind: 'table', head: cells(line), rows, align });
      continue;
    }
    const heading = line.match(/^\s{0,3}#{1,6}\s+(.+)$/);
    const item = line.match(/^\s*(?:([-*+])|\d+[.)])\s+(.+)$/);
    if (heading) {
      flush();
      open = null;
      out.push({ kind: 'h5', text: heading[1] ?? '' });
    } else if (item) {
      flush();
      const kind = item[1] ? 'ul' : 'ol';
      if (!open || open.kind !== kind) {
        open = { kind, items: [] };
        out.push(open);
      }
      open.items.push(item[2] ?? '');
    } else if (/^\s*>\s?/.test(line)) {
      flush();
      open = null;
      out.push({ kind: 'quote', text: line.replace(/^\s*>\s?/, '') });
    } else if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      flush();
      open = null;
      out.push({ kind: 'hr' });
    } else {
      open = null;
      paragraph.push(line.trim());
    }
  }
  flush();
  if (fence) flushCode();
  return out;
}

function BlockView({ block }: { block: Block }) {
  switch (block.kind) {
    case 'p':
      return <p>{inline(block.text)}</p>;
    case 'h5':
      return <h5>{inline(block.text)}</h5>;
    case 'quote':
      return <blockquote>{inline(block.text)}</blockquote>;
    case 'code':
      return (
        <pre>
          <code>{block.text}</code>
        </pre>
      );
    case 'hr':
      return <hr />;
    case 'table':
      return (
        <table>
          <thead>
            <tr>
              {block.head.map((h, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: columns have no identity beyond their place
                <th key={i} style={{ textAlign: block.align[i] }}>
                  {inline(h)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row, r) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: rows have no identity beyond their place
              <tr key={r}>
                {row.map((c, i) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: cells have no identity beyond their place
                  <td key={i} style={{ textAlign: block.align[i] }}>
                    {inline(c)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
    case 'ul':
    case 'ol': {
      const List = block.kind;
      return (
        <List>
          {block.items.map((item, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: list items have no identity beyond their place
            <li key={i}>{inline(item)}</li>
          ))}
        </List>
      );
    }
  }
}

export function Prose({ text }: { text: string }) {
  return (
    <div className={classes.prose}>
      {blocks(text).map((b, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: blocks have no identity beyond their place
        <BlockView key={i} block={b} />
      ))}
    </div>
  );
}

/** Model text, bounded at LIMIT characters; the full answer stays one click away instead of a raw dump. */
export function TextBlock({ value }: { value: unknown }) {
  const text = plain(value);
  return (
    <>
      <Prose text={text.slice(0, LIMIT)} />
      {text.length > LIMIT && (
        <Lazy summary={`Read full answer (${text.length.toLocaleString()} characters)`}>
          {() => <Prose text={text} />}
        </Lazy>
      )}
    </>
  );
}
