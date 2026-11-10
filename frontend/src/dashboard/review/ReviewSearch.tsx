// Instant text search of this day's loaded sources, with an explicitly labeled local semantic fallback.
import { Button, Combobox, Group, Loader, Mark, Text, TextInput, useCombobox } from '@mantine/core';
import {
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { messageOf } from '../../shared/http';
import { hms } from '../lib/format';
import { LoadError } from '../lib/page';
import type { TranscriptEntry } from '../live/types';
import { lineSearchText } from '../transcript/model';
import { type Block, type SemanticSearchResult, semanticSearch } from './api';
import { createSearchIndex, type SearchDocument, searchIndex } from './search';
import classes from './search.module.css';

export const SEMANTIC_DELAY_MS = 300;
type Coverage = 'loading' | 'partial' | 'ready' | 'error';
type Meaning = {
  key: string;
  status: 'waiting' | 'loading' | 'ready' | 'error';
  results: SemanticSearchResult[];
  error?: string;
};

/** Matcher ranges refer to the original text's UTF16 offsets, never HTML. */
function highlighted(text: string, ranges: readonly [number, number][]): ReactNode {
  let at = 0;
  const parts: ReactNode[] = [];
  for (const [start, end] of ranges) {
    parts.push(text.slice(at, start), <Mark key={`${start}:${end}`}>{text.slice(start, end)}</Mark>);
    at = end;
  }
  parts.push(text.slice(at));
  return parts;
}

export function ReviewSearch({
  day,
  entries,
  blocks,
  loading = false,
  summary = 'ready',
  onSeek,
  onSearching,
  beside = false,
  onRetrySummary,
}: {
  day: string | null;
  entries: readonly TranscriptEntry[];
  blocks: readonly Block[];
  loading?: boolean;
  summary?: Coverage;
  onSeek: (ts: number) => void;
  /** The search as typed (trimmed), or '' when there's none. */
  onSearching?: (query: string) => void;
  /** The day's transcript shows below, dimmed to the matches: the results take a fixed share, not all of it. */
  beside?: boolean;
  onRetrySummary?: () => void;
}) {
  const [text, setText] = useState('');
  const query = text.trim();
  const meaningful = query.length <= 500 && /[\p{L}\p{N}]/u.test(query);
  const input = useRef<HTMLInputElement>(null);
  const key = JSON.stringify([day, query]);
  const id = useId();
  const [active, setActive] = useState(0);
  const [manual, setManual] = useState({ key: '', attempt: 0 });
  const [meaning, setMeaning] = useState<Meaning | null>(null);
  const documents = useMemo<SearchDocument[]>(
    () => [
      ...entries
        .filter(entry => entry.text?.trim())
        .map((entry, index) => ({
          id: `transcript:${entry.line ?? (typeof entry.id === 'string' ? entry.id : `${entry.ts_start}:${index}`)}`,
          kind: 'transcript' as const,
          ts: entry.ts_start,
          // The text the transcript below searches too: the name it shows, never a placeholder like
          // "Speaker 250", then the words.
          text: lineSearchText(entry.speaker, String(entry.text ?? '')),
        })),
      ...blocks.map(block => ({
        id: `summary:${block.start}:${block.end}`,
        kind: 'summary' as const,
        ts: block.start,
        text: block.text,
      })),
    ],
    [entries, blocks],
  );
  const index = useMemo(() => createSearchIndex(documents), [documents]);
  const hits = useMemo(() => (query ? searchIndex(index, query) : []), [index, query]);
  const explicit = manual.key === key && manual.attempt > 0;
  const requestKey = JSON.stringify([day, query, explicit ? manual.attempt : 0]);
  const wanted = !!day && meaningful && (explicit || (!loading && hits.length === 0));
  const current = meaning?.key === requestKey ? meaning : null;
  const similar = wanted && current?.status === 'ready' ? current.results : [];
  const results = [
    ...hits.map(hit => ({ id: hit.document.id, ts: hit.document.ts })),
    ...similar.map(result => ({ id: `semantic:${result.id}`, ts: result.ts })),
  ];
  const selected = Math.min(active, Math.max(0, results.length - 1));
  const activeId = results.length ? `${id}-${selected}` : undefined;
  const combobox = useCombobox({ opened: !!query && results.length > 0 });
  const { selectOption, resetSelectedOption } = combobox;
  const byMeaning = () => setManual(previous => ({ key, attempt: previous.attempt + 1 }));

  useLayoutEffect(() => onSearching?.(query), [onSearching, query]);
  useEffect(() => {
    if (!wanted || !day) return;
    const controller = new AbortController();
    let stale = false;
    setMeaning({ key: requestKey, status: 'waiting', results: [] });
    const timer = setTimeout(
      async () => {
        setMeaning({ key: requestKey, status: 'loading', results: [] });
        try {
          const found = await semanticSearch(day, query, controller.signal);
          if (!stale && !controller.signal.aborted) setMeaning({ key: requestKey, status: 'ready', results: found });
        } catch (error) {
          if (!stale && !controller.signal.aborted)
            setMeaning({ key: requestKey, status: 'error', results: [], error: messageOf(error) });
        }
      },
      explicit ? 0 : SEMANTIC_DELAY_MS,
    );
    return () => {
      stale = true;
      clearTimeout(timer);
      controller.abort();
    };
  }, [wanted, day, query, requestKey, explicit]);

  useEffect(() => {
    if (query && results.length) selectOption(selected);
    else resetSelectedOption();
  }, [query, results.length, selected, selectOption, resetSelectedOption]);

  const clear = () => {
    setText('');
    setActive(0);
    setManual({ key: '', attempt: 0 });
    input.current?.focus();
  };
  const keys = (event: KeyboardEvent<HTMLElement>) => {
    event.stopPropagation();
    if (event.nativeEvent.isComposing || event.keyCode === 229) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      clear();
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      // EventsTarget already handles the input. Result buttons use the same store when tabbed to.
      if (event.target !== input.current) {
        if (event.key === 'ArrowDown') combobox.selectNextOption();
        else combobox.selectPreviousOption();
      }
      setActive(Math.max(0, combobox.getSelectedOptionIndex()));
    } else if (event.key === 'Enter' && event.target === input.current && !results.length && meaningful) {
      event.preventDefault();
      byMeaning();
    }
  };
  const option = (result: { id: string; ts: number }, offset: number, label: string, body: ReactNode) => (
    <Combobox.Option
      key={result.id}
      value={result.id}
      renderRoot={props => <button {...props} type="button" />}
      id={`${id}-${offset}`}
      aria-selected={selected === offset}
      className={classes.hit}
      onMouseEnter={() => setActive(offset)}
      onFocus={() => setActive(offset)}
    >
      <span className={classes.meta}>
        {label} · {hms(result.ts)}
      </span>
      <span>{body}</span>
    </Combobox.Option>
  );

  return (
    <Combobox
      store={combobox}
      onOptionSubmit={value => {
        const result = results.find(result => result.id === value);
        if (result) onSeek(result.ts);
      }}
    >
      <section
        className={classes.search}
        data-active={!!query || undefined}
        data-beside={beside || undefined}
        aria-label="Search this day"
        data-review-search
        onKeyDown={keys}
      >
        <Combobox.EventsTarget withExpandedAttribute aria-activedescendant={activeId}>
          <TextInput
            ref={input}
            maxLength={500}
            type="search"
            label="Search transcript and summary"
            placeholder="Words, a phrase, or a typo"
            value={text}
            disabled={!day}
            onChange={event => {
              setText(event.currentTarget.value.slice(0, 500));
              setActive(0);
              setManual({ key: '', attempt: 0 });
            }}
            aria-autocomplete="list"
          />
        </Combobox.EventsTarget>
        {query && (
          <Group gap="xs">
            <Button size="compact-xs" variant="default" onClick={byMeaning} disabled={!meaningful}>
              Search by meaning
            </Button>
            <Button size="compact-xs" variant="subtle" onClick={clear}>
              Clear search
            </Button>
          </Group>
        )}
        {query && summary !== 'ready' && (
          <p className={classes.notice} role="status">
            {summary === 'error' ? (
              <>
                The day's summary could not be loaded. Only available text is searched.{' '}
                {onRetrySummary && (
                  <Button size="compact-xs" variant="subtle" onClick={onRetrySummary}>
                    Retry summary
                  </Button>
                )}
              </>
            ) : summary === 'partial' ? (
              'The summary is still being written; available summary blocks are included.'
            ) : (
              'Loading the full-day summary; only available text is searched so far.'
            )}
          </p>
        )}
        {query && loading && (
          <Group gap="xs" role="status" aria-live="polite">
            <Loader size="xs" />
            <Text size="sm" c="dimmed">
              Loading this day's text for instant search.
            </Text>
          </Group>
        )}
        {query && wanted && current?.status !== 'ready' && current?.status !== 'error' && (
          <Group gap="xs" role="status" aria-live="polite">
            <Loader size="xs" />
            <Text size="sm" c="dimmed">
              Searching by meaning…
            </Text>
          </Group>
        )}
        {query && wanted && current?.status === 'error' && (
          <LoadError message={`Meaning search failed: ${current.error}`} onRetry={byMeaning} />
        )}
        {query && !meaningful && <p className={classes.notice}>Enter words to search this day.</p>}
        <Combobox.Options hidden={!query} className={classes.results} aria-label="Review search results">
          {!!hits.length && (
            <Combobox.Group label="Text matches" classNames={{ group: classes.group, groupLabel: classes.heading }}>
              {hits.map((hit, offset) =>
                option(
                  hit.document,
                  offset,
                  hit.document.kind === 'summary' ? 'Summary' : 'Transcript',
                  highlighted(hit.document.text, hit.ranges),
                ),
              )}
            </Combobox.Group>
          )}
          {!!similar.length && (
            <Combobox.Group
              label="Similar in meaning"
              classNames={{ group: classes.group, groupLabel: classes.heading }}
            >
              {similar.map((result, offset) =>
                option(
                  { ...result, id: `semantic:${result.id}` },
                  hits.length + offset,
                  'Similar in meaning',
                  result.text,
                ),
              )}
            </Combobox.Group>
          )}
          {!hits.length && !similar.length && !loading && current?.status === 'ready' && (
            <p className={classes.notice}>No matching text or similar moments found.</p>
          )}
        </Combobox.Options>
      </section>
    </Combobox>
  );
}
