// The Ingest tab: bring photos, video, and audio in from files and links, and see what was brought in. Sources
// (network cameras and photo libraries) are set up in Settings (settings/sources/). Everything goes through the same
// pipeline (src/liberational/ingest/): kept by content hash, converted, and registered on Continuum.
import {
  ActionIcon,
  Anchor,
  Badge,
  Box,
  Button,
  FileButton,
  Group,
  Progress,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { IconTrash } from '@tabler/icons-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, type JSX, type ReactNode, useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { HttpError, messageOf } from '../../shared/http';
import { confirm } from '../lib/confirm';
import { dayWords, hm, isoDay, plural } from '../lib/format';
import { PageEmpty, PageError, PageHeader, PageLoading } from '../lib/page';
import { reviewURL } from '../review/url';
import { href, linkClick } from '../router';
import { CARD_ID } from '../settings/sources/model';
import {
  addLink,
  deleteItem,
  getIngest,
  INGEST_KEY,
  type IngestDeletion,
  type IngestItem,
  type IngestView,
  type Job,
  retryItem,
  type SourceInfo,
  uploadFile,
} from './api';
import { Failure, type Message, muted, Status, useRefresh } from './bits';
import classes from './ingest.module.css';

const PENDING = new Set(['queued', 'converting', 'registering', 'running']);
const STATUS: Record<IngestItem['status'], string> = {
  queued: 'Waiting',
  converting: 'Converting',
  registering: 'Adding',
  done: 'Added',
  failed: 'Failed',
};
// Which time a row shows: when it was taken or recorded, from the media itself or its source, else when it
// was added here.
const TIME_WORD: Record<string, string> = {
  exif: 'taken',
  ffprobe: 'recorded',
  source: 'taken',
  upload: 'added',
};

/** A time as the app says it, in 24 hours: "today, 13:09", "Sat 27 Sep, 15:09", and the year when it isn't this
 *  one ("Sat 17 Aug 2019, 14:12"), since media brought in can be years old. */
export function mediaTime(ts: number, now = new Date()): string {
  const d = new Date(ts * 1000);
  const day =
    d.getFullYear() === now.getFullYear()
      ? dayWords(isoDay(d), now)
      : d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
  return `${day}, ${hm(ts)}`;
}

/** Where an item came from, in words: a source by the name the Ingest tab gives it, never its internal id. */
export function fromWords(source: string, sources: readonly SourceInfo[]): string {
  if (source === 'upload') return 'added here';
  if (source === 'link') return 'from a link';
  return `from ${sources.find(s => s.name === source)?.label ?? source}`;
}

/** A job in words ("link <url>" is "Fetching <url>", "poll wyze" is "Checking Wyze cameras"), and the Settings card
 *  a failed one is fixed on, when it's a source's. */
export function jobWords(job: Job, sources: readonly SourceInfo[]): { what: string; card: string | null } {
  if (job.what.startsWith('link ')) return { what: `Fetching ${job.what.slice(5)}`, card: null };
  const polled = job.what.startsWith('poll ') ? sources.find(s => s.name === job.what.slice(5)) : undefined;
  const source = polled ?? sources.find(s => job.what.startsWith(`${s.label}:`));
  if (!source) return { what: job.what, card: null };
  return {
    what: polled ? `Checking ${source.label}` : job.what,
    card: source.name === 'wyze' ? CARD_ID.camera : CARD_ID.media,
  };
}

/** Whether anything is still being worked on, so the view polls quickly until it's done. */
export const busy = (view: IngestView | undefined): boolean =>
  !!view && (view.items.some(i => PENDING.has(i.status)) || view.jobs.some(j => PENDING.has(j.status)));

function useDeleteItem(onDeleted: (name: string) => void) {
  const client = useQueryClient();
  return useCallback(
    async (item: { id: number; name: string }, options: { poll?: boolean; signal?: AbortSignal } = {}) => {
      let reply: Awaited<ReturnType<typeof deleteItem>>;
      try {
        reply = await deleteItem(item.id, options);
      } catch (e) {
        if (e instanceof HttpError && e.status === 409) {
          client.setQueryData<IngestView>(INGEST_KEY, previous =>
            previous
              ? {
                  ...previous,
                  items: previous.items.filter(i => i.id !== item.id),
                  deletions: [
                    ...(previous.deletions ?? []).filter(d => d.id !== item.id),
                    { id: item.id, name: item.name, detail: messageOf(e), blocked: true },
                  ],
                }
              : previous,
          );
          void client.invalidateQueries({ queryKey: INGEST_KEY });
        }
        throw e;
      }
      client.setQueryData<IngestView>(INGEST_KEY, previous => {
        if (!previous) return previous;
        const deletions = (previous.deletions ?? []).filter(d => d.id !== item.id);
        if (reply.pending) deletions.push({ id: item.id, name: item.name, detail: '' });
        return { ...previous, items: previous.items.filter(i => i.id !== item.id), deletions };
      });
      // Cards lack per-item provenance. Clear affected cached evidence so hidden tabs cannot show deleted media
      // while their next read runs; keep the ingest cleanup notice even when refreshing its list fails.
      const affected = new Set(['people', 'faces', 'voices', 'clones', 'review', 'live', 'objects']);
      if (!options.poll || !reply.pending)
        void client.resetQueries({ predicate: query => affected.has(String(query.queryKey[0])) });
      if (!reply.pending) onDeleted(item.name);
      if (!options.poll || !reply.pending) void client.invalidateQueries({ queryKey: INGEST_KEY });
      return reply;
    },
    [client, onDeleted],
  );
}

/** Bring a row into view and focus it, for "Show it" on a file that was already here. */
function showItem(id: number) {
  const row = document.querySelector<HTMLElement>(`[data-item-id="${id}"]`);
  if (!row) return;
  row.scrollIntoView({ block: 'center', behavior: 'smooth' });
  row.focus({ preventScroll: true });
}

const MATCHED_SHOWN = 5;

/** The files an upload found already here, each with the item it matched: "copy.jpg was already here as
 *  beach.jpg", so the owner can see it's the same thing. The first MATCHED_SHOWN, then how many more. */
function Matched({ matched }: { matched: { name: string; item: IngestItem }[] }) {
  return (
    <Stack gap={0} mt={4}>
      {matched.slice(0, MATCHED_SHOWN).map(({ name, item }) => (
        <Text key={`${name}-${item.id}`} size="sm" c="dimmed">
          {name === item.name ? `${name} was already here` : `${name} was already here as ${item.name}`}{' '}
          <Anchor component="button" type="button" size="sm" onClick={() => showItem(item.id)}>
            Show it
          </Anchor>
        </Text>
      ))}
      {matched.length > MATCHED_SHOWN && (
        <Text size="sm" c="dimmed">
          and {plural(matched.length - MATCHED_SHOWN, 'more', 'more')}
        </Text>
      )}
    </Stack>
  );
}

// One file of an upload. A batch is uploaded one file at a time; past one file it shows as a queue, so file 4 of 9 is
// visible and the rest aren't silent. Files dropped while it runs join the end.
type Queued = {
  key: number;
  file: File;
  state: 'waiting' | 'sending' | 'added' | 'here' | 'failed';
  item?: IngestItem;
  err?: string;
};

const QUEUED_WORD: Record<Queued['state'], string> = {
  waiting: 'waiting',
  sending: 'uploading',
  added: 'added',
  here: 'already here',
  failed: 'failed',
};

// What a finished batch came to: "Added 7 files; 1 was already here; 1 failed."
function summary(rows: readonly Queued[]): Message {
  const count = (state: Queued['state']) => rows.filter(r => r.state === state).length;
  const [added, here, failed] = [count('added'), count('here'), count('failed')];
  const only = rows.length === 1 ? rows[0] : null;
  if (only?.state === 'failed') return { err: `${only.file.name}: ${only.err}` };
  const parts = [
    added ? `Added ${plural(added, 'file', 'files')}` : '',
    here
      ? added
        ? `${plural(here, 'was', 'were')} already here`
        : `${here === 1 ? '1 file was' : `${here} files were`} already here`
      : '',
    failed ? `${failed} failed` : '',
  ].filter(Boolean);
  const text = `${parts.join('; ')}.`;
  return failed ? { err: text } : { text };
}

function Queue({ rows, sent }: { rows: readonly Queued[]; sent: number | null }) {
  return (
    <Box component="ol" aria-label="Upload queue" className={classes.queue}>
      {rows.map(r => (
        <li key={r.key} data-state={r.state}>
          <Text size="sm" truncate title={r.file.name} className={classes.queueName}>
            {r.file.name}
          </Text>
          {r.state === 'sending' && sent !== null ? (
            <Progress value={Math.round(sent * 100)} size="sm" aria-label={`${r.file.name} uploaded so far`} w={80} />
          ) : null}
          <Text size="sm" c={r.state === 'failed' ? 'var(--lib-bad)' : 'dimmed'}>
            {r.state === 'failed' && r.err ? `failed: ${r.err}` : QUEUED_WORD[r.state]}
          </Text>
        </li>
      ))}
    </Box>
  );
}

const hasFiles = (e: globalThis.DragEvent) => !!e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files');

function Upload() {
  const refresh = useRefresh();
  const [msg, setMsg] = useState<Message>(null);
  const [sent, setSent] = useState<number | null>(null); // the share of the file being uploaded that's sent
  const [over, setOver] = useState(false);
  const [, rerender] = useReducer((n: number) => n + 1, 0);
  const queue = useRef<Queued[]>([]);
  const running = useRef(false);
  const nextKey = useRef(0);
  const resetFile = useRef<() => void>(null);
  const set = (key: number, change: Partial<Queued>) => {
    queue.current = queue.current.map(r => (r.key === key ? { ...r, ...change } : r));
    rerender();
  };
  const send = async (files: File[]) => {
    if (!files.length) return;
    if (!running.current) queue.current = []; // a new batch starts a new queue
    queue.current = [
      ...queue.current,
      ...files.map(file => ({ key: nextKey.current++, file, state: 'waiting' as const })),
    ];
    rerender();
    if (running.current) return; // the running loop picks these up
    running.current = true;
    const next = () => queue.current.find(r => r.state === 'waiting');
    for (let row = next(); row; row = next()) {
      const at = queue.current.indexOf(row) + 1;
      set(row.key, { state: 'sending' });
      setMsg({ text: `Uploading ${row.file.name} (${at} of ${queue.current.length})` });
      setSent(0);
      try {
        const reply = await uploadFile(row.file, setSent); // one request per file: the body is the file itself
        set(row.key, { state: reply.new ? 'added' : 'here', item: reply.item });
      } catch (e) {
        set(row.key, { state: 'failed', err: messageOf(e) }); // the rest still go
      }
    }
    running.current = false;
    setSent(null);
    setMsg(summary(queue.current));
    void refresh();
  };
  const sendRef = useRef(send);
  sendRef.current = send;
  // A drop anywhere on the page adds the files, with the whole page outlined while dragging.
  useEffect(() => {
    let depth = 0; // dragenter and dragleave fire for every element crossed
    const enter = (e: globalThis.DragEvent) => {
      if (!hasFiles(e)) return;
      depth++;
      setOver(true);
    };
    const leave = (e: globalThis.DragEvent) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (!depth) setOver(false);
    };
    const dragOver = (e: globalThis.DragEvent) => {
      if (hasFiles(e)) e.preventDefault(); // without this the browser opens the file instead
    };
    const drop = (e: globalThis.DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setOver(false);
      void sendRef.current(Array.from(e.dataTransfer?.files ?? []));
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('dragover', dragOver);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('dragover', dragOver);
      window.removeEventListener('drop', drop);
    };
  }, []);
  const rows = queue.current;
  const matched = rows.flatMap(r => (r.state === 'here' && r.item ? [{ name: r.file.name, item: r.item }] : []));
  return (
    <Box className={classes.drop} data-over={over || undefined}>
      {over && (
        <div className={classes.page} aria-hidden="true">
          <span>Drop to add</span>
        </div>
      )}
      <Text fw={600}>Add media</Text>
      <Text {...muted}>
        Drop photos, video, or audio anywhere on this page, choose files, or paste a link. Each is kept once, however
        often it's added.
      </Text>
      <Group mt="xs">
        <FileButton
          multiple
          accept="image/*,video/*,audio/*"
          inputProps={{ 'aria-label': 'Choose files' }}
          resetRef={resetFile}
          onChange={files => {
            void send(files);
            resetFile.current?.();
          }}
        >
          {props => (
            <Button {...props} variant="default" size="xs">
              Choose files
            </Button>
          )}
        </FileButton>
      </Group>
      <Status msg={msg} />
      {rows.length > 1 && <Queue rows={rows} sent={sent} />}
      {!running.current && !!matched.length && <Matched matched={matched} />}
      {rows.length === 1 && sent !== null && (
        <Progress
          value={Math.round(sent * 100)}
          size="sm"
          mt={4}
          aria-label="Uploaded so far"
          className={classes.bar}
        />
      )}
      {/* One card, one entry point: files and links both start here. */}
      <Link />
    </Box>
  );
}

function Link() {
  const refresh = useRefresh();
  const [url, setUrl] = useState('');
  const [msg, setMsg] = useState<Message>(null);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    try {
      await addLink(url.trim());
      setMsg({ text: 'Fetching it in the background.' });
      setUrl('');
    } catch (err) {
      setMsg({ err: messageOf(err) });
    }
    void refresh();
  };
  return (
    <Box component="form" onSubmit={submit} mt="sm">
      <Group align="end" wrap="nowrap">
        <TextInput
          label="Add from a link"
          description="A link to a photo, video, or audio file, or a page with a video (read with yt-dlp)"
          placeholder="https://"
          value={url}
          onChange={e => setUrl(e.currentTarget.value)}
          style={{ flex: 1 }}
        />
        <Button type="submit" variant="default">
          Add
        </Button>
      </Group>
      <Status msg={msg} />
    </Box>
  );
}

function ItemRow({
  item,
  sources,
  onDeleted,
}: {
  item: IngestItem;
  sources: readonly SourceInfo[];
  onDeleted: (name: string) => void;
}) {
  const refresh = useRefresh();
  const removeItem = useDeleteItem(onDeleted);
  const [msg, setMsg] = useState<Message>(null);
  const [deleting, setDeleting] = useState(false);
  const status = STATUS[item.status];
  const remove = async () => {
    if (
      deleting ||
      !(await confirm({
        title: `Delete ${item.name}`,
        body: "Delete this and everything found in it? This can't be undone.",
        confirm: 'Delete',
        danger: true,
      }))
    )
      return;
    setDeleting(true);
    setMsg({ text: 'Deleting…' });
    try {
      await removeItem(item);
    } catch (e) {
      setMsg({ err: messageOf(e) });
    } finally {
      setDeleting(false);
    }
  };
  return (
    <Box component="li" className={classes.item} data-item-id={item.id} tabIndex={-1}>
      <Stack gap={0} style={{ minWidth: 0 }}>
        {/* Two lines at most, and the whole name on hover: a link's address was nothing but its start. */}
        <Text fw={600} lineClamp={2} title={item.name} className={classes.name}>
          {item.name}
        </Text>
        <Text {...muted}>
          {[
            item.kind ?? 'file',
            // An upload with no time of its own says "added today, 13:09" once, not "added here, added …".
            item.source === 'upload' && item.time_from === 'upload' && item.original_ts
              ? null
              : fromWords(item.source, sources),
            item.original_ts ? `${TIME_WORD[item.time_from ?? ''] ?? 'taken'} ${mediaTime(item.original_ts)}` : null,
          ]
            .filter(Boolean)
            .join(', ')}
        </Text>
        {item.status === 'done' && <Found item={item} />}
        {PENDING.has(item.status) && (
          // Still coming in: the pipeline says which step, not how far.
          <Progress value={100} striped animated size="xs" mt={4} aria-label={`${status}…`} className={classes.bar} />
        )}
        {item.detail &&
          (item.status === 'failed' ? (
            <Failure detail={item.detail} />
          ) : (
            <Text size="sm" c="dimmed">
              {item.detail}
            </Text>
          ))}
        <Status msg={msg} />
      </Stack>
      <Group gap="xs" wrap="nowrap">
        <Badge
          variant="light"
          color={item.status === 'done' ? 'green' : item.status === 'failed' ? 'red' : 'blue'}
          aria-label={`Status: ${status}`}
        >
          {status}
        </Badge>
        {item.status === 'failed' && (
          <Button
            size="xs"
            variant="subtle"
            disabled={deleting}
            onClick={async () => {
              setMsg(null);
              try {
                await retryItem(item.id);
              } catch (e) {
                setMsg({ err: messageOf(e) });
              }
              void refresh();
            }}
          >
            Retry
          </Button>
        )}
        {/* Quiet until it's meant: a red Delete on every row was the loudest thing in the list. It still asks. */}
        <ActionIcon
          variant="subtle"
          color="gray"
          className={classes.delete}
          aria-label={`Delete ${item.name}`}
          title="Delete"
          loading={deleting}
          onClick={() => void remove()}
        >
          <IconTrash size={16} aria-hidden />
        </ActionIcon>
      </Group>
    </Box>
  );
}

/** Polls only a previously requested deletion, including after reload; ordinary GET listings remain read-only. */
function CleanupRow({ item, onDeleted }: { item: IngestDeletion; onDeleted: (name: string) => void }) {
  const removeItem = useDeleteItem(onDeleted);
  const [attempt, setAttempt] = useState(0);
  const [phase, setPhase] = useState<'waiting' | 'failed' | 'expired' | 'blocked'>(
    item.blocked ? 'blocked' : 'waiting',
  );
  const [msg, setMsg] = useState<Message>(null);
  const { id, name } = item;
  const blocked = !!item.blocked;
  useEffect(() => {
    if (blocked) {
      setPhase('blocked');
      setMsg(null);
      return;
    }
    let stopped = false;
    let next: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    setPhase('waiting');
    setMsg(null);
    const deadline = setTimeout(() => {
      stopped = true;
      clearTimeout(next);
      controller.abort();
      setPhase('expired');
    }, 120_000);
    const poll = async () => {
      try {
        const reply = await removeItem({ id, name }, { poll: true, signal: controller.signal });
        if (stopped) return;
        if (reply.pending) next = setTimeout(() => void poll(), 2000);
        else clearTimeout(deadline);
      } catch (e) {
        if (stopped) return;
        clearTimeout(deadline);
        setPhase(e instanceof HttpError && e.status === 409 ? 'blocked' : 'failed');
        setMsg({ err: messageOf(e) });
      }
    };
    next = setTimeout(() => void poll(), attempt ? 0 : 2000);
    return () => {
      stopped = true;
      clearTimeout(next);
      clearTimeout(deadline);
      controller.abort();
    };
  }, [attempt, blocked, id, name, removeItem]);
  return (
    <Box component="li" className={classes.item}>
      <Stack gap={0} style={{ minWidth: 0 }}>
        <Text fw={600} truncate>
          {item.name}
        </Text>
        <Text
          {...muted}
          c={phase === 'blocked' ? 'var(--lib-bad)' : 'dimmed'}
          role={phase === 'blocked' ? 'alert' : undefined}
        >
          {phase === 'blocked'
            ? 'Deletion needs manual attention.'
            : phase === 'waiting'
              ? 'Deleting…'
              : 'Deleted from the list, still cleaning up storage.'}
        </Text>
        {phase !== 'waiting' && item.detail && <Text {...muted}>{item.detail}</Text>}
        <Status msg={phase === 'blocked' ? null : msg} />
      </Stack>
      {phase !== 'waiting' && phase !== 'blocked' && (
        <Button
          size="xs"
          variant="subtle"
          aria-label={`Retry cleanup for ${item.name}`}
          onClick={() => setAttempt(a => a + 1)}
        >
          Retry
        </Button>
      )}
    </Box>
  );
}

function Jobs({ jobs, sources }: { jobs: Job[]; sources: readonly SourceInfo[] }) {
  const shown = jobs.filter(j => j.status !== 'done').slice(0, 5);
  if (!shown.length) return null;
  return (
    <Stack gap={2} mt="xs">
      {shown.map(j => {
        const { what, card } = jobWords(j, sources);
        // A link's failure in plain words; a source's own message already says what to do.
        if (j.status === 'failed' && j.what.startsWith('link '))
          return (
            <div key={j.id}>
              <Text size="sm" c="var(--lib-bad)">
                {what}:
              </Text>
              <Failure detail={j.error} />
            </div>
          );
        return (
          <Text key={j.id} size="sm" c={j.status === 'failed' ? 'var(--lib-bad)' : 'dimmed'}>
            {what}: {j.status === 'failed' ? j.error : j.status === 'running' ? 'working' : 'waiting'}
            {/* A source's failure is fixed in Settings, where it signs in or is set up. */}
            {j.status === 'failed' && card && (
              <>
                {' '}
                <Anchor href={`${href('/settings')}#${card}`} onClick={linkClick} inherit>
                  Fix it in Settings
                </Anchor>
              </>
            )}
          </Text>
        );
      })}
    </Stack>
  );
}

/** What came of an item: what reading it found, and a way to its moment in Review when lines were heard.
 *  Review's transcript shows a brought-in item's lines at its time; its camera lanes don't show the item's video, so
 *  without lines there's nothing there to open. */
function Found({ item }: { item: IngestItem }) {
  const a = item.analysis;
  if (!a) return null;
  if (a.state === 'waiting') return <Text {...muted}>Reading it for words and pictures…</Text>;
  if (a.state === 'retrying') return <Text {...muted}>Couldn't read it this time; trying again soon.</Text>;
  if (a.state === 'failed')
    return (
      <Text size="sm" c="var(--lib-bad)">
        Couldn't read it{a.error ? `: ${a.error}` : '.'}
      </Text>
    );
  const found = [
    a.captions ? plural(a.captions, 'caption', 'captions') : '',
    a.lines ? `${plural(a.lines, 'line', 'lines')} heard` : '',
  ].filter(Boolean);
  const at = item.original_ts;
  return (
    <Text {...muted}>
      {found.length ? found.join(' · ') : 'Nothing in it to describe or hear'}
      {a.lines > 0 && at != null && (
        <>
          {' · '}
          <Anchor href={reviewURL({ day: isoDay(new Date(at * 1000)), t: at })} onClick={linkClick} inherit>
            Open in Review
          </Anchor>
        </>
      )}
    </Text>
  );
}

const ADDED_SHOWN = 10; // added items shown before "Show all"

/** One group of the list: a small heading with its count, then its rows. */
function ListGroup({ title, count, children }: { title: string; count: number; children: ReactNode }) {
  return (
    <Box component="section" aria-label={title}>
      <Title order={3} className={classes.groupTitle}>
        {title}{' '}
        <Text span c="dimmed" inherit fw={400}>
          {count}
        </Text>
      </Title>
      {children}
    </Box>
  );
}

/** The list grouped by what needs you: what's still coming in, what needs attention (a failure, or a deletion
 *  still cleaning up), then what was added, newest first, the first ADDED_SHOWN until "Show all". A failure can't
 *  scroll out of sight below a day's photos. */
function Groups({ view, onDeleted }: { view: IngestView; onDeleted: (name: string) => void }) {
  const [all, setAll] = useState(false);
  const sources = view.sources ?? [];
  const working = view.items.filter(i => PENDING.has(i.status));
  const failed = view.items.filter(i => i.status === 'failed');
  const added = view.items.filter(i => i.status === 'done');
  const deletions = view.deletions ?? [];
  const shown = all ? added : added.slice(0, ADDED_SHOWN);
  const row = (i: IngestItem) => <ItemRow key={i.id} item={i} sources={sources} onDeleted={onDeleted} />;
  const attention = failed.length + deletions.length;
  const summary = [working.length ? `${working.length} working` : '', attention ? `${attention} need attention` : '']
    .filter(Boolean)
    .join(' · ');
  return (
    <Stack gap="md">
      {summary && (
        // What's going on, at a glance. Not a live region: it changes with every poll, and each row says its own.
        <Text size="sm" c="dimmed">
          {summary}
        </Text>
      )}
      {!!working.length && (
        <ListGroup title="In progress" count={working.length}>
          <Box component="ul" className={classes.list}>
            {working.map(row)}
          </Box>
        </ListGroup>
      )}
      {!!(failed.length || deletions.length) && (
        <ListGroup title="Needs attention" count={failed.length + deletions.length}>
          {!!failed.length && (
            <Box component="ul" className={classes.list}>
              {failed.map(row)}
            </Box>
          )}
          {!!deletions.length && (
            <Box component="ul" className={classes.list} aria-label="Deleted items awaiting cleanup">
              {deletions.map(item => (
                <CleanupRow key={item.id} item={item} onDeleted={onDeleted} />
              ))}
            </Box>
          )}
        </ListGroup>
      )}
      {!!added.length && (
        <ListGroup title="Added" count={added.length}>
          <Box component="ul" className={classes.list}>
            {shown.map(row)}
          </Box>
          {added.length > ADDED_SHOWN && (
            <Button variant="subtle" size="xs" mt={4} onClick={() => setAll(a => !a)}>
              {all ? 'Show fewer' : `Show all ${added.length}`}
            </Button>
          )}
        </ListGroup>
      )}
    </Stack>
  );
}

export function IngestTab(): JSX.Element {
  const [deleted, setDeleted] = useState<string | null>(null);
  const onDeleted = useCallback((name: string) => setDeleted(name), []);
  const query = useQuery({
    queryKey: INGEST_KEY,
    queryFn: ({ signal }) => getIngest(signal),
    refetchInterval: q => (busy(q.state.data) ? 2000 : 30000),
  });
  const view = query.data;
  return (
    <section aria-labelledby="ingest-h">
      <PageHeader
        titleId="ingest-h"
        title="Ingest"
        lede="Photos, video, and audio from elsewhere, read for faces, voices, and words like the ones recorded here."
      />
      <Stack gap="lg">
        <Upload />
        <Text {...muted}>
          To bring media in on its own, set up{' '}
          <Anchor href={`${href('/settings')}#${CARD_ID.camera}`} onClick={linkClick} inherit>
            network cameras
          </Anchor>{' '}
          like Wyze or{' '}
          <Anchor href={`${href('/settings')}#${CARD_ID.media}`} onClick={linkClick} inherit>
            photo libraries
          </Anchor>{' '}
          like Google Photos in Settings.
        </Text>
        <Box>
          <Title order={2}>Brought in</Title>
          {query.isError && !view && (
            <PageError
              title="Couldn't load what was brought in"
              detail={messageOf(query.error)}
              onRetry={() => void query.refetch()}
              busy={query.isFetching}
            />
          )}
          {view && <Jobs jobs={view.jobs} sources={view.sources ?? []} />}
          {deleted && (
            <Text {...muted} role="status">
              Deleted {deleted}.
            </Text>
          )}
          {query.isPending ? (
            <PageLoading label="what was brought in" shape="rows" lines={3} />
          ) : !view?.items.length && !view?.deletions?.length ? (
            !query.isError && (
              <PageEmpty title="Nothing brought in yet" compact>
                Photos, video, and audio you bring in show up here.
              </PageEmpty>
            )
          ) : (
            view && <Groups view={view} onDeleted={onDeleted} />
          )}
        </Box>
      </Stack>
    </section>
  );
}
